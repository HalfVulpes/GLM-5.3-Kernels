# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for optional decode integration; never imports vLLM."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name,ROOT/relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class DecodeContracts(unittest.TestCase):
    def test_runtime_bytes_and_pointer_only_derivation(self):
        manifest = json.loads((ROOT/'integration/decode-manifest.json').read_text())
        raw = (ROOT/'kernels/titan_kda_strided.py').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),manifest['runtime_modules']['titan_kda_strided.py'])
        contract = load('public_kda_contract','integration/kda_strided_contract.py')
        _, body = contract.function_source(raw.decode(),'_titan_kda_strided_kernel')
        original = contract.recover_original_kernel(body)
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),contract.BASE_KERNEL_SHA256)

    def test_only_optional_module_imports_vllm(self):
        tree = ast.parse((ROOT/'kernels/titan_kda_strided.py').read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertIn('vllm.triton_utils',imports)
        self.assertIn('vllm.third_party.flash_linear_attention.ops.op',imports)
        for node in tree.body:
            # Module scope contains definitions/imports/constants; no device
            # queries or allocation calls are made by importing this module.
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
                self.fail('Unexpected top-level call')

    def test_paths_reject_parent_absolute_and_symlink(self):
        installer = load('public_decode_installer','integration/apply_decode.py')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'outside').mkdir(); (root/'link').symlink_to(root/'outside',target_is_directory=True)
            for name in ('../bad','/absolute','a/../bad','a//bad','link/file'):
                with self.assertRaises(ValueError,msg=name): installer.contained(root,name)
            self.assertEqual(installer.contained(root,'a/b.py'),root/'a/b.py')

    def test_installer_rolls_back_completed_writes(self):
        installer = load('public_decode_rollback','integration/apply_decode.py')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); planned=[]
            for name in ('a.py','b.py'):
                path=root/name; path.write_bytes(b'old = 1\n')
                planned.append({'relative':name,'path':path,'before':path.read_bytes(),
                                'after':b'new = 2\n','mode':0o640})
            original = installer.atomic_write
            def injected(path,raw,mode):
                if path.name=='b.py' and raw==b'new = 2\n': raise OSError('injected second write failure')
                return original(path,raw,mode)
            with patch.object(installer,'atomic_write',side_effect=injected):
                with self.assertRaises(OSError): installer.apply_planned(root,planned,{'source_state':'original'})
            for row in planned: self.assertEqual(row['path'].read_bytes(),row['before'])
            self.assertFalse((root/installer.STATE).exists())
            self.assertTrue((root/installer.BACKUP/'a.py').is_file())

    def test_applied_sources_are_not_rewritten(self):
        installer = load('public_decode_idempotence','integration/apply_decode.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(installer,'atomic_write',side_effect=AssertionError('unexpected write')):
                self.assertFalse(installer.apply_planned(root,[],{'source_state':'candidate'}))
            self.assertEqual(list(root.iterdir()),[])

    def test_unknown_source_fails_before_transform(self):
        router = load('public_router_transform','integration/patch_router_gate.py')
        with self.assertRaises(ValueError): router.transform(str(router.MODEL),'# drift\n')
        with self.assertRaises(ValueError): router.transform(str(router.RUNNER),'# drift\n')

    def test_both_padding_transforms_reject_drift(self):
        for name in ('kda_padding','moe_padding'):
            module=load('public_'+name,'integration/patch_'+name+'.py')
            with self.assertRaises(ValueError): module.apply_candidate('# unrecognized source\n')

    def test_final_profile_has_all_four_components_and_targets(self):
        manifest=json.loads((ROOT/'integration/decode-manifest.json').read_text())
        self.assertEqual(manifest['profile'],['router_gate_v3','kda_strided','kda_padding','moe_padding'])
        self.assertEqual(len(manifest['files']),4)
        self.assertIn('models/glm5next/nvidia/kda.py',manifest['files'])
        self.assertEqual(len(manifest['unchanged_companions']),3)


if __name__ == '__main__': unittest.main()

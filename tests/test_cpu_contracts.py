# SPDX-License-Identifier: Apache-2.0
"""Public default test suite: standard library only, no GPU execution."""
import importlib.util
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class CPUContracts(unittest.TestCase):
    def test_moe_layout_oracle(self):
        module = load_file("public_moe_benchmark", ROOT / "benchmarks/moe_align.py")
        report = module.cpu_checks()
        self.assertTrue(report["cpu_passed"])
        self.assertEqual(report["cases"], 856)
        self.assertFalse(report["gpu_tested"])

    def test_core_has_no_vllm_imports(self):
        checker = load_file("public_source_checker", ROOT / "tests/verify_source_equivalence.py")
        self.assertEqual(len(checker.check_core_imports(ROOT / "kernels")), 3)

    def test_core_import_does_not_query_cuda(self):
        # Execute definitions/decorators with dependency doubles. This detects
        # import-time CUDA calls without requiring torch/triton installation;
        # it is explicitly not a real dependency/runtime import test.
        fake_torch = types.ModuleType("torch"); fake_torch.Tensor = object
        class NoCUDA:
            def __getattr__(self, name):
                raise AssertionError(f"Import-time torch.cuda access: {name}")
        fake_torch.cuda = NoCUDA()
        fake_triton = types.ModuleType("triton"); fake_triton.__path__ = []
        fake_language = types.ModuleType("triton.language"); fake_language.constexpr = object
        fake_triton.language = fake_language
        fake_triton.Config = lambda values, **kwargs: types.SimpleNamespace(kwargs=values, **kwargs)
        fake_triton.jit = lambda function: function
        fake_triton.autotune = lambda **kwargs: lambda function: function
        with patch.dict(sys.modules, {"torch": fake_torch, "triton": fake_triton, "triton.language": fake_language}):
            for filename, exported in (("titan_kv.py", "store_int8"),
                                       ("titan_mla.py", "triton_mla_sparse_attention"),
                                       ("titan_moe_align.py", "titan_moe_align_small")):
                module = load_file("isolated_" + filename.removesuffix(".py"), ROOT / "kernels" / filename)
                self.assertTrue(callable(getattr(module, exported)))

    def test_gpu_entrypoints_default_to_cpu(self):
        for script in ("tests/verify_mla.py", "benchmarks/moe_align.py", "benchmarks/mla.py"):
            result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                                    capture_output=True, text=True, check=True)
            self.assertIs(json.loads(result.stdout)["gpu_tested"], False, script)

    def test_integration_tests_require_explicit_source(self):
        for script in ("tests/verify_moe_dispatch.py", "tests/verify_indexer_sharding.py"):
            result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, script)
            self.assertIn("required", result.stderr)


if __name__ == "__main__": unittest.main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU source preflight against a caller-supplied pinned vLLM package tree."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def verify_padding_guards(planned):
    """Execute actual transformed guard ASTs with CPU indexing doubles.

    This checks slices, active NaN preservation and fallback control flow;
    it is not a PyTorch/Triton or GPU numerical comparison.
    """
    sources = {row['relative']: row['after'].decode() for row in planned}
    model = sources['models/glm5next/nvidia/model.py']
    prior = model.replace('import vllm.envs as envs\n','',1).replace(
        'if envs.VLLM_MOE_SKIP_PADDING and not self.is_sequence_parallel',
        'if not self.is_sequence_parallel',1)
    prior_sha = hashlib.sha256(prior.encode()).hexdigest()
    if prior_sha != '7b57f99a64f1399f3b1d0c4273e8e41ca5ebc375edd676c7da5a74263f0134ca':
        raise ValueError('Environment-guard removal does not recover the tested v2 model')
    def guard(name, expression):
        nodes = [node for node in ast.walk(ast.parse(sources[name]))
                 if isinstance(node,ast.If) and ast.unparse(node.test)==expression]
        if len(nodes)!=1: raise ValueError('Expected one pinned padding guard: '+name)
        return compile(ast.Module(body=nodes,type_ignores=[]),name,'exec')
    kda = guard('models/glm5next/nvidia/kda.py','core_attn_out.shape[1] > num_actual_tokens')
    moe = guard('models/glm5next/nvidia/model.py',
                'envs.VLLM_MOE_SKIP_PADDING and (not self.is_sequence_parallel) and is_forward_context_available()')
    class Buffer:
        def __init__(self,total,actual):
            self.shape=(1,total,16,128);self.values=[1.0]*actual+[float('nan')]*(total-actual)
        def __getitem__(self,key):
            assert key[0]==slice(None)
            start=key[1].start; assert key[1].stop is None
            parent=self
            class Suffix:
                def zero_(self): parent.values[start:]=[0.0]*(len(parent.values)-start)
            return Suffix()
    for total,actual in ((8,8),(2304,2303),(8,0),(16,1)):
        buffer=Buffer(total,actual)
        exec(kda,{'core_attn_out':buffer,'num_actual_tokens':actual})
        assert buffer.values==[1.0]*actual+[0.0]*(total-actual)
    class Mask:
        def __getitem__(self,key):
            assert key[1] is None
            return [False,True,False,True][key[0]]
    class Matrix:
        shape=(3,2)
        def __init__(self): self.rows=[[float('nan'),-2.0],[float('nan'),float('inf')],[3.0,4.0]]
        def masked_fill_(self,mask,value):
            assert len(mask)==len(self.rows)
            for i,selected in enumerate(mask):
                if selected:self.rows[i]=[value]*2
    for enabled,sp,context,hasmask in ((True,False,True,True),(True,True,True,True),
                                     (True,False,False,True),(True,False,True,False),
                                     (False,False,True,True)):
        matrix=Matrix()
        def get_context():
            assert context and enabled
            return SimpleNamespace(is_padding=Mask() if hasmask else None)
        exec(moe,{'self':SimpleNamespace(is_sequence_parallel=sp),'hidden_states':matrix,
                  'envs':SimpleNamespace(VLLM_MOE_SKIP_PADDING=enabled),
                  'is_forward_context_available':lambda:context,'get_forward_context':get_context})
        assert math.isnan(matrix.rows[0][0]) and matrix.rows[0][1]==-2.0
        assert matrix.rows[2]==[3.0,4.0]
        if enabled and not sp and context and hasmask: assert matrix.rows[1]==[0,0]
        else: assert math.isnan(matrix.rows[1][0]) and math.isinf(matrix.rows[1][1])
    return {'cpu_indexing_double_cases_passed':9,'active_nan_preserved':True,
            'disabled_skip_padding_does_not_access_mask':True,
            'tested_v2_model_recovered_sha256':prior_sha,
            'v3_change_is_environment_guard_only':True,
            'scope':'Actual transformed guard ASTs with standard-library tensor/indexing doubles; no torch/CUDA execution'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-root',type=Path,required=True)
    parser.add_argument('--exercise-installer',action='store_true',
                        help='Apply only to a disposable copy; original source stays untouched')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('decode_installer',ROOT/'integration/apply_decode.py')
    installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)
    planned,summary=installer.preflight(args.vllm_root)
    report={'cpu_passed':True,'gpu_tested':False,'source_preflight_passed':True,
            'final_profile_output_hashes_match':True,'source_revision':summary['revision'],
            'source_state':summary['source_state'],'temporary_install_and_idempotence_tested':False,
            'verified_source_files':len(summary['files'])+len(summary['unchanged_companions']),
            'padding_guards':verify_padding_guards(planned)}
    if args.exercise_installer:
        if summary['source_state']!='original': raise ValueError('Temporary install test requires original source')
        with tempfile.TemporaryDirectory(prefix='glm53-decode-test-') as temp:
            root=Path(temp)
            for name in list(summary['files'])+list(summary['unchanged_companions']):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(args.vllm_root/name,path)
            candidate,receipt=installer.preflight(root)
            assert installer.apply_planned(root,candidate,receipt)
            candidate,receipt=installer.preflight(root)
            assert receipt['source_state']=='candidate'
            assert not installer.apply_planned(root,candidate,receipt)
            report['temporary_install_and_idempotence_tested']=True
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

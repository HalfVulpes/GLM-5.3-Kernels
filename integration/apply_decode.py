#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optional router/KDA/padding integration for pinned public vLLM source.

Default: preflight only; no target writes. --apply requires an offline/stopped
tree. No image, model weights, GPU, network or service operations are used.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

HERE = Path(__file__).resolve().parent
STATE = '.glm53-decode-applied.json'
BACKUP = '.glm53-decode-original'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def contained(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(p in ('', '.', '..') for p in relative.split('/')):
        raise ValueError(f'Unsafe relative path: {relative}')
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink(): raise ValueError(f'Symlink not accepted: {relative}')
    if not path.resolve().is_relative_to(root.resolve()): raise ValueError('Path escapes package root')
    return path


def load_transforms():
    # These standard-library-only modules never import the runtime/GPU module.
    import sys
    if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
    from patch_router_gate import transform
    from patch_kda_strided import apply_candidate
    return transform, apply_candidate


def preflight(root, manifest=None):
    manifest = manifest or json.loads((HERE/'decode-manifest.json').read_text())
    if root.is_symlink() or not root.is_dir(): raise ValueError('Expected a real vLLM package directory')
    root = root.resolve()
    for name, expected in manifest['runtime_modules'].items():
        path = contained(HERE.parent/'kernels',name)
        if digest(path.read_bytes()) != expected: raise ValueError('Runtime payload drift: '+name)
        ast.parse(path.read_bytes(),filename=name)
    for name, expected in manifest['unchanged_companions'].items():
        if digest(contained(root,name).read_bytes()) != expected: raise ValueError('Companion drift: '+name)
    states = {}
    originals = {}
    for name, hashes in manifest['files'].items():
        path = contained(root,name)
        raw = path.read_bytes(); value = digest(raw)
        if value == hashes['before_sha256']: state = 'original'
        elif value == hashes['after_sha256']: state = 'candidate'
        else: raise ValueError('Pinned source drift: '+name)
        states[name] = state; originals[name] = raw
    if len(set(states.values())) != 1: raise ValueError('Mixed original/candidate files; recover the offline tree first')
    state = next(iter(states.values()))
    router, kda = load_transforms()
    planned = []
    for name, raw in originals.items():
        if state == 'candidate':
            changed = raw
        elif name == 'third_party/flash_linear_attention/ops/kda.py':
            changed = kda(raw.decode()).encode()
        elif name == 'models/glm5next/nvidia/kda.py':
            from patch_kda_padding import apply_candidate as kda_padding
            changed = kda_padding(raw.decode()).encode()
        else:
            changed = router(name,raw.decode())[0].encode()
            if name == 'models/glm5next/nvidia/model.py':
                from patch_moe_padding import apply_candidate as moe_padding
                changed = moe_padding(changed.decode()).encode()
        if digest(changed) != manifest['files'][name]['after_sha256']:
            raise ValueError('Transformation does not reproduce tested source: '+name)
        ast.parse(changed,filename=name)
        path = contained(root,name)
        planned.append({'relative':name,'path':path,'before':raw,'after':changed,
                        'mode':stat.S_IMODE(path.stat().st_mode)})
    summary = {'schema':1,'revision':manifest['revision'],'profile':manifest['profile'],
               'source_state':state,'files':manifest['files'],
               'unchanged_companions':manifest['unchanged_companions'],
               'runtime_modules':manifest['runtime_modules'],
               'qualification':'See docs/decode-qualification.json; source preflight is not model qualification',
               'installer_sha256':digest(Path(__file__).read_bytes())}
    return planned, summary


def atomic_write(path, raw, mode):
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temp,mode)
        os.replace(temp,path)
    finally:
        Path(temp).unlink(missing_ok=True)


def apply_planned(root, planned, summary):
    state = contained(root,STATE); backup = contained(root,BACKUP)
    if summary['source_state'] == 'candidate':
        # Idempotence is based on all target/companion hashes, including changes
        # installed by another offline builder; never rewrite matching sources.
        return False
    if state.exists() or backup.exists():
        raise ValueError('Previous decode state/backup exists; inspect or restore before applying')
    for row in planned:
        if row['path'].read_bytes() != row['before']: raise ValueError('Source changed during preflight')
    backup.mkdir()
    written = []
    try:
        for row in planned:
            path = contained(backup,row['relative']); path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream: stream.write(row['before'])
            os.chmod(path,row['mode'])
        for row in planned:
            if row['path'].read_bytes() != row['before']: raise ValueError('Source changed before write')
            atomic_write(row['path'],row['after'],row['mode']); written.append(row)
        atomic_write(state,(json.dumps(summary,indent=2)+'\n').encode(),0o644)
    except BaseException:
        # Keep the exclusive backup as a recovery record even after rollback.
        for row in reversed(written): atomic_write(row['path'],row['before'],row['mode'])
        raise
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-root',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    planned, summary = preflight(args.vllm_root)
    applied = apply_planned(args.vllm_root.resolve(),planned,summary) if args.apply else False
    print(json.dumps({'preflight_passed':True,'applied':applied,
                     'target_writes_requested':args.apply,**summary},indent=2))


if __name__ == '__main__': main()

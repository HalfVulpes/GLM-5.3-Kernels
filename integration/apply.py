#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preflight the complete kernel integration in a temporary tree before writing.

Default: check only. --apply writes an offline source/install tree and backups.
Does not import vLLM, load model weights, execute CUDA, or restart any service.
"""
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = '.glm53-kernels-applied.json'
BACKUP = '.glm53-kernels-original'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-root', required=True, type=Path,
                        help='The vllm package directory, not its parent repository')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.vllm_root.resolve()
    pinned = json.loads((HERE / 'upstream-manifest.json').read_text())
    files = pinned['files_sha256']
    payload = {p.name: digest(p) for p in HERE.glob('*.py')}
    state_path = root / STATE
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('integration_sha256') != payload:
            raise SystemExit('Integration scripts changed; start from a fresh pinned source tree.')
        if any(digest(root / name) != value for name, value in state['after_sha256'].items()):
            raise SystemExit('Previously patched source was modified; refusing to overwrite it.')
        print('Already applied; installed source hashes match.')
        return
    for name, expected in files.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise SystemExit(f'Pristine pinned source required: {name}')
    if (root / BACKUP).exists():
        raise SystemExit('A previous backup exists without a matching state manifest; inspect it first.')
    before = {name: (root / name).read_bytes() for name in files}
    with tempfile.TemporaryDirectory(prefix='glm53-kernels-') as temp:
        stage = Path(temp)
        for name, data in before.items():
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        from patch_int8_layout import patch_tree as layout
        from patch_indexer_workspace import patch_tree as workspace
        layout(stage)
        workspace(stage)
        for script, extra in [('patch_indexer_sharding.py', []),
                              ('patch_decode_indexer.py', ['--mode', 'slim-query8']),
                              ('patch_moe_align.py', [])]:
            result = subprocess.run([sys.executable, str(HERE / script),
                                     '--root', str(stage), '--apply', *extra],
                                    capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(f'{script} preflight failed:\n{result.stdout}\n{result.stderr}')
        shutil.copyfile(HERE / 'titan_backend.py',
                        stage / 'v1/attention/backends/mla/triton_mla_sparse.py')
        notice = '# Modified by GLM-5.3-Kernels: specialized SM80 MLA/indexer/MoE integration.\n'
        after = {name: (notice + (stage / name).read_text()).encode() for name in files}
        for name, data in after.items():
            ast.parse(data, filename=name)
    summary = {'revision': pinned['revision'], 'changed_files': list(files),
               'before_sha256': files,
               'after_sha256': {n: hashlib.sha256(d).hexdigest() for n, d in after.items()},
               'integration_sha256': payload}
    if args.apply:
        # No worker should be using this tree: cross-file replacement is not
        # atomic to concurrent readers. All transformations were checked above.
        if any((root / n).read_bytes() != data for n, data in before.items()):
            raise SystemExit('Source changed during preflight; no files written.')
        backup = root / BACKUP
        for name, data in before.items():
            path = backup / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        try:
            for name, data in after.items():
                (root / name).write_bytes(data)
            state_path.write_text(json.dumps(summary, indent=2) + '\n')
        except Exception:
            for name, data in before.items():
                (root / name).write_bytes(data)
            raise
    print(json.dumps({'applied': args.apply, **summary}, indent=2))


if __name__ == '__main__':
    main()

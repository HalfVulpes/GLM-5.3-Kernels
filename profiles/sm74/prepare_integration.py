#!/usr/bin/env python3
"""Build an offline source overlay, never modify the supplied installation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from hooks import transform


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site-packages', type=Path, required=True)
    p.add_argument('--output', type=Path, help='new directory; omit for validation only')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'integration-manifest.json').read_text())
    source = json.loads((here / 'source-manifest.json').read_text())
    for name, digest in source['files'].items():
        if sha((here / name).read_bytes()) != digest:
            raise ValueError('Published source changed: ' + name)
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp) / 'overlay'
        stage.mkdir()
        for name, hashes in manifest.items():
            f = a.site_packages / name
            if f.is_symlink() or sha(f.read_bytes()) != hashes['input']:
                raise ValueError('Unqualified input source: ' + name)
            out = stage / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
        transform(stage)
        for name, hashes in manifest.items():
            if sha((stage / name).read_bytes()) != hashes['output']:
                raise ValueError('Generated source differs from tested runtime: ' + name)
        shutil.copy2(here / 'titan_recipe_thin.py', stage / 'vllm/titan_recipe_thin.py')
        shutil.copytree(here / 'titan_prefill', stage / 'titan_prefill',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if a.output:
            # copytree refuses an existing output, including a symlink.
            shutil.copytree(stage, a.output)
        print('Qualified source hashes match. Input installation was not modified.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Prepare the pinned upstream source for Titan group-32 prefill only."""
from pathlib import Path
import argparse,hashlib,re
EXPECTED = {'kernel_selector_tiles.h': 'f4a7fac6df8446b3ec4f228d9a7307b543617f7dd31470d97ef174c5cafd26ab', 'kernels_tiles_sm80.cu': '9f3627aaba203cb95a07e5a9e6577dff3ee5772e0fe428c34cf70f8a1aed12a0', 'module.cpp': '87da8276b682141a4b05ae0d03e22875fa01c56b1769f4b7c19834a2175c4753', 'build_standalone.py': '8fdd663f165099d88af7ff38d906289314d15886d85c8cd369107febe6c30d0a'}
p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args();root=a.source/'csrc/libtorch_stable/moe/ampere_marlin'
for name,digest in EXPECTED.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, 'Source pin mismatch: '+name
for name in ['kernels_tiles_sm80.cu','kernel_selector_tiles.h']:
    out=[];skip=False
    for line in (root/name).read_text().splitlines(keepends=True):
        if name.endswith('.h') and 'fast_dequant == true' in line:skip=True;continue
        if skip and 'kernel = Marlin<' in line:skip=False;continue
        if 'Marlin<' in line:
            m=re.search(r'Marlin<([^>]+)>',line);assert m
            args=[x.strip() for x in m.group(1).split(',')];assert len(args)==13 and args[10]=='8'
            if args[12]=='true':continue
            args[10]='2'
            line=line[:m.start(1)]+', '.join(args)+line[m.end(1):]
        out.append(line.replace('group_blocks == 8','group_blocks == 2'))
    (root/name).write_text(''.join(out))
p=root/'module.cpp';s=p.read_text().replace('"{s:i,s:s,s:i,s:i,s:i}"','"{s:i,s:s,s:i,s:i,s:i,s:i}"').replace('"prefill_schedule_version", 2);','"prefill_schedule_version", 2, "prefill_tile_group_size", 32);');p.write_text(s)
p=root/'build_standalone.py';s=p.read_text().replace('("decode.cu", "decode_orig.cu", "ops.cu", "kernels_sm80.cu",','("ops.cu", "kernels_sm80.cu",');p.write_text(s)
print('Prepared group-32 tiles; existing group-128 wide kernels unchanged; no decode extension sources.')

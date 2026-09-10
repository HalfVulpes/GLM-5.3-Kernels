# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for the GLM-5.3-Kernels SM80 integration; see docs/SOURCES.md.
"""Install the 528-byte MLA layout in both the layer and hybrid cache planner."""
import ast
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Pinned source mismatch: {old!r}')
    return source.replace(old, new, 1)


def patch_tree(root):
    root = Path(root)
    path = root / 'model_executor/layers/attention/mla_attention.py'
    source = path.read_text()
    source = replace_once(source,
        'state_content_bytes=656 if self.kv_cache_dtype == "fp8_ds_mla" else None,',
        'state_content_bytes=(528 if self.kv_cache_dtype == "int8_per_token_head" else 656 if self.kv_cache_dtype == "fp8_ds_mla" else None),')
    source = replace_once(source,
        'if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla":',
        'if fp8_attention and self.kv_cache_dtype not in ("fp8_ds_mla", "int8_per_token_head"):')
    ast.parse(source)
    path.write_text(source)
    path = root / 'v1/kv_cache_interface.py'
    source = path.read_text()
    start = source.index('class MLAAttentionSpec(')
    end = source.index('class HiddenStateCacheSpec(', start)
    section = replace_once(source[start:end],
        '        super().__post_init__()\n        _apply_alignment_padding(self)',
        '        super().__post_init__()\n'
        '        if self.cache_dtype_str == "int8_per_token_head":\n'
        '            assert self.head_size == 512, "INT8 MLA requires latent 512"\n'
        '            object.__setattr__(self, "state_content_bytes", 528)\n'
        '        _apply_alignment_padding(self)')
    source = source[:start] + section + source[end:]
    ast.parse(source)
    path.write_text(source)

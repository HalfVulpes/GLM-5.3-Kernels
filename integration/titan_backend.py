# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for the GLM-5.3-Kernels SM80 integration; see docs/SOURCES.md.
# Derived from vLLM contributors' Triton/XPU sparse MLA backends.
"""Titan-only 512-latent MLA, TP4, dynamic per-token INT8 cache."""
from typing import ClassVar
import torch
from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseBackend, XPUMLASparseImpl, XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view, triton_convert_req_index_to_global_index,
)
from titan_mla import triton_mla_sparse_attention
from titan_kv import store_int8

class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

class TritonMLASparseImpl(XPUMLASparseImpl):
    supports_quant_query_input = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.head_size == 512, 'Titan kernel requires GLM-5.3 NoPE MLA'
        assert self.num_heads == 16, 'Titan kernel requires TP4 (16 heads/rank)'
        self._sm_count = 70

    def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping,
                           kv_cache_dtype, k_scale):
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype == 'int8_per_token_head':
            assert k_pe.numel() == 0
            store_int8(kv_c_normed, kv_cache, slot_mapping)
        else:
            super().do_kv_cache_update(kv_c_normed, k_pe, kv_cache, slot_mapping,
                                      kv_cache_dtype, k_scale)

    def forward_mqa(self, q, kv_cache, attn_metadata, layer):
        if isinstance(q, tuple):
            q = q[0] if q[1].numel() == 0 else torch.cat(q, dim=-1)
        buf = self._indexer.topk_indices_buffer if self._indexer is not None else self.topk_indices_buffer
        indices = buf[:q.shape[0]]
        if self.kv_cache_dtype == 'int8_per_token_head':
            kv_cache = kv_cache.view(torch.int8)
        rows, stride = flat_kv_row_view(kv_cache, attn_metadata.block_size)
        indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token, attn_metadata.block_table, indices,
            BLOCK_SIZE=attn_metadata.block_size, BLOCK_STRIDE_ROWS=stride,
            NUM_TOPK_TOKENS=indices.shape[1])
        out = triton_mla_sparse_attention(q, rows.view(-1, 1, rows.shape[-1]),
            indices.view(q.shape[0], 1, -1), self.softmax_scale, sm_count=self._sm_count)
        return out, None

class TritonMLASparseBackend(XPUMLASparseBackend):
    supported_kv_cache_dtypes = ['auto', 'bfloat16', 'int8_per_token_head']
    @staticmethod
    def get_name(): return 'TRITON_MLA_SPARSE'
    @staticmethod
    def get_supported_kernel_block_sizes(): return [MultipleOf(64)]
    @classmethod
    def get_supported_head_sizes(cls): return [512]
    @staticmethod
    def get_builder_cls(): return TritonMLASparseMetadataBuilder
    @staticmethod
    def get_impl_cls(): return TritonMLASparseImpl

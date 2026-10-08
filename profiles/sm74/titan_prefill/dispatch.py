"""TP4-only prefill integration, preserving recurrent state and KV interfaces."""
import os
import torch
from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate as _original_kda
from . import kda,marlin
from .extension import prefill_op

_KDA=os.environ.get('TITAN_KDA_PREFILL','0')=='1'
_MARLIN=os.environ.get('TITAN_MARLIN_PREFILL','0')=='1'

def chunk_kda_with_fused_gate(*args,**kw):
 if not _KDA or args:return _original_kda(*args,**kw)
 q,k,v,g= (kw.get(n) for n in ['q','k','v','raw_g']);state=kw.get('initial_state');cu=kw.get('cu_seqlens');bias=kw.get('g_bias');a=kw.get('A_log');beta=kw.get('beta')
 valid=(all(isinstance(x,torch.Tensor) for x in [q,k,v,g,state,cu,bias,a,beta])
        and q.ndim==4 and q.shape[0]==1 and 1<=q.shape[1]<=2304 and tuple(q.shape[2:])==(16,128)
        and tuple(k.shape)==tuple(q.shape) and tuple(v.shape)==tuple(q.shape) and tuple(g.shape)==tuple(q.shape)
        and all(x.dtype==torch.bfloat16 for x in [q,k,v,g]) and all(x.dtype==torch.float32 for x in [state,bias,a,beta])
        and state.ndim==4 and tuple(state.shape[1:])==(16,128,128) and cu.ndim==1 and 2<=cu.numel()<=9
        and state.shape[0]>=cu.numel()-1 and kw.get('output_final_state') is True and kw.get('use_qk_l2norm_in_kernel') is True
        and kw.get('safe_gate') is True and float(kw.get('lower_bound',-5.))==-5.)
 return (kda.chunk_kda_with_fused_gate if valid else _original_kda)(**kw)

def maybe_marlin(layer,output,x,w1,w2,weights,ids,activation,global_experts,expert_map,router_on_input,workspace13,workspace2):
 if not _MARLIN or not 384<=x.shape[0]<=2304 or not layer.is_k_full:return False
 if any(getattr(layer,n,None) is not None and getattr(layer,n).numel() for n in ['w13_g_idx','w2_g_idx','w13_g_idx_sort_indices','w2_g_idx_sort_indices']):return False
 if marlin.gate_reason(layer,x,w1,w2,weights,ids,activation,global_experts,expert_map,router_on_input,allowed_n=(512,)) is not None:return False
 capturing=torch.cuda.is_current_stream_capturing();buf=marlin._buffers(x.device,288,x.shape[0]*8,create=not capturing)
 if buf is None:return False
 scratch=marlin._tile_scratch(x.device,create=not capturing)
 if scratch is None:return False
 marlin.run(layer,output,x,w1,w2,weights,ids,activation,workspace13,workspace2,buf,compiled=(prefill_op(),marlin.TILE_TABLES[512][1],scratch,None));return True

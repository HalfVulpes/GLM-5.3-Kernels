"""Load only the qualified group-32 prefill tile operator; never JIT in serving."""
import functools,importlib,importlib.util,os
import torch

@functools.cache
def prefill_op():
 path=os.environ.get('TITAN_PREFILL_EXTENSION_PATH')
 if path:
  spec=importlib.util.spec_from_file_location('vllm._ampere_marlin_C',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 else:module=importlib.import_module('vllm._ampere_marlin_C')
 info=module.build_info()
 expected={'abi_version':1,'torch_version':torch.__version__.split('+')[0],'cxx11_abi':int(torch._C._GLIBCXX_USE_CXX11_ABI),'prefill_schedule_version':2,'prefill_tile_group_size':32}
 if any(info.get(k)!=v for k,v in expected.items()) or info['cuda_version']//1000!=int(torch.version.cuda.split('.')[0]):raise RuntimeError('Prefill extension ABI or group-size mismatch')
 names='a c_or_none b_q_weight b_bias_or_none b_scales a_scales global_scale b_zeros_or_none workspace sorted_token_ids expert_ids num_tokens_past_padded topk_weights moe_block_size top_k mul_topk_weights b_type_id size_m size_n size_k use_atomic_add use_fp32_reduce is_zp_float thread_k thread_n blocks_per_sm c_tmp redo'
 schema=torch._C._dispatch_find_schema_or_throw('_ampere_marlin_C::prefill_tile_gemm','').schema()
 if ' '.join(a.name for a in schema.arguments)!=names:raise RuntimeError('Prefill operator schema mismatch')
 return torch.ops._ampere_marlin_C.prefill_tile_gemm

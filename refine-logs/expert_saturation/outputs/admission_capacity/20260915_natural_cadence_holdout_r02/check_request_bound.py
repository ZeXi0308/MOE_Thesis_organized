"""One CPU regression: execute the real qualifier against a bounded engine substitute."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as S
from unittest.mock import patch

root=Path(__file__).resolve().parent
old=root.parent/'20260915_natural_cadence_holdout_r01/pkg/safe_static.py'
source=root/'pkg/safe_static.py'
assert source.read_text()==old.read_text().replace("config['requests'] == 64","config['requests'] == 128")
spec=importlib.util.spec_from_file_location('repaired_safe_static',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class FullAttentionSpec:
    block_size=16
    sliding_window=None
    attention_chunk_size=None

fake=ModuleType('vllm.v1.kv_cache_interface');fake.FullAttentionSpec=FullAttentionSpec
pool=S(num_gpu_blocks=4097,get_num_free_blocks=lambda:4096,null_block=S(block_id=0,is_null=True))
coordinator=type('KVCacheCoordinatorNoPrefixCache',(),{})()
coordinator.single_type_managers=[S(block_pool=pool,block_size=16)]
coordinator.scheduler_block_size=16
layout=S(kv_cache_groups=[S(kv_cache_spec=FullAttentionSpec(),layer_names=['test.layer'],is_eagle_group=False)],num_blocks=4097)
manager=S(block_pool=pool,coordinator=coordinator,num_kv_cache_groups=1,watermark_blocks=0,
    enable_caching=False,use_eagle=False,kv_cache_config=layout)
scheduler=S(kv_cache_manager=manager,kv_cache_config=layout,num_lookahead_tokens=0,
    num_spec_tokens=0,dcp_world_size=1,pcp_world_size=1,requests={},get_request_counts=lambda:(0,0))
engine=S(engine_core=S(engine_core=S(scheduler=scheduler)),has_unfinished_requests=lambda:False,
    vllm_config=S(cache_config=S(enable_prefix_caching=False),speculative_config=None,
        scheduler_config=S(max_num_seqs=32),model_config=S(max_model_len=4096)))
config=json.loads((root/'pkg/inputs/config.json').read_text())
with patch.dict(sys.modules,{'vllm.v1.kv_cache_interface':fake}):
    result=module.qualify_safe_cap(engine,config)
    assert result['status']=='QUALIFIED',result
    assert result['usable_blocks']==4096 and result['maximum_request_tokens']==4088
    for n in (64,127,129):
        rejected=module.qualify_safe_cap(engine,dict(config,requests=n))
        assert rejected['status']=='QUALIFICATION_FAILED'
        assert rejected['error']=='ValueError: open episode upper bounds differ',rejected
print(json.dumps(dict(status='TARGETED_CPU_PASS_GPU_UNRUN',qualified_requests=128,
    rejected_requests=[64,127,129],runtime_diff='one request-count literal only',
    boundary='Actual qualifier, substituted engine/spec; native GPU behavior not rerun.')))

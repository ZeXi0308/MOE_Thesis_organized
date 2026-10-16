"""Small, native-only BF16/online-FP8 Triton MoE tuning grid for vLLM 0.26.

Run GPU work only through run_locked.py. --plan-only is CPU-only and checks
actual native route coverage without importing torch/vLLM. Real local OLMoE,
Qwen3-MoE or DeepSeek-V2-Lite routed-expert weights are assembled read-only.
The default is the first MoE layer (0 for OLMoE/Qwen3, 1 for DeepSeek-V2-Lite);
DeepSeek's dense and shared experts are excluded. Activations are synthetic
standard-normal BF16 and are explicitly not captured model activations.
--quantization fp8 invokes vLLM's actual online per-tensor weight conversion
on this BF16 layer, then its TRITON layout and quant-config builders. Dynamic
FP8 activation quantization is part of the measured expert block. Supply native
traces from the intended model/format: route shapes/provenance are checked, but
this script cannot infer the numerical format of every input trace artifact.

The objective is native top-k expert-block latency, not post-skip latency.
All comparison arms must use the SAME emitted VLLM_TUNED_CONFIG_FOLDER in a
fresh engine. CUDA graph timing includes alignment, both GEMMs, activation,
and reduction. It excludes router selection, CPU dispatch, and policy work.
Repeated cache conditioning is a declared approximation, not proof of cold
DRAM. A selected tile is a kernel tuning result, not a service speedup claim.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack,contextmanager
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def append_json(path,value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def mode_from_spec(spec):
    if spec is None:return 'native'
    if isinstance(spec,dict):return spec.get('mode','dense')
    if isinstance(spec,str):
        match=re.search(r"mode=['\"]([^'\"]+)",spec)
        return match.group(1) if match else None
    return None


def iter_traces(node,mode=None,location='$'):
    """Carry arm/spec provenance into nested worker trace lists."""
    if isinstance(node,list):
        for index,item in enumerate(node):
            yield from iter_traces(item,mode,f'{location}[{index}]')
    elif isinstance(node,dict):
        if 'arm' in node:
            mode=node['arm'] if node['arm'] in ('native','dense') else str(node['arm'])
        if 'spec' in node:
            mode=mode_from_spec(node['spec'])
        if 'execution_mode' in node:
            mode=node['execution_mode']
        if 'topk_ids' in node and 'topk_weights' in node:
            yield node,mode,location
            return
        for key,value in node.items():
            if isinstance(value,(dict,list)):
                yield from iter_traces(value,mode,f'{location}.{key}')


def load_native_routes(paths,layer,experts,top_k):
    selected=defaultdict(list)
    rejected=Counter()
    sources=[]
    layer_pattern=re.compile(rf'(^|\.)layers\.{layer}\.')
    for path in paths:
        path=Path(path)
        if path.suffix=='.jsonl':
            data=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            data=json.loads(path.read_text())
        sources.append({'path':str(path.resolve()),'sha256':sha256_file(path)})
        for row,mode,location in iter_traces(data):
            reason=None
            if mode not in ('native','dense'):reason='not_explicit_native_or_dense'
            elif row.get('phase')!='decode':reason='not_decode'
            elif not layer_pattern.search(row.get('layer','')):reason='other_layer'
            elif row.get('source')!='vllm_actual_select_experts':reason='not_actual_vllm_route'
            elif row.get('forward_completed') is not True:reason='incomplete_forward'
            if reason:
                rejected[reason]+=1
                continue
            ids,weights=row['topk_ids'],row['topk_weights']
            batch=len(ids)
            if (not batch or len(weights)!=batch or row.get('num_experts')!=experts
                    or any(len(a)!=top_k or len(b)!=top_k for a,b in zip(ids,weights))):
                raise ValueError(f'{path}:{location}: native routing shape does not match model')
            for token_ids,token_weights in zip(ids,weights):
                if (len(set(token_ids))!=top_k or any(not isinstance(e,int) or e<0 or e>=experts for e in token_ids)
                    or any(not math.isfinite(w) or w<0 for w in token_weights)):
                    raise ValueError(f'{path}:{location}: invalid native routes')
            if 'post_topk_ids' in row and row['post_topk_ids']!=ids:
                raise ValueError(f'{path}:{location}: reported native route has post-select drops')
            fingerprint=hashlib.sha256(json.dumps([ids,weights],separators=(',',':')).encode()).hexdigest()
            selected[batch].append({'layer':row['layer'],'step':row.get('step'),'batch':batch,
                'num_experts':experts,'top_k':top_k,'topk_ids':ids,'topk_weights':weights,
                'native_mode':mode,'source_file':str(path.resolve()),'source_json_path':location,
                'route_sha256':fingerprint,'distinct_experts':len({e for token in ids for e in token})})
    return dict(selected),{'sources':sources,'rejected':dict(rejected)}


def sample_routes(routes,batches,limit):
    result={}
    for batch in batches:
        rows=routes.get(batch,[])
        if not rows:
            raise ValueError(f'No complete native layer route for B={batch}; capture this batch, do not synthesize/concatenate routes')
        if len(rows)<=limit:
            result[batch]=rows
        else:
            indexes=([len(rows)//2] if limit==1 else
                     [round(i*(len(rows)-1)/(limit-1)) for i in range(limit)])
            result[batch]=[rows[i] for i in indexes]
    return result


def small_grid():
    return [dict(BLOCK_SIZE_M=m,BLOCK_SIZE_N=n,BLOCK_SIZE_K=k,GROUP_SIZE_M=1,
                 SPLIT_K=1,num_warps=4,num_stages=3)
            for m,n,k in itertools.product((16,32),(64,128),(64,128))]


def unique_configs(configs):
    result=[];seen=set()
    for cfg in configs:
        key=json.dumps(cfg,sort_keys=True)
        if key not in seen:
            result.append(dict(cfg));seen.add(key)
    return result


def read_model_config(model):
    model=Path(model).resolve()
    if not model.is_dir():raise ValueError('--model must be an existing local checkpoint directory')
    config=json.loads((model/'config.json').read_text())
    model_geometry(config)
    return model,config


def model_geometry(config):
    """Resolve only the three requested BF16 model families' routed experts.

    Qwen3 and DeepSeek have a separate dense intermediate_size; using it for
    expert tensors would tune a different kernel shape. Shared experts never
    enter the routed expert count, tensor layout, or objective.
    """
    kind=config.get('model_type')
    if kind not in ('olmoe','qwen3_moe','deepseek_v2') or config.get('quantization_config'):
        raise ValueError('Expected an unquantized OLMoE, Qwen3-MoE or DeepSeek-V2-Lite checkpoint')
    if config.get('hidden_act','silu')!='silu':
        raise ValueError('This native BF16 tuner requires gated SiLU experts')
    e_key='n_routed_experts' if kind=='deepseek_v2' else 'num_experts'
    i_key='intermediate_size' if kind=='olmoe' else 'moe_intermediate_size'
    keys={'experts':e_key,'top_k':'num_experts_per_tok','hidden':'hidden_size',
          'intermediate':i_key,'layers':'num_hidden_layers'}
    shape={}
    for name,key in keys.items():
        value=config.get(key)
        if type(value) is not int or value<=0:raise ValueError(f'{kind}: expected positive integer {key}')
        shape[name]=value
    if shape['top_k']>shape['experts']:raise ValueError('top-k exceeds routed expert count')
    if kind=='deepseek_v2':
        first=config.get('first_k_dense_replace',0)
        freq=config.get('moe_layer_freq',1)
        shared=config.get('n_shared_experts') or 0
        if config.get('routed_scaling_factor',1.0)!=1.0:
            raise ValueError('Only the requested DeepSeek-V2-Lite unit routed_scaling_factor is supported')
        if type(first) is not int or first<0 or type(freq) is not int or freq<1:
            raise ValueError('Invalid DeepSeek dense prefix or MoE layer frequency')
        if type(shared) is not int or shared<0:raise ValueError('Invalid shared expert count')
        layers=[layer for layer in range(first,shape['layers']) if layer%freq==0]
    elif kind=='qwen3_moe':
        freq=config.get('decoder_sparse_step',1)
        dense=config.get('mlp_only_layers',[])
        if type(freq) is not int or freq<1 or not isinstance(dense,list):
            raise ValueError('Invalid Qwen3 sparse layer schedule')
        if any(type(layer) is not int or not 0<=layer<shape['layers'] for layer in dense):
            raise ValueError('Invalid Qwen3 mlp_only_layers')
        layers=[layer for layer in range(shape['layers']) if layer not in dense and (layer+1)%freq==0]
        shared=0
    else:
        layers=list(range(shape['layers']));shared=0
    if not layers:raise ValueError('Checkpoint has no routed MoE layer')
    return {**shape,'model_type':kind,'first_moe_layer':layers[0],'moe_layers':layers,
            'shared_experts_excluded':shared,'intermediate_config_key':i_key}


def resolve_layer(config,layer=None):
    geometry=model_geometry(config)
    layer=geometry['first_moe_layer'] if layer is None else layer
    if type(layer) is not int or layer not in geometry['moe_layers']:
        raise ValueError(f'Layer {layer} is not a routed MoE layer; first is {geometry["first_moe_layer"]}')
    return layer


def load_layer_weights(model,config,layer):
    import torch
    from safetensors import safe_open
    model=Path(model)
    geometry=model_geometry(config)
    layer=resolve_layer(config,layer)
    E,D,I=(geometry[key] for key in ('experts','hidden','intermediate'))
    keys=[f'model.layers.{layer}.mlp.experts.{e}.{projection}.weight'
          for e in range(E) for projection in ('gate_proj','up_proj','down_proj')]
    index=model/'model.safetensors.index.json'
    if index.exists():
        mapping=json.loads(index.read_text())['weight_map']
    else:
        mapping={}
        for path in sorted(model.glob('*.safetensors')):
            with safe_open(str(path),framework='pt',device='cpu') as handle:
                for key in handle.keys():
                    if key in mapping:raise ValueError(f'Duplicate checkpoint tensor {key}')
                    mapping[key]=path.name
    missing=[key for key in keys if key not in mapping]
    if missing:raise ValueError(f'missing {geometry["model_type"]} routed-expert keys, first: {missing[:3]}')
    files=sorted({mapping[key] for key in keys})
    for file in files:
        if not isinstance(file,str) or Path(file).is_absolute() or '..' in Path(file).parts:
            raise ValueError(f'Invalid checkpoint shard path: {file}')
        if not (model/file).is_file():
            raise FileNotFoundError(f'Required routed-expert shard is not complete: {model/file}')
    w1=torch.empty((E,2*I,D),dtype=torch.bfloat16)
    w2=torch.empty((E,D,I),dtype=torch.bfloat16)
    dtypes=set()
    with ExitStack() as stack:
        handles={file:stack.enter_context(safe_open(str(model/file),framework='pt',device='cpu')) for file in files}
        for e in range(E):
            for projection,target,shape in [('gate_proj',w1[e,:I],(I,D)),
                                             ('up_proj',w1[e,I:],(I,D)),('down_proj',w2[e],(D,I))]:
                key=f'model.layers.{layer}.mlp.experts.{e}.{projection}.weight'
                source=handles[mapping[key]].get_tensor(key)
                if tuple(source.shape)!=shape:raise ValueError(f'{key}: {source.shape} != {shape}')
                if source.dtype not in (torch.bfloat16,torch.float16,torch.float32):
                    raise ValueError(f'{key}: unsupported checkpoint dtype {source.dtype}')
                dtypes.add(str(source.dtype));target.copy_(source)
    def tensor_hash(tensor):
        return hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()
    meta={'model_path':str(model),'layer':layer,'model_type':geometry['model_type'],
          'source_files':files,'source_dtypes':sorted(dtypes),'geometry':geometry,
          'execution_dtype':'torch.bfloat16','w1_shape':list(w1.shape),'w2_shape':list(w2.shape),
          'w1_sha256':tensor_hash(w1),'w2_sha256':tensor_hash(w2),
          'config_sha256':sha256_file(model/'config.json'),
          'index_sha256':sha256_file(index) if index.exists() else None,
          'checkpoint_key_template':f'model.layers.{layer}.mlp.experts.{{expert}}.{{gate_proj,up_proj,down_proj}}.weight',
          'weight_layout':'w1[e]=concat(gate_proj,up_proj) rows; w2[e]=down_proj, TP=1',
          'scope':'routed expert block only; excludes dense/shared experts and router selection',
          'expert_weight_bytes':(2*I*D+D*I)*2}
    return w1,w2,meta


def prepare_online_fp8(w1,w2):
    """Use installed vLLM 0.26 conversion, without constructing a full engine.

    The native method calls our _setup_kernel hook after quantization. This
    hook invokes the same official TRITON converter and quant-config builder
    used by _Fp8OnlineMoEBase; only full modular-kernel construction is omitted,
    because the tuner calls the native fused_experts entry point directly.
    """
    import inspect
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineMoEMethod
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,convert_to_fp8_moe_kernel_format,make_fp8_moe_quant_config)
    if importlib.metadata.version('vllm').split('+')[0]!='0.26.0':
        raise RuntimeError('Online FP8 conversion requires vLLM 0.26.0')
    if not w1.is_cuda or not w2.is_cuda or w1.dtype!=torch.bfloat16 or w2.dtype!=torch.bfloat16:
        raise ValueError('Online FP8 conversion expects CUDA BF16 routed-expert tensors')
    if w1.ndim!=3 or w2.ndim!=3 or w1.shape!=(w2.shape[0],2*w2.shape[2],w2.shape[1]):
        raise ValueError('Expected TP=1 gate/up/down weight layout')
    started=time.perf_counter()
    layer=torch.nn.Module()
    layer.w13_weight=torch.nn.Parameter(w1,requires_grad=False)
    layer.w2_weight=torch.nn.Parameter(w2,requires_grad=False)
    layer.num_experts=layer.local_num_experts=w1.shape[0]
    layer.intermediate_size_per_partition=w2.shape[2]
    layer.weight_block_size=None

    class CaptureTritonWeights:
        def _setup_kernel(self,layer,w13,w2,w13_scale,w2_scale,w13_input_scale,w2_input_scale):
            if w13_input_scale is not None or w2_input_scale is not None:
                raise ValueError('Expected dynamic per-tensor activation quantization')
            self.result=convert_to_fp8_moe_kernel_format(
                fp8_backend=Fp8MoeBackend.TRITON,layer=layer,w13=w13,w2=w2,
                w13_scale=w13_scale,w2_scale=w2_scale,
                w13_input_scale=w13_input_scale,w2_input_scale=w2_input_scale)
            self.quant_config=make_fp8_moe_quant_config(
                fp8_backend=Fp8MoeBackend.TRITON,w1_scale=self.result[2],w2_scale=self.result[3],
                a1_scale=w13_input_scale,a2_scale=w2_input_scale,block_shape=None,
                per_act_token_quant=False,per_out_ch_quant=False,layer=layer)

    capture=CaptureTritonWeights()
    native_method=Fp8PerTensorOnlineMoEMethod.process_weights_after_loading
    with torch.inference_mode():
        native_method(capture,layer)
    converted=dict(zip(('w1','w2','w1_scale','w2_scale'),capture.result))
    tensors={}
    for name,value in converted.items():
        is_weight=name in ('w1','w2')
        expected_shape=(w1.shape if name=='w1' else w2.shape) if is_weight else (w1.shape[0],)
        expected_dtype=torch.float8_e4m3fn if is_weight else torch.float32
        if value.shape!=expected_shape or value.dtype!=expected_dtype or not value.is_contiguous():
            raise ValueError(f'Unexpected native TRITON FP8 {name}: {value.shape}, {value.dtype}')
        # Per-expert checks/transfers keep temporary FP32 and host copies small.
        digest=hashlib.sha256()
        for part in value.unbind(0):
            if not bool(torch.isfinite(part.float()).all()):
                raise ValueError(f'Nonfinite online FP8 tensor: {name}')
            raw=part.reshape(-1).view(torch.uint8).detach().cpu().contiguous()
            digest.update(memoryview(raw.numpy()))
        if not is_weight and not bool((value>0).all()):
            raise ValueError(f'Nonpositive online FP8 scale: {name}')
        tensors[name]={'shape':list(value.shape),'dtype':str(value.dtype),
                       'stride':list(value.stride()),'sha256':digest.hexdigest(),
                       'bytes':value.numel()*value.element_size()}
        if not is_weight:tensors[name].update(min=float(value.min()),max=float(value.max()))
    quant=capture.quant_config
    if (not quant.use_fp8_w8a8 or quant.block_shape is not None or quant.per_act_token_quant
            or quant.a1_scale is not None or quant.a2_scale is not None):
        raise ValueError('Native builder did not produce dynamic per-tensor FP8 W8A8 config')
    sources=[]
    for function in (native_method,convert_to_fp8_moe_kernel_format,make_fp8_moe_quant_config,ops.scaled_fp8_quant):
        path=inspect.getsourcefile(function)
        sources.append({'function':function.__module__+'.'+function.__qualname__,
                        'path':path,'file_sha256':sha256_file(path),
                        'function_sha256':hashlib.sha256(inspect.getsource(function).encode()).hexdigest()})
    torch.cuda.synchronize()
    metadata={'quantization':'fp8','config_dtype':'fp8_w8a8','backend':'TRITON',
        'native_online_method_invoked':True,'full_engine_loaded':False,
        'gate_up_quantization':'concatenate BF16 gate/up first; native per-expert quantization of the merged w13 tensor',
        'activation_quantization':'dynamic per-tensor FP8 inside measured fused_experts; BF16 input/output',
        'block_shape':None,'per_act_token_quant':False,'per_out_ch_quant':False,
        'tensors':tensors,'sources':sources,
        'preparation_and_verification_seconds':time.perf_counter()-started,
        'preparation_in_timing_objective':False}
    return converted['w1'],converted['w2'],quant,metadata


@contextmanager
def safe_override(package,config):
    # v0.26 override_config lacks try/finally after yield. Preserve its previous
    # process-local value even if a candidate compilation/resource check fails.
    previous=package.get_config()
    try:
        with package.override_config(config):yield
    finally:
        package._config=previous


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--traces',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[1,8,32,64])
    p.add_argument('--layer',type=int,default=None,help='Default: first routed MoE layer from config (DeepSeek-V2-Lite: 1; OLMoE/Qwen3: 0)')
    p.add_argument('--quantization',choices=['bf16','fp8'],default='bf16',
                   help='BF16 by default; fp8 uses native vLLM online per-tensor W8A8 conversion of the same checkpoint')
    p.add_argument('--events-per-batch',type=int,default=3)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--replays-per-sample',type=int,default=10)
    p.add_argument('--evict-mib',type=int,default=256)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--max-relative-l2',type=float,default=.02)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if min(args.events_per_batch,args.rounds,args.replays_per_sample)<=0 or args.evict_mib<0:
        raise ValueError('positive event/round/replay counts and nonnegative eviction size required')
    if len(set(args.batches))!=len(args.batches) or any(b<1 or b>64 for b in args.batches):
        raise ValueError('--batches must be distinct integers in [1,64]')
    if not math.isfinite(args.max_relative_l2) or args.max_relative_l2<=0:
        raise ValueError('--max-relative-l2 must be finite and positive')
    model,config=read_model_config(args.model)
    geometry=model_geometry(config)
    args.layer=resolve_layer(config,args.layer)
    E,K,D,I=(geometry[key] for key in ('experts','top_k','hidden','intermediate'))
    config_dtype='fp8_w8a8' if args.quantization=='fp8' else None
    routes,provenance=load_native_routes(args.traces,args.layer,E,K)
    selected=sample_routes(routes,args.batches,args.events_per_batch)
    args.output_dir.mkdir(parents=True)
    plan={'status':'CPU_PLAN' if args.plan_only else 'PREPARING','model':str(model),
          'shape':{'experts':E,'top_k':K,'hidden':D,'intermediate':I},'layer':args.layer,
          'model_type':geometry['model_type'],'model_geometry':geometry,
          'quantization':args.quantization,'config_dtype':config_dtype,
          'weight_preparation':('vLLM 0.26 Fp8PerTensorOnlineMoEMethod.process_weights_after_loading + '
              'official TRITON converter and quant-config builder' if config_dtype else 'BF16 routed-expert assembly'),
          'config_sha256':sha256_file(model/'config.json'),
          'weight_scope':'routed experts only; dense/shared experts excluded; captured routing weights preserved',
          'grid':small_grid(),'include_current_native_config':True,'selected_routes':selected,
          'route_provenance':provenance,'rounds':args.rounds,'replays_per_sample':args.replays_per_sample,
          'trace_format_scope':'provided native routes; model/precision identity must match the intended run, not inferred from tensor shapes',
          'seed':args.seed,'cache_conditioning':{'buffer_mib':args.evict_mib,
               'method':'read-modify-write separate CUDA graph before each measured MoE replay' if args.evict_mib else 'none; repeated same-layer hot-cache replay',
               'eviction_time_in_objective':False,'guarantees_cold_DRAM':False},
          'activation_source':'synthetic standard-normal BF16; not captured model activations',
          'selection_target':'native top-k only; equal-trace mean of per-trace median round latencies',
          'scope':'kernel tuning only; all arms share output configs; service remeasurement required'}
    write_json(args.output_dir/'plan.json',plan)
    if args.plan_only:
        print('CPU_PLAN',str(args.output_dir/'plan.json'),flush=True);return
    if importlib.metadata.version('vllm').split('+')[0]!='0.26.0':
        raise RuntimeError('This tuner requires vLLM 0.26.0')
    import torch
    import triton
    import vllm.model_executor.layers.fused_moe as package
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts,get_config_file_name,get_moe_configs,try_get_optimal_moe_config)
    if getattr(package,'get_config')() is not None:
        raise RuntimeError('process already has a MoE override config')
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    raw_path=args.output_dir/'samples.jsonl'
    state={'status':'RUNNING','plan':'plan.json','results':{},'candidate_checks':[],
           'quantization':args.quantization,'config_dtype':config_dtype,
           'environment':{'torch':torch.__version__,'triton':triton.__version__,'vllm':importlib.metadata.version('vllm'),
                          'gpu':torch.cuda.get_device_name(),'existing_tuned_folder':os.environ.get('VLLM_TUNED_CONFIG_FOLDER')}}
    try:
        w1,w2,weight_meta=load_layer_weights(model,config,args.layer)
        write_json(args.output_dir/'weights.json',weight_meta)
        w1,w2=w1.cuda(),w2.cuda()
        quant_config=None
        if config_dtype:
            w1,w2,quant_config,conversion_meta=prepare_online_fp8(w1,w2)
            write_json(args.output_dir/'fp8_conversion.json',conversion_meta)
            state['fp8_conversion']='fp8_conversion.json'
            state['weight_metadata_scope']='weights.json identifies source BF16 tensors; fp8_conversion.json identifies executed FP8 tensors/scales'
        defaults={b:dict(try_get_optimal_moe_config(w1.shape,w2.shape,K,config_dtype,b)) for b in args.batches}
        original_map=get_moe_configs(E,I,config_dtype)
        emitted={int(b):dict(cfg) for b,cfg in (original_map or {}).items()}
        # Preserve stock matrix points and default transition boundaries;
        # these anchors are NOT tuned measurements. File lookup is nearest-B.
        anchors={1,2,4,8,16,32,64,65,96,97,128,129,512,513,1024,2048,4096,8192,E*129-1,E*129}
        for batch in sorted(anchors):
            if batch not in emitted:
                emitted[batch]=dict(try_get_optimal_moe_config(w1.shape,w2.shape,K,config_dtype,batch))
        state['untimed_preserved_anchor_batches']=sorted(anchors-set(args.batches))
        state['current_native_configs']=defaults
        evict_graph=None
        if args.evict_mib:
            scratch=torch.zeros(args.evict_mib*1024*1024//4,device='cuda',dtype=torch.float32)
            scratch.add_(1.)
            torch.cuda.synchronize()
            evict_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(evict_graph):scratch.add_(1.)
        batch_order=list(args.batches);rng.shuffle(batch_order)
        state['batch_order']=batch_order
        for batch in batch_order:
            cfgs=unique_configs([defaults[batch],*small_grid()])
            events=[]
            for route_index,row in enumerate(selected[batch]):
                generator=torch.Generator(device='cuda').manual_seed(args.seed+batch*1009+route_index)
                x=torch.randn(batch,D,generator=generator,device='cuda',dtype=torch.bfloat16)
                ids=torch.tensor(row['topk_ids'],device='cuda',dtype=torch.int32)
                weights=torch.tensor(row['topk_weights'],device='cuda',dtype=torch.float32)
                with safe_override(package,defaults[batch]):
                    reference=fused_experts(x,w1,w2,weights,ids,global_num_experts=E,quant_config=quant_config)
                events.append((x,weights,ids,reference))
            graphs={};valid=[]
            build_order=list(range(len(cfgs)));rng.shuffle(build_order)
            for candidate in build_order:
                ok=True
                temporary=[]
                for route_index,(x,weights,ids,reference) in enumerate(events):
                    check={'batch':batch,'candidate':candidate,'config':cfgs[candidate],'trace_index':route_index}
                    try:
                        with safe_override(package,cfgs[candidate]):
                            for _ in range(3):
                                output=fused_experts(x,w1,w2,weights,ids,global_num_experts=E,quant_config=quant_config)
                            torch.cuda.synchronize()
                            error=float((output.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12))
                            finite=bool(torch.isfinite(output).all())
                            if not finite or error>args.max_relative_l2:
                                check.update(status='NUMERICAL_MISMATCH',relative_l2=error if math.isfinite(error) else None,finite=finite)
                                ok=False
                            else:
                                graph=torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    graph_output=fused_experts(x,w1,w2,weights,ids,global_num_experts=E,quant_config=quant_config)
                                graph.replay()
                                graph_error=float((graph_output.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12))
                                graph_finite=bool(torch.isfinite(graph_output).all())
                                if not graph_finite or graph_error>args.max_relative_l2:
                                    check.update(status='GRAPH_NUMERICAL_MISMATCH',relative_l2=error,
                                        graph_relative_l2=graph_error if math.isfinite(graph_error) else None,
                                        graph_finite=graph_finite)
                                    ok=False
                                else:
                                    temporary.append((route_index,graph,graph_output))
                                    check.update(status='VALID',relative_l2=error,finite=True,
                                        graph_relative_l2=graph_error,graph_finite=True)
                    except Exception as exc:
                        check.update(status='COMPILE_OR_RESOURCE_ERROR',error=f'{type(exc).__name__}: {exc}')
                        ok=False
                        # CUDA execution faults may poison subsequent timings.
                        if isinstance(exc,RuntimeError) and any(s in str(exc).lower() for s in
                                ('illegal memory','device-side assert','unspecified launch failure')):
                            state['candidate_checks'].append(check)
                            write_json(args.output_dir/'summary.json',state)
                            raise
                    state['candidate_checks'].append(check)
                    write_json(args.output_dir/'summary.json',state)
                    if not ok:break
                if ok:
                    valid.append(candidate)
                    for route_index,graph,output in temporary:graphs[(candidate,route_index)]=(graph,output)
            if 0 not in valid:
                raise RuntimeError(f'current native config failed at B={batch}; cannot select against an invalid reference')
            timings=defaultdict(list)
            for repeat in range(args.rounds):
                order=list(itertools.product(valid,range(len(events))));rng.shuffle(order)
                for order_index,(candidate,route_index) in enumerate(order):
                    graph,_=graphs[(candidate,route_index)]
                    # One conditioning/warmup cycle, excluded from samples.
                    if evict_graph:evict_graph.replay()
                    graph.replay()
                    pairs=[]
                    for _ in range(args.replays_per_sample):
                        if evict_graph:evict_graph.replay()
                        begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                        begin.record();graph.replay();end.record();pairs.append((begin,end))
                    pairs[-1][1].synchronize()
                    values=[begin.elapsed_time(end)*1000 for begin,end in pairs]
                    sample={'batch':batch,'round':repeat,'order_index':order_index,'candidate':candidate,
                            'config':cfgs[candidate],'trace_index':route_index,
                            'route_sha256':selected[batch][route_index]['route_sha256'],
                            'replay_us':values,'median_replay_us':statistics.median(values)}
                    append_json(raw_path,sample)
                    timings[(candidate,route_index)].append(sample['median_replay_us'])
            scores={candidate:statistics.mean(statistics.median(timings[(candidate,trace)])
                       for trace in range(len(events))) for candidate in valid}
            winner=min(valid,key=lambda candidate:(scores[candidate],candidate))
            emitted[batch]=cfgs[winner]
            state['results'][batch]={'candidates':cfgs,'valid_candidates':valid,'score_us':scores,
                'selected_candidate':winner,'selected_config':cfgs[winner],
                'stock_candidate':0,'selected_vs_stock_ratio':scores[winner]/scores[0],
                'trace_count':len(events),'selection_uses_postskip_data':False}
            write_json(args.output_dir/'summary.json',state)
            del graphs,events
        folder=args.output_dir/'tuned_configs';folder.mkdir()
        filename=get_config_file_name(E,I,config_dtype)
        payload={'triton_version':triton.__version__,**{str(b):cfg for b,cfg in sorted(emitted.items())}}
        write_json(folder/filename,payload)
        state.update(status='PASS',completed_unix=time.time(),tuned_config_folder=str(folder.resolve()),
            config_filename=filename,scope='native kernel tuning; use same folder for every service arm and rerun service measurements',
            uncertainty='rounds/traces characterize this microbenchmark; token/graph samples are not independent service runs')
        write_json(args.output_dir/'summary.json',state)
        print('PASS',str(folder.resolve()),flush=True)
    except Exception as exc:
        state.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',failed_unix=time.time())
        write_json(args.output_dir/'summary.json',state)
        raise


if __name__=='__main__':main()

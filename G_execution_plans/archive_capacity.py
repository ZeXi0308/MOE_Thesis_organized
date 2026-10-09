#!/usr/bin/env python3
"""Recompute the archived G capacity ledger using local evidence only.

Standard library only. No GPU, network, lock acquisition or new workload.
Writes capacity_boundary.md and evidence/capacity_ledger.json inside G.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {}


def read(name):
    raw = (ROOT / name).read_bytes()
    SOURCES[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def main():
    c = read('evidence/review_20261008/compact.json')
    frozen = read('evidence/review_endpoint_config.json')
    history = read('evidence/historical_feasibility.json')
    model = read('native_sources/model_config.json')
    assert c['status'] == 'STARTUP_COMPLETE_NO_SERVICE_RUN'
    assert c['engine_args'] == frozen['engine_args']['compact']
    assert c['engine_args']['dtype'] == 'bfloat16' and c['engine_args']['tensor_parallel_size'] == 1
    domain = dict(num_lookahead_tokens=0, watermark_blocks=0,
                  enable_prefix_caching=False, async_scheduling=False,
                  has_kv_transfer_config=False, has_speculative_config=False,
                  num_kv_cache_groups=1, scheduler_block_size_tokens=16,
                  kv_cache_spec_types=['FullAttentionSpec'], model_type='olmoe',
                  scheduler_reserve_full_isl=True)
    assert c['observed_capacity_domain'] == domain
    # This verifies the preserved source snapshot, not an unobserved deployment.
    for name, expected in read('evidence/capacity_source_sha256.json').items():
        path = 'native_sources/' + name
        SOURCES[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert SOURCES[path] == expected
    mib, gib = 2**20, 2**30
    token_bytes = (model['num_hidden_layers'] * 2 * model['num_key_value_heads'] *
                   (model['hidden_size'] // model['num_attention_heads']) * 2)
    block_bytes = 16 * token_bytes
    assert token_bytes == 131072 and block_bytes == 2097152
    total, usable = c['scheduler_kv_blocks_including_reserved_block'], c['scheduler_free_kv_blocks']
    assert total - usable == 1  # Reserved null block.
    work = {}
    for label in ('low', 'high'):
        name = f'inputs/workload_{label}.json'
        requests = read(name)
        assert SOURCES[name] == history['source_sha256'][f'D_prefill_budget_20261004/workload_{label}.json']
        lengths = [len(r['prompt_token_ids']) + r['max_tokens'] for r in requests]
        assert max(lengths) <= c['engine_args']['max_model_len']
        bound = sum((n + 15) // 16 for n in lengths)
        work[label] = dict(requests=len(requests), max_request_tokens=max(lengths),
                           envelope_blocks=bound, envelope_bytes=bound * block_bytes,
                           margin_blocks=usable-bound, margin_bytes=(usable-bound)*block_bytes,
                           nonbinding_proven=usable >= bound)
    e = {x['event']: x for x in c['stage_events']}
    steady = e['startup_explicit_empty_cache_after']['memory']
    before = e['startup_explicit_empty_cache_before']['memory']
    kv_bytes = e['kv_allocate_after']['allocated_kv']['unique_storage_bytes']
    assert kv_bytes == total * block_bytes
    graph_groups = {g['mode']: len(g['token_buckets']) for g in e['live_capture_order']['groups']}
    nonkv = steady['total_bytes'] - steady['free_bytes'] - kv_bytes
    pools = steady['torch_allocator_pools']
    metrics = dict(kv_total_blocks=total, kv_usable_blocks=usable, kv_storage_bytes=kv_bytes,
                   kv_usable_bytes=usable*block_bytes, actual_graphs=sum(graph_groups.values()),
                   graph_groups=graph_groups,
                   graph_estimate_bytes=e['profile_cudagraph_memory_after']['native_return_bytes'],
                   native_capture_delta_bytes=e['capture_model_after']['native_return_bytes'],
                   shared_workspace_bytes=steady['shared_workspace']['unique_storage_bytes'],
                   torch_graph_pool_reserved_bytes=sum(v['total_bytes'] for k,v in pools.items()
                                                       if k not in ('[0, 0]', '"unknown"')),
                   torch_allocated_bytes=steady['allocated_bytes'],
                   torch_reserved_bytes=steady['reserved_bytes'],
                   whole_device_nonkv_bytes=nonkv,
                   cache_cleanup_released_bytes=steady['free_bytes']-before['free_bytes'],
                   kv_blocks_before_cleanup=e['startup_ready']['num_blocks'],
                   kv_blocks_after_cleanup=e['startup_explicit_empty_cache_after']['num_blocks'],
                   graph_estimation_s=e['profile_cudagraph_memory_after']['elapsed_s'],
                   capture_s=e['capture_model_after']['elapsed_s'],
                   engine_startup_s=c['engine_startup_wall_s'],
                   graph_exclusive_residency_bytes=None, whole_device_transient_peak_bytes=None)
    dense = []
    for path in sorted((ROOT/'evidence/review_20261008').glob('dense*.json')):
        record = json.loads(path.read_text())
        if record.get('plan') != 'dense':
            continue
        name = str(path.relative_to(ROOT))
        record = read(name)
        # Stop rather than silently carry forward an obsolete archival claim.
        assert record['status'] == 'LOCK_BUSY_NO_GPU_INITIALIZED'
        assert not record['gpu_experiment_launched']
        dense.append(dict(path=name, status=record['status'], gpu_initialized=False))
    assert dense
    lost = work['high']['margin_blocks'] + 1
    result = dict(status='CLOSED_CURRENT_CANDIDATE_BOUNDARY_ARCHIVE',
                  new_gpu_runs=0, measured_g_service_runs=0,
                  token_bytes=token_bytes, block_bytes=block_bytes,
                  compact=metrics, workloads=work, dense_attempts=dense,
                  minimum_block_loss_to_invalidate_high_certificate=lost,
                  block_loss_byte_equivalent=lost*block_bytes,
                  threshold_meaning='Loss of a sufficient certificate, not proof of pressure or a graph-residency threshold',
                  source_sha256=SOURCES)
    (ROOT/'evidence/capacity_ledger.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    rows = []
    def row(name, value, kind, meaning):
        rows.append(f'| {name} | {value} | {kind} | {meaning} |')
    row('KV token / block', f'{token_bytes//1024} KiB / {block_bytes//mib} MiB', '结构推导', 'BF16 TP1；16 tokens/block；按模型配置复算')
    row('compact 总 / 可用 KV', f'{total} / {usable} blocks', '启动实测', '总数扣除1个null block；可用容量不是服务实测占用')
    row('compact 实际 KV storage / 可用字节等价', f'{kv_bytes/gib:.6f} / {usable*block_bytes/gib:.6f} GiB', '实测 / 算术换算', 'live unique storage / 可用blocks×2 MiB')
    for label,v in work.items():
        row(f'{label}（{v["requests"]}请求）最大占用包络', f'{v["envelope_blocks"]} blocks / {v["envelope_bytes"]/gib:.6f} GiB', '条件上界，非实测峰值', 'Σ ceil((prompt tokens + max output tokens)/16)')
        row(f'compact 相对 {label} 包络余量', f'{v["margin_blocks"]} blocks / {v["margin_bytes"]/gib:.6f} GiB', '实测容量减上界', '已测配置在声明语义下容量不生效')
    row('使 high 充足证书失效的最小可用块损失', f'{lost} blocks / {lost*block_bytes/gib:.6f} GiB', '离散算术界', '降至34429才低于34430；不是graph驻留阈值或实测瓶颈')
    row('compact 实际图数', str(metrics['actual_graphs']), '启动实测', '10 PIECEWISE + 9 FULL；非请求桶数')
    for name,key,kind,meaning in [
        ('graph 原生启动估计','graph_estimate_bytes','运行时估计','用于KV分配前的预留；不是最终graph独立驻留'),
        ('正式 capture 原生差额','native_capture_delta_bytes','阶段实测','原生设备空闲内存差；含阶段影响，不能独立归因'),
        ('共享 workspace','shared_workspace_bytes','持有storage实测','共享池；不乘层数或图数，不保证可返还KV'),
        ('PyTorch graph 池 reserved','torch_graph_pool_reserved_bytes','allocator快照','非默认segment pool；不含全部driver图元数据'),
        ('显式清cache释放','cache_cleanup_released_bytes','阶段实测','释放后已分配KV不扩容'),
    ]:
        row(name, f'{metrics[key]/mib:.0f} MiB', kind, meaning)
    row('清cache前 / 后 KV blocks', f'{metrics["kv_blocks_before_cleanup"]} / {metrics["kv_blocks_after_cleanup"]}', '启动实测', '均含null block；不是在线KV扩容')
    row('整设备占用减KV storage', f'{nonkv/gib:.6f} GiB', '设备计数器与storage之差', '含模型、context、driver、allocator、workspace和graph，非graph独占')
    row('Torch allocated / reserved', f'{metrics["torch_allocated_bytes"]/gib:.6f} / {metrics["torch_reserved_bytes"]/gib:.6f} GiB', 'allocator快照', '与其他分项重叠，不可相加为集合成本')
    row('graph独立驻留 / 完整瞬时峰值', 'NA / NA', '未单独测得', 'capture差和reserved均不能替代精确归因或完整峰值')
    row('估计 / capture / 引擎启动时间', f'{metrics["graph_estimation_s"]:.3f} / {metrics["capture_s"]:.3f} / {metrics["engine_startup_s"]:.3f} s', '一次启动观测', '含观测开销、缓存冷热影响；非执行速度比较，无重复噪声估计')
    row('dense 容量 / graph成本 / 服务效果', 'NA / NA / NA', f'{len(dense)}次尝试均锁忙，未初始化CUDA', '不是负性能结果，不能继承compact证书')
    row('native集合 / 其他tile或backend', 'NA', '未测', '不假定桶集合包含关系意味着内存单调，不声称存在实测速度—容量取舍')
    table='\n'.join(rows)
    text=f'''# G 容量账表与适用边界（封存）

**决定：关闭当前候选投入，作为边界材料归档。** 本页由 `archive_capacity.py` 从G目录现有原始记录和冻结输入生成；不执行实验。dense保持未测，关闭投入不等于一般问题被否定。

| 账目 | 数值 | 证据类型 | 解释与边界 |
|---|---:|---|---|
{table}

单位均为二进制MiB/GiB；精确字节数和输入SHA见 [capacity_ledger.json](evidence/capacity_ledger.json)。compact原件为 [compact.json](evidence/review_20261008/compact.json)，其中阶段事件保留固定测量顺序。原生流程为估计→KV分配→正式capture→清cache；0.9是原生显存利用策略，不是精确90%整设备硬上限。以上估计、capture差、workspace、graph池和整设备非KV数值互有重叠，不相加。没有第二个已测G配置，无法估计计划之间的驻留/容量差或排序。

**为什么当前证据不足以检验graph/KV取舍。** 已测compact可用容量高于整个high有限批次的最大需求，甚至把所有请求同时推到声明输出上限仍有{work['high']['margin_bytes']/gib:.6f} GiB余量。执行速度改变批次或完成时间也不能越过这个包络；因此进一步释放少量graph内存无法扩大这批请求的可容纳集合。至少损失{lost}个可用块才会失去这个充分保证，但失去保证仍不代表实际达到瓶颈。这个“块损失”等价量不能换写为新增多少graph字节：估计预留、取整、共享池和实际分配顺序会影响K(P)。132 MiB估计或76 MiB capture差均不是“可优化的两计划差值”。dense未运行，所以整个冻结负载对所有计划是否容量充足仍未知；当前数据不能完成设想的跨计划取舍检验。G服务吞吐、尾延迟和正确性比较均未运行。

**可进入论文的适用边界说明。** 在单张RTX PRO 6000、BF16 OLMoE、TP1及既定vLLM配置下，compact计划一次启动提供{usable}个可用KV块，每块2 MiB。对于冻结的12/160请求有限负载，在单输出、full attention、无prefix/speculation/connector、同步调度、零lookahead/watermark且无其他请求共池的条件下，由逐请求最大长度得到的容量包络分别为{work['low']['envelope_blocks']}与{work['high']['envelope_blocks']}块。因此，该已测计划在这两批请求上不会因KV容量不足触发分配或准入约束；此时计划驻留的少量下降本身不能带来容量收益，执行速度与启动开销仍可能有价值。该结论是启动测量与分配语义共同支持的条件边界，不是完整服务收益、实测峰值或所有执行计划的结论。dense及其他配置未测，尚未建立graph内存差异改变服务能力的证据；也不能据此否定更接近容量边界的部署中的联合选择价值。

**什么部署变化可能重新使问题重要。** 重新立题需先声明有业务依据的改变，例如持续到达并实际保留更多并发上下文、更长的输入/输出或常驻会话契约，或实际迁到更小显存设备/具有明确资源配额的部署；本次不构造或执行这些负载。新设备（包括5090）必须重新核对模型、后端与启动峰值可行性，不能按总显存大小推断收益。决定性条件是合法计划的实测K(P)差异足以跨越真实服务决策所需容量：在匹配前态的单事件诊断中，可检验 K_small < R_required ≤ K_large（R含当前持块和此次准入/执行所需块）；完整策略则保持外部负载与预算可比，允许内部轨迹改变。若所有计划仍覆盖需求，或容量虽紧张但计划差异不足以改变准入、抢占或可承载上下文，联合KV项仍无增量价值。有限trace包络是离线诊断，不向在线方法提供未来信息；未经论证不移用于持续到达或不同KV语义。

历史D运行仅作背景：不同GPU身份、不同调度代码，既有high峰值32031/可用36764块，graph capture差约0.17 GiB为舍入日志值；不能与本次compact拼成受控的计划比较。完整历史口径与未运行服务表保留在 [service_comparison.md](service_comparison.md)。所有失败启动原件保持不变。

最小复现（从毕业设计目录运行，仅Python标准库，全部输入位于G目录）：

```bash
python3 -B G_execution_plans/archive_capacity.py
```

命令只重写本页与 `evidence/capacity_ledger.json`，核验冻结输入、源码快照和实际配置，复算账目并保留dense未知状态；不导入torch/vLLM，不连接服务器，不获取GPU锁。
'''
    (ROOT/'capacity_boundary.md').write_text(text)
    print(json.dumps(dict(status=result['status'], high_margin_blocks=work['high']['margin_blocks'],
                          minimum_loss_blocks=lost, dense='NOT_RUN', new_gpu_runs=0)))


if __name__ == '__main__':
    main()

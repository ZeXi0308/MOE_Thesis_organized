#!/usr/bin/env python3
"""Prepare four native APC off/on/on/off cells; never upload or launch GPU work."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import rotation_runtime_repeat as repeat

ID = '20260914_prefix_cache_baseline_r01'
PROTOCOL = '''# 原生前缀缓存强基线：CPU 准备，GPU UNRUN

唯一问题：同固定 KV 预算下，原生 APC 是否改变抢占恢复成本及完整请求结果？
这是后续强基线准备；当前唯一 GPU 实验仍为四项同路径时间重复。本包不上传、不占队。
复用已观察 cohort2；32 requests × 3072 prompt / 1024 output，50 ms steady。
OLMoE BF16、vLLM0.26、原编译模式、FCFS、full-history reservation、cap32、token budget1024。
固定 KV 16089350144 bytes；初始化核实 7671 可用块。四个独立引擎顺序 off/on/on/off。
两臂不安装 headroom/rotation 策略；native_capture、memory_telemetry、metrics 保持逐字不变。
三个公共 warmup 输入/顺序/输出长度不变，保留全部 warmup；APC 可能改变其实际缓存命中。
warmup 后完全排空，两臂均调用 LLMEngine.reset_prefix_cache(False, False)，保存返回值及块状态。
reset 必须返回 True；非 null 块 ref_cnt 必须为0，全部 hash 为空，空闲块仍7671，否则测量 UNRUN。
每格初始及测量前检查 GPU；占用或查询失败 ABORT，不终止其他进程。每格结束回传再继续。
按 block 比较 off→on；两次同角色重复仅描述。四格全部合格才做全组比较，不事后挑选或替换。
请求独立推进，保留全部 token、schedule、缓存跳变、重算、失败、长调用和环境信息。
主指标：完整请求吞吐、平均完成、max-ITL；同时保留 TTFT、抢占数、实际重算 token 和等待跨度。
wall=scheduler_inclusive+engine_non_schedule+outside_engine_calls；恢复跨度可含其它请求有用 decode。
APC computed_adjustment 是缓存复用跳变，不是执行 token；只累计成功 engine.step 内的正跳变。
首次准入与抢占后的正跳变分开记录；不累计 waiting 失败重试的 cache lookup 命中。
原 capture 的 high-water 取成功调用后的 computed_after，包含继承的 APC 前缀；其 recompute_tokens
表示重建此前已建立的前缀，不能一概解释为该请求此前亲自执行过的 GPU token 再执行。
严格 request-specific 重复执行量应另用成功执行区间并集分析；本包不改采集公式。
cohort2 的32个首16-token块均不同；在原 block_size=16 且测量前冷缓存下，应无首次跨请求命中。
memory_trace free 包含可驱逐缓存块，used 是非空闲物理块，不是“带有效 KV 内容的块”。
APC 共享可使各请求块表长度之和大于物理占用；禁止借此推导错误守恒或仍用独占块的旧分析器。
所有策略保持独立 token/KV/完成状态，不从旧 trace 构造反事实；输出一致性不等于任务质量等价。
收益/退化/无变化均报告。若 APC 吸收原生恢复成本，先更新强基线；否则定位缓存存活与排队约束。
不据本四格声称 method GO、业务 SLO、跨模型显著性或 APC-aware 轮转成立。
停止：四格结束或任一资源/身份/会计条件失败；失败原样保留。此实验上限 NATIVE_SERVING/MEASUREMENT_ONLY。

源码 API 核验（官方 v0.26.0）：
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/engine/llm_engine.py : reset_prefix_cache 返回 bool，默认不重置运行请求/connector。
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/block_pool.py : reset 在非 null 块全 free 后清除 hash。
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_manager.py : APC 恢复查 request.num_tokens−1 内完整块。
'''
HELPERS = '''
def cache_pool_state(engine):
    scheduler = engine.engine_core.engine_core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    blocks = [b for b in pool.blocks if not b.is_null]
    return dict(free_blocks=pool.get_num_free_blocks(), usable_blocks=len(blocks),
        nonzero_refcount_blocks=sum(b.ref_cnt != 0 for b in blocks),
        negative_refcount_blocks=sum(b.ref_cnt < 0 for b in blocks),
        hashed_blocks=sum(b.block_hash is not None for b in blocks),
        pending_requests=len(scheduler.requests), counts=list(scheduler.get_request_counts()))


def reset_measured_cache(engine, out):
    receipt = dict(api='LLMEngine.reset_prefix_cache(False, False)', before=cache_pool_state(engine))
    dump(out/'prefix-cache-reset.json', receipt)
    if engine.has_unfinished_requests() or receipt['before']['pending_requests'] or receipt['before']['counts'] != [0, 0]:
        raise RuntimeError('prefix reset requires drained engine')
    receipt['returned'] = engine.reset_prefix_cache(reset_running_requests=False, reset_connector=False)
    receipt['after'] = cache_pool_state(engine)
    dump(out/'prefix-cache-reset.json', receipt)
    s = receipt['after']
    if receipt['returned'] is not True or s['free_blocks'] != 7671 or s['usable_blocks'] != 7671 or s['nonzero_refcount_blocks'] or s['hashed_blocks']:
        raise RuntimeError('prefix reset/cache refcounts did not qualify')


def cache_adjustments(raw):
    successful = {i for c in raw['engine_steps'] if c['completed']
        for i in range(c['scheduler_step_start'], c['scheduler_step_end'])}
    rows = []
    for i in sorted(successful):
        before = raw['memory_trace'][i]['before']['requests']
        for r in raw['scheduler_steps'][i]['scheduled']:
            if r['computed_adjustment'] > 0:
                rows.append(dict(step=i, request_id=r['request_id'], tokens=r['computed_adjustment'],
                    role='post_preemption' if before[r['internal_request_id']]['num_preemptions'] else 'first_admission'))
    return dict(successful_positive_adjustments=rows,
        first_admission_tokens=sum(r['tokens'] for r in rows if r['role'] == 'first_admission'),
        post_preemption_tokens=sum(r['tokens'] for r in rows if r['role'] == 'post_preemption'),
        scope='Observed successful computed jumps, not failed lookups or counterfactual savings.')


'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source anchor differs: '+old[:80])
    return text.replace(old, new, 1)


def accounting_check(path):
    text = path.read_text()
    fragment = text[text.index('                actual_start = after - amount'):text.index('                scheduled.append(')]
    fragment = '\n'.join(line[16:] for line in fragment.splitlines())
    cases = [(0, 1024, 1024, 0, (1024, 0, 0)), (0, 3073, 1, 0, (0, 1, 0)),
        (0, 3329, 257, 3584, (0, 0, 257)), (0, 3585, 1, 3584, (0, 1, 0))]
    for prior, after, amount, high, expected in cases:
        state = dict(after=after, amount=amount, prior=prior, prompt_length=3072,
            rid='request', executed_high_water={'request': high})
        exec(compile(fragment, str(path)+':APC_boundary_check', 'exec'), state)
        assert (state['prefill'], state['decode'], state['recompute']) == expected
        assert sum(expected) == amount
    return dict(status='PASS_CPU_ACCOUNTING_BOUNDARIES', cases=len(cases), gpu_validation='UNRUN')


def prepare(parent, out):
    if out.exists():
        raise FileExistsError(out)
    repeat.validate(parent)
    receipt = json.loads((parent.parent/'status.json').read_text())
    assert all(digest(parent/p) == sha for p, sha in receipt['unchanged_parent_files'].items())
    parent_manifest = json.loads((parent/'campaign.json').read_text())
    out.mkdir(parents=True)
    source = out/'source'; source.mkdir()
    names = ('run_probe.py', 'run_campaign.py', 'run_one_cell.py', 'safe_static.py', 'native_capture.py', 'memory_telemetry.py', 'metrics.py')
    for name in names:
        shutil.copyfile(parent/name, source/name)
    for name in ('inputs_preparation', 'cohorts'):
        shutil.copytree(parent/name, source/name)
    cells = [dict(label=f'cohort2-block{b}-apc_{apc}', cohort_id='cohort2', block=b,
        role='apc_'+apc, completion_policy='native', cap=32, victim_order='least_progress', prefix_caching=apc)
        for b, apc in ((0, 'off'), (0, 'on'), (1, 'on'), (1, 'off'))]
    write(source/'campaign.json', dict(schema_version=1, experiment_id=ID, cohorts=parent_manifest['cohorts'],
        cells=cells, design='native-only APC off/on/on/off; same observed cohort; fresh engines', parent_campaign_sha256=digest(parent/'campaign.json')))
    probe = (source/'run_probe.py').read_text()
    for line in ('from completion_headroom import install\n', 'from absence_rotation import RotationConfig\n', 'from rotation_native import install as install_rotation\n'):
        probe = once(probe, line, '')
    probe = once(probe, 'def main():', HELPERS+'def main():')
    probe = once(probe, '    args = parser.parse_args()', "    parser.add_argument('--prefix-caching', choices=['off', 'on'], required=True)\n    args = parser.parse_args()\n    if args.completion_policy != 'native':\n        raise ValueError('only the unmodified native scheduler is permitted')")
    probe = once(probe, "headroom_observer='fast'", "headroom_observer='not_installed', enable_prefix_caching=args.prefix_caching == 'on'")
    probe = once(probe, "preemption_mode='native_recompute', rotation_config=vars(RotationConfig()),", "preemption_mode='native_recompute', rotation_config=None,")
    probe = once(probe, ", 'completion_headroom.py', 'absence_rotation.py', 'rotation_native.py'", '')
    probe = once(probe, 'enable_prefix_caching=False,', "enable_prefix_caching=args.prefix_caching == 'on',")
    probe = once(probe, "dict(status='INITIALIZING')", "dict(status='INITIALIZING', measurement_executed=False)")
    start = probe.index('        scheduler = engine.engine_core.engine_core.scheduler\n        if args.completion_policy')
    end = probe.index("        raw['gpu_before'] = before", start)
    probe = probe[:start]+"        dump(out/'status.json', dict(status='MEASURING', measurement_executed=True))\n        raw = capture_with_memory(engine, capture_episode, workload, config, allow_preemption=True,\n            regime='steady', arrival_scale=1.0, run_id='measured', max_seconds=120)\n"+probe[end:]
    probe = once(probe, '        set_empty_admission_cap(engine, args.cap)\n', '        set_empty_admission_cap(engine, args.cap)\n        reset_measured_cache(engine, out)\n')
    probe = once(probe, "        dump(out/'raw.json', raw)", "        dump(out/'raw.json', raw)\n        dump(out/'cache-accounting.json', cache_adjustments(raw))")
    probe = once(probe, "dict(status=status, requests_completed=", "dict(status=status, measurement_executed=True, requests_completed=")
    probe = once(probe, "dict(previous, status='INCOMPLETE', error=", "dict(previous, status='INCOMPLETE' if previous.get('measurement_executed') else 'UNRUN', error=")
    (source/'run_probe.py').write_text(probe)
    qualifier = (source/'safe_static.py').read_text()
    qualifier = once(qualifier, "require(type(coordinator).__name__ == 'KVCacheCoordinatorNoPrefixCache', 'unverified KV coordinator')", "require(type(coordinator).__name__ == ('UnitaryKVCacheCoordinator' if config['enable_prefix_caching'] else 'KVCacheCoordinatorNoPrefixCache'), 'unverified KV coordinator')")
    qualifier = once(qualifier, "require(not manager.enable_caching and not engine.vllm_config.cache_config.enable_prefix_caching,\n                'prefix sharing unsupported')", "require(manager.enable_caching == engine.vllm_config.cache_config.enable_prefix_caching == config['enable_prefix_caching'], 'APC configuration mismatch')")
    qualifier = once(qualifier, '        usable = total - 1', "        require(all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null), 'nonempty reference counts')\n        require(sum(not b.is_null for b in pool.blocks) == total - 1, 'null block count mismatch')\n        usable = total - 1")
    (source/'safe_static.py').write_text(qualifier)
    launcher = (source/'run_campaign.py').read_text()
    launcher = once(launcher, "'--victim-order', cell['victim_order'],", "'--prefix-caching', cell['prefix_caching'], '--victim-order', cell['victim_order'],")
    (source/'run_campaign.py').write_text(launcher.replace('frozen eight-cell order', 'frozen four-cell APC order'))
    unchanged = [p for p in source.rglob('*') if p.is_file() and p.name not in ('run_probe.py', 'run_campaign.py', 'safe_static.py', 'campaign.json')]
    assert all(digest(p) == digest(parent/p.relative_to(source)) for p in unchanged)
    for p in source.glob('*.py'):
        compile(p.read_text(), str(p), 'exec')
    assert not any(n in probe for n in ('install_rotation(', 'decisions, uninstall =', 'from completion_headroom'))
    labels = json.loads(subprocess.check_output([sys.executable, '-B', str(source/'run_campaign.py'), '--list'], text=True))
    assert labels == [c['label'] for c in cells] and len(set(labels)) == 4
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('apc_prepared_probe', source/'run_probe.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for root, domain in ((source, 'short'), (source, 'long'), (source/'cohorts/cohort2', 'long')):
        _, checked_workload = module.load_inputs(root, domain)
    first_blocks = len({tuple(t[:16]) for t in checked_workload['actual_prompt_token_ids']})
    assert first_blocks == 32
    sample = dict(engine_steps=[dict(completed=i != 1, scheduler_step_start=i, scheduler_step_end=i+1) for i in range(3)],
        memory_trace=[dict(before=dict(requests={'r': dict(num_preemptions=int(i > 0))})) for i in range(3)],
        scheduler_steps=[dict(scheduled=[dict(request_id='r', internal_request_id='r', computed_adjustment=n)]) for n in (16, 1000, 32)])
    observed = module.cache_adjustments(sample)
    assert (observed['first_admission_tokens'], observed['post_preemption_tokens']) == (16, 32)
    tree = ast.parse(probe)
    def engine_keywords(text):
        node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'kwargs' for t in n.targets))
        return {kw.arg:ast.dump(kw.value) for kw in node.value.keywords if kw.arg != 'enable_prefix_caching'}
    assert engine_keywords(probe) == engine_keywords((parent/'run_probe.py').read_text())
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'reset_prefix_cache' for n in ast.walk(tree))
    checks = dict(syntax='PASS', parent_receipt='PASS', three_input_sets='PASS', order=labels,
        engine_arguments_except_apc='UNCHANGED', successful_cache_adjustments='PASS_16_INITIAL_32_RECOVERY_FAILED_CALL_EXCLUDED',
        cohort2_unique_first_16_token_blocks=first_blocks,
        capture_accounting=accounting_check(source/'native_capture.py'), native_policy_only='PASS',
        unchanged_parent_files={str(p.relative_to(source)):digest(p) for p in unchanged})
    write(out/'cpu-checks.json', checks)
    (source/'DECISIONS.md').write_text(PROTOCOL)
    (out.parent/'REPORT.md').write_text(PROTOCOL)
    (out/'commands.txt').write_text('python -B run_campaign.py --list\n'+''.join(f'python -B run_one_cell.py {label}\n' for label in labels)+'# Run in source; one cell, read back, then next. GPU execution is UNRUN.\n')
    archive = out/'execution.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for p in sorted(source.iterdir()):
            bundle.add(p, arcname=p.name)
    write(out/'status.json', dict(status='PREPARED_UNRUN', uploaded=False, gpu_executions=0,
        archive_sha256=digest(archive), source_parent=str(parent.resolve()), labels=labels,
        preparation_script_sha256=digest(Path(__file__)), source_files_sha256={p.name:digest(p) for p in source.glob('*.py')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent_source, args.output_dir)

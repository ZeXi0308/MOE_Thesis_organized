#!/usr/bin/env python3
"""Prepare APC native/most/most/native from completed APC evidence; no GPU launch."""
import argparse
import ast
import importlib.util
import itertools
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace as NS

from prepare_prefix_cache_baseline import accounting_check, digest, once, write

ID = '20260914_apc_rotation_r01'
PROTOCOL = '''# APC 下固定轮转的最小验证

PREPARED_UNRUN；本包仅CPU准备，不上传、不执行GPU。整组排在A d6八格、B jit两格之后。
根会话在bundle外用共同flock /root/autodl-tmp/moe-research-gpu.lock保护整组，串行调用四个run_one_cell入口。
每格原始结果保留，整组终态后统一tar/hash回读；运行中不释放整组物理锁。
主问题仍为固定KV压力下的长恢复等待。已完成APC原生对照减小重算，但长恢复等待仍存在。
唯一问题：默认APC开启后，原most_output轮转能否缩短长暂停，并保持完整请求成本可接受？
四格独立引擎：native_apc / most_apc / most_apc / native_apc；同已观察cohort2，不声称新holdout。
保持OLMoE/BF16/vLLM0.26、32×3072/1024、50ms steady、KV16089350144bytes/7671usable blocks。
cap32、max_num_batched_tokens1024、full-history FCFS、原编译模式及三个公共warmup不变。
APC全部开启；每格warmup后原样排空/reset并保留receipt；捕获与metrics文件逐字复用APC包。
most只用原RotationConfig、most_output，不扫阈值；native不安装策略。保留全部失败/长调用/逐token输出。
动作仍调用原生_preempt_request和worker resumed块表通道；不手动touch、清缓存、截断KV或改变need。
need=ceil(当前完整history/B)-owned_len，含ref0缓存被touch时的free消耗，是保守完整历史预算。
强制受害者释放量为唯一非null/ref_cnt==1物理块数；原生preempt前后实际free增量必须相等。
仅支持单组精确FullAttentionSpec、Unitary、B=hash_block_size=scheduler_block_size=16；禁partial/CoW。
保留同步、无spec/connector/deferred free等旧防护；安装与两臂初始化绑定APC四格共同7项vLLM源码hash。
cohort首16-token块全unique、测量前冷缓存；激活与交换前后检查live块无共享、引用/对象/ID正确。
这限定当前无跨请求共享域，不声称适用于共享system-prompt负载。未知状态直接失败，不能解除防护硬跑。
held请求不得丢KV/推进，protected必须每步得到调度，free仍覆盖remaining need，禁止自然二次抢占。
4格全部合格再按block native→most比较；APC基线旧格只作背景，不替换新native。全部重复保留。
主指标为完整吞吐、平均完成、max-ITL及逐请求收益/受害者损失；报告抢占、hold、重算与调度开销。
wall=scheduler_inclusive+engine_non_schedule+outside_engine_calls，策略检查在scheduler内，不能重复扣除。
capture high-water含APC继承区间；保留原分类，严格重复执行量另由成功执行区间并集分析。
输出沿各自策略独立演化；token一致性不是任务质量等价。参考SLO不证明业务长暂停SLO成立。
无动作记INVALID_NO_ACTION；安全/资源/身份失败保留原始过程并停止。四格结束后只报告MEASUREMENT_ONLY。
若有暂停收益但完整成本退化，报告权衡；若无收益，定位动作/预算/成本，不自动改阈值或更换问题。
Oracle、跨模型/跨负载、质量、最近邻系统、方法GO均未验证。本轮不修改旧raw、共用文件或台账。
'''
HELPERS = '''
def qualify_source_files(root):
    observed = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in VLLM_SOURCE_SHA256}
    if observed != VLLM_SOURCE_SHA256:
        raise ValueError('vLLM sources differ from the completed APC baseline')
    return observed


def releasable_blocks(pool, blocks):
    ids = set()
    for b in blocks:
        if b.is_null or b.block_id in ids or b.block_id < 0 or b.block_id >= len(pool.blocks) or pool.blocks[b.block_id] is not b or b.ref_cnt < 1:
            raise RuntimeError('invalid victim block ownership')
        ids.add(b.block_id)
    return sum(pool.blocks[i].ref_cnt == 1 for i in ids)


def qualify_unshared(pool, owned, single):
    if getattr(single, '_partial_hit_reqs', {}) or getattr(single, '_pending_cow_copies', ()):
        raise RuntimeError('partial hit or pending CoW is unsupported')
    seen = set()
    for blocks in owned.values():
        releasable_blocks(pool, blocks)
        for b in blocks:
            if b.ref_cnt != 1 or b.block_id in seen:
                raise RuntimeError('shared live blocks outside qualified cohort domain')
            seen.add(b.block_id)
    active = {b.block_id for b in pool.blocks if not b.is_null and b.ref_cnt > 0}
    if seen != active or any(b.ref_cnt < 0 for b in pool.blocks):
        raise RuntimeError('physical references differ from request ownership')
    return dict(live_owned_blocks=len(seen), live_block_tables=len(owned))


'''


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cpu_state_checks(adapter, scheduler_source):
    compile(adapter.patched_schedule_tree(scheduler_source), 'pinned-native-schedule', 'exec')
    pool = NS(blocks=[NS(block_id=i, is_null=i == 0, ref_cnt=int(i != 0)) for i in range(3)])
    owned = {'victim': [pool.blocks[1]], 'survivor': [pool.blocks[2]]}; single = NS()
    assert adapter.qualify_unshared(pool, owned, single)['live_owned_blocks'] == 2
    for invalid in ([pool.blocks[0]], [pool.blocks[1], pool.blocks[1]], [NS(block_id=1, is_null=False, ref_cnt=1)]):
        try:
            adapter.releasable_blocks(pool, invalid)
            raise AssertionError('invalid ownership was not rejected')
        except RuntimeError:
            pass
    pool.blocks[2].ref_cnt = 2
    assert adapter.releasable_blocks(pool, pool.blocks[1:]) == 1
    try:
        adapter.qualify_unshared(pool, owned, single)
        raise AssertionError('shared live block was not rejected')
    except RuntimeError:
        pass
    pool.blocks[2].ref_cnt = 1
    try:
        adapter.qualify_unshared(pool, owned, NS(_partial_hit_reqs={'r': (0, pool.blocks[1])}))
        raise AssertionError('partial cache state was not rejected')
    except RuntimeError:
        pass
    tree = ast.parse(scheduler_source); cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_preempt_request', '_make_cached_request_data')]
    states = NS(RUNNING=NS(name='RUNNING'), PREEMPTED=NS(name='PREEMPTED'))
    namespace = dict(RequestStatus=states, CachedRequestData=NS, itertools=itertools)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+methods, type_ignores=[])), 'native-transitions', 'exec'), namespace)
    victim = NS(request_id='victim', status=states.RUNNING, num_computed_tokens=16, spec_token_ids=[],
        num_preemptions=0, output_token_ids=[7, 8], num_output_tokens=2, num_output_placeholders=0, all_token_ids=list(range(18)))
    waiting = []
    def free(request):
        for b in owned.pop(request.request_id):
            b.ref_cnt -= 1
    sched = NS(_free_request_blocks=free, encoder_cache_manager=NS(free=lambda r: None),
        _inflight_prefills=NS(discard=lambda r: None), log_stats=False,
        waiting=NS(prepend_request=lambda r: waiting.insert(0, r)), reset_preempted_req_ids=set(),
        use_pp=False, use_v2_model_runner=False, prev_step_scheduled_req_ids=set())
    before = sum(b.ref_cnt == 0 for b in pool.blocks[1:]); expected = adapter.releasable_blocks(pool, owned['victim'])
    namespace['_preempt_request'](sched, victim, 0.0)
    assert sum(b.ref_cnt == 0 for b in pool.blocks[1:])-before == expected == 1
    assert victim.num_computed_tokens == 0 and victim.output_token_ids == [7, 8] and waiting == [victim]
    adapter.qualify_unshared(pool, owned, single)
    victim.num_computed_tokens = 16
    blocks = NS(get_block_ids=lambda allow_none: ([1, 2],))
    cached = namespace['_make_cached_request_data'](sched, [], [victim], {'victim': 2}, {}, {'victim': blocks})
    assert cached.resumed_req_ids == {'victim'} and cached.num_computed_tokens == [16] and cached.new_block_ids == [([1, 2],)]
    return dict(status='PASS_CPU_ONLY', release_and_shared_rejection=True, native_preempt_and_resumed_payload=True, worker_execution='UNRUN')


def prepare(apc_run, rotation_source, scheduler_source, out):
    if out.exists():
        raise FileExistsError(out)
    parent = apc_run/'frozen'; read = lambda p: json.loads(p.read_text())
    manifest, executed = read(parent/'campaign.json'), read(apc_run/'execution.json')
    assert manifest['experiment_id'] == '20260914_prefix_cache_baseline_r01' and executed['status'] == 'COMPLETE' and len(executed['cells']) == 4
    prep = apc_run.parent/'preparation'
    assert all(digest(parent/n) == h for n,h in read(prep/'status.json')['source_files_sha256'].items())
    assert all(digest(parent/n) == h for n,h in read(prep/'cpu-checks.json')['unchanged_parent_files'].items())
    assert digest(parent/'campaign.json') == digest(prep/'source/campaign.json')
    environments = [read(apc_run/'gpu_results'/c['label']/'environment.json') for c in manifest['cells']]
    pins = environments[0]['vllm_source_sha256']
    assert len(pins) == 7 and all(e['vllm_source_sha256'] == pins for e in environments)
    assert all(read(apc_run/'gpu_results'/c['label']/'status.json')['status'] == 'COMPLETE' for c in manifest['cells'])
    assert digest(scheduler_source) == pins['v1/core/sched/scheduler.py']
    out.mkdir(parents=True); source = out/'source'
    shutil.copytree(parent, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'results'))
    for name in ('absence_rotation.py', 'rotation_native.py'):
        shutil.copyfile(rotation_source/name, source/name)
    rotation_receipt = read(rotation_source.parent.parent/'preparation/status.json')
    assert all(digest(rotation_source/n) == rotation_receipt['unchanged_parent_files'][n] for n in ('absence_rotation.py', 'rotation_native.py'))
    cells = [dict(label=f'cohort2-block{b}-{role}', cohort_id='cohort2', block=b, role=role,
        completion_policy='rotate' if role == 'most_apc' else 'native', cap=32,
        victim_order='most_output' if role == 'most_apc' else 'least_progress', prefix_caching='on')
        for b,role in ((0,'native_apc'), (0,'most_apc'), (1,'most_apc'), (1,'native_apc'))]
    write(source/'campaign.json', dict(schema_version=1, experiment_id=ID, cohorts=manifest['cohorts'], cells=cells,
        design='APC on; native/most/most/native; same observed cohort; independent engines', parent_campaign_sha256=digest(parent/'campaign.json')))
    adapter = (source/'rotation_native.py').read_text()
    adapter = once(adapter, 'import inspect\n', 'import inspect\nfrom pathlib import Path\n')
    adapter = once(adapter, 'def patched_schedule_tree(source):', 'VLLM_SOURCE_SHA256 = '+repr(pins)+'\n'+HELPERS+'def patched_schedule_tree(source):')
    adapter = once(adapter, 'or manager.enable_caching or manager.num_kv_cache_groups != 1', 'or not manager.enable_caching or not vllm_config.cache_config.enable_prefix_caching or manager.num_kv_cache_groups != 1')
    adapter = once(adapter, '"KVCacheCoordinatorNoPrefixCache"', '"UnitaryKVCacheCoordinator"')
    adapter = once(adapter, '    owned = singles[0].req_to_blocks', "    from vllm.v1.kv_cache_interface import FullAttentionSpec\n    spec = singles[0].kv_cache_spec\n    if (type(spec) is not FullAttentionSpec or spec.sliding_window is not None or spec.attention_chunk_size is not None\n            or not (block_size == spec.block_size == singles[0].block_size == scheduler.hash_block_size\n                    == scheduler.block_size == manager.coordinator.scheduler_block_size == pool.hash_block_size == 16)):\n        raise ValueError('requires plain full-attention and matching complete-block APC granularity')\n    owned = singles[0].req_to_blocks\n    qualify_unshared(pool, owned, singles[0])")
    adapter = once(adapter, '    namespace = dict(original.__func__.__globals__)', '    qualify_source_files(Path(source).parents[3])\n    namespace = dict(original.__func__.__globals__)')
    adapter = once(adapter, '            current["forced_preempted"].append(victim.request_id)\n', '')
    adapter = once(adapter, '            scheduler._preempt_request(victim, timestamp)', "            expected_release = releasable_blocks(pool, owned[victim.request_id])\n            free_before = pool.get_num_free_blocks()\n            scheduler._preempt_request(victim, timestamp)\n            current['forced_preempted'].append(victim.request_id)\n            actual_release = pool.get_num_free_blocks() - free_before\n            current.update(actual_released_blocks=actual_release, expected_released_blocks=expected_release)\n            if actual_release != expected_release:\n                raise RuntimeError('native preempt free delta differs from exclusive owned blocks')")
    adapter = once(adapter, '            cohort = set(scheduler.requests)', '            qualify_unshared(pool, owned, singles[0])\n            cohort = set(scheduler.requests)')
    adapter = once(adapter, '                    released = len(owned.get(victim.request_id, ()))', "                    current['pre_exchange_ownership'] = qualify_unshared(pool, owned, singles[0])\n                    released = releasable_blocks(pool, owned[victim.request_id])")
    adapter = once(adapter, 'or (r.num_computed_tokens, tuple(b.block_id for b in owned[rid])) != before):', 'or (r.num_computed_tokens, tuple(b.block_id for b in owned[rid])) != before\n                            or any(b.ref_cnt != 1 for b in owned[rid])):')
    adapter = once(adapter, '                tracker.note_rotation_applied()', "                current['post_exchange_ownership'] = qualify_unshared(pool, owned, singles[0])\n                tracker.note_rotation_applied()")
    (source/'rotation_native.py').write_text(adapter)
    probe = (source/'run_probe.py').read_text()
    probe = once(probe, 'from safe_static import qualify_safe_cap', 'from safe_static import qualify_safe_cap\nfrom absence_rotation import RotationConfig\nfrom rotation_native import install as install_rotation, qualify_source_files')
    probe = once(probe, "if args.completion_policy != 'native':", "if args.completion_policy not in ('native', 'rotate') or args.prefix_caching != 'on':")
    probe = once(probe, "only the unmodified native scheduler is permitted", "only APC-on native or frozen rotation is permitted")
    probe = once(probe, 'rotation_config=None,', 'rotation_config=vars(RotationConfig()),')
    probe = once(probe, "'metrics.py', 'safe_static.py']", "'metrics.py', 'safe_static.py', 'rotation_native.py', 'absence_rotation.py']")
    probe = once(probe, "        model = config['model']", "        qualify_source_files(Path(vllm.__file__).parent)\n        model = config['model']")
    probe = once(probe, "    out.mkdir(parents=True, exist_ok=False)", "    if len({tuple(t[:16]) for t in workload['actual_prompt_token_ids']}) != 32:\n        raise ValueError('cohort must have 32 distinct first blocks')\n    out.mkdir(parents=True, exist_ok=False)")
    original_capture = "        raw = capture_with_memory(engine, capture_episode, workload, config, allow_preemption=True,\n            regime='steady', arrival_scale=1.0, run_id='measured', max_seconds=120)"
    measure = "        decisions, uninstall = [], None\n        try:\n            if args.completion_policy == 'rotate':\n                decisions, uninstall = install_rotation(engine.engine_core.engine_core.scheduler,\n                    vllm_config=engine.vllm_config, block_size=qualification['block_size'], expected_requests=32,\n                    rotation_config=RotationConfig(**config['rotation_config']), victim_order='most_output')\n            dump(out/'status.json', dict(status='MEASURING', measurement_executed=True))\n"+'\n'.join('    '+line for line in original_capture.splitlines())+"\n        finally:\n            if uninstall is not None:\n                uninstall()\n            dump(out/'headroom-decisions.json', decisions)\n        if args.completion_policy == 'rotate' and raw['status'] == 'COMPLETE' and not any(d['forced_preempted'] for d in decisions):\n            raw.update(status='INVALID_NO_ACTION', error='frozen rotation never forced preemption')"
    probe = once(probe, "        dump(out/'status.json', dict(status='MEASURING', measurement_executed=True))\n", '')
    probe = once(probe, original_capture, measure)
    (source/'run_probe.py').write_text(probe)
    sys.path.insert(0, str(source)); module = load_module('prepared_apc_rotation', source/'rotation_native.py')
    probe_module = load_module('prepared_apc_rotation_probe', source/'run_probe.py')
    for root, domain in ((source,'short'), (source,'long'), (source/'cohorts/cohort2','long')):
        checked_config, checked_workload = probe_module.load_inputs(root, domain)
    assert checked_config['workload_sha256'] == manifest['cohorts'][0]['workload_sha256']
    assert len({tuple(t[:16]) for t in checked_workload['actual_prompt_token_ids']}) == 32
    oldtree, newtree = [ast.parse(t) for t in ((rotation_source/'rotation_native.py').read_text(), adapter)]
    for name in ('patched_schedule_tree', 'need'):
        find = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(find(oldtree)) == ast.dump(find(newtree))
    oldprobe = ast.parse((parent/'run_probe.py').read_text()); newprobe = ast.parse(probe)
    for name in ('reset_measured_cache', 'cache_adjustments'):
        assert ast.dump(find(oldprobe)) == ast.dump(find(newprobe))
    getargs = lambda tree: next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'kwargs' for t in n.targets))
    assert ast.dump(getargs(oldprobe)) == ast.dump(getargs(newprobe))
    changed = {'run_probe.py','rotation_native.py','absence_rotation.py','campaign.json','DECISIONS.md'}
    unchanged = {str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file() and p.name not in changed}
    assert all(digest(parent/n) == sha for n,sha in unchanged.items())
    for p in source.glob('*.py'):
        compile(p.read_text(), str(p), 'exec')
    labels = json.loads(subprocess.check_output([sys.executable,'-B',str(source/'run_campaign.py'),'--list'],text=True))
    assert labels == [c['label'] for c in cells]
    state_checks = cpu_state_checks(module, scheduler_source.read_text())
    write(out/'cpu-checks.json', dict(identity=dict(status='PASS', order=labels, engine_args='UNCHANGED', inputs='THREE_SETS_VERIFIED'),
        causal_cutoff=dict(status='PASS', scheduler_AST_insertion_and_need='UNCHANGED', rotation_config_sha256=digest(source/'absence_rotation.py')),
        accounting=accounting_check(source/'native_capture.py'), action_state=state_checks))
    write(out/'inheritance.json', dict(unchanged_from_apc=unchanged, unchanged_from_rotation={'absence_rotation.py':digest(source/'absence_rotation.py')},
        expected_vllm_source_sha256=pins, source_parent=str(parent.resolve()), rotation_parent=str(rotation_source.resolve())))
    shutil.copyfile(scheduler_source, out/'reference_scheduler.py')
    (source/'DECISIONS.md').write_text(PROTOCOL); (out.parent/'REPORT.md').write_text(PROTOCOL)
    command = shlex.join([sys.executable, '-B', *sys.argv])
    (out/'commands.txt').write_text('# CPU preparation (use a fresh output directory):\n'+command+'\n# On the qualified host, the external driver holds one flock across all four commands.\n# Retain every cell locally; tar/hash/readback after the whole group reaches terminal state.\npython -B run_campaign.py --list\n'+''.join(f'python -B run_one_cell.py {label}\n' for label in labels))
    archive = out/'execution.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        for p in sorted(source.iterdir()):
            if p.name != '__pycache__': bundle.add(p,arcname=p.name)
    write(out/'status.json', dict(status='PREPARED_UNRUN', uploaded=False, gpu_executions=0, labels=labels,
        archive_sha256=digest(archive), preparation_script_sha256=digest(Path(__file__)),
        source_files_sha256={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apc-run', type=Path, required=True)
    parser.add_argument('--rotation-source', type=Path, required=True)
    parser.add_argument('--scheduler-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.apc_run, args.rotation_source, args.scheduler_source, args.output_dir)

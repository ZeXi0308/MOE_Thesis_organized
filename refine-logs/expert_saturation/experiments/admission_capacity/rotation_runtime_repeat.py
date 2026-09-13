#!/usr/bin/env python3
"""Prepare/analyze four unchanged-runtime repeats after a same-path time discrepancy."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import analyze_rotation_strong_baseline as base

shared, outcome = base.shared, base.outcome
ID = '20260914_rotation_runtime_repeat_r01'
PARENT_SHA = 'f25f34fd2d1869a8b9104a9828fb7a25cdd39cc18b3b20bbfd00a3bb4dc5d24e'
SCOPE = ('Same previously observed cohort2, native/most then most/native, four fresh engines. '
         'This is an exploratory runtime repeat, not a fresh-document holdout. Retain all '
         'calls, including long calls. No cache reset, pressure warmup, drift subtraction, '
         'significance, noise bound, quality equivalence or method GO.')
PROTOCOL = '''# 同执行路径时间差：四项受控重复

PREPARED_UNRUN。前轮已观察同一most_output逻辑路径的一次恢复调用时间差；原因未定。
问题：保持输入、策略与公共预热，长调用是否再次出现，native/most完整吞吐方向是否重复？
只改变执行批次；复用已经观察过的cohort2，不声称新holdout或新运行域。
OLMoE BF16/vLLM0.26/RTX5090；KV16089350144bytes、7671usable blocks；32×3072输入/1024输出、50ms steady。
顺序：block0 native, most_output；block1 most_output, native。四项均独立引擎，原三个公共预热不变。
策略、runtime、采集、模型、输入、token预算、动作阈值全部复用父包；不清缓存，不追加压力预热，不预设冷编译原因。
整组连续交接，排在GPU_COORDINATION现有F/X及A-review之后；每格边界/初始化/测量前查占用，忙或查询失败ABORT。
已有gpu_state记录功率、温度与SM时钟。保留环境、全部调用、失败和中断；不自动重跑、不替换canonical。
全四项同资源、同输入、真实动作及请求账本合格后，按block比较native→most；同角色重复只作描述。
主指标为完整吞吐与最大ITL，同时报告平均完成/TTFT/逐请求损益、调度、重算与收尾成本。
wall=scheduler_inclusive+engine_non_schedule+outside_engine_calls；decision在scheduler内。含重算调用不是纯GPU税。
诊断按第一次成功forced交换对应请求的恢复末次调用对齐；只有逻辑轨迹相同才比较同step编号。
所有长调用保留在主结果；不减去前轮0.734617秒差值，不事后挑选快调用或仅报告有利block。
若同路径长调用重现，再加能区分host等待/设备执行的最小观测；若未重现，只记本重复未见，不宣称根因已找到。
若轨迹改变，先定位第一次实际schedule/action差；若完整收益变号，保留权衡，不扫描阈值。
固定只跑四项，不按结果继续追加。当前问题OPEN，结果上限NATIVE_SERVING/MEASUREMENT_ONLY。
最近邻系统、headroom跨域、业务SLO、质量和Oracle仍未补测；参考SLO全部通过不等于长暂停SLO成立。
'''


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate(frozen):
    manifest = outcome.read(frozen/'campaign.json')
    parent_path = frozen/'parent_campaign.json'
    outcome.require(shared.digest(parent_path) == PARENT_SHA, 'parent campaign changed')
    parent = outcome.read(parent_path)
    cohorts, inputs = base.validate_manifest(parent, frozen)
    expected = [c for c in parent['cells'] if c['role'] in ('native', 'most_output')]
    outcome.require(manifest == dict(schema_version=1, experiment_id=ID,
        parent_campaign_sha256=PARENT_SHA, cohorts=parent['cohorts'], cells=expected,
        design='same-cohort runtime repeat; A/B/B/A; no new holdout'), 'repeat design changed')
    return manifest, cohorts, inputs


def prepare(parent_run, out):
    if out.exists():
        raise FileExistsError(out)
    state = outcome.read(parent_run/'execution.json')
    outcome.require(state['status'] == 'COMPLETE' and len(state['cells']) == 8 and
        all(c['status'] == 'READ_BACK' for c in state['cells']), 'parent eight cells incomplete')
    parent = parent_run/'frozen'
    outcome.require(shared.digest(parent/'campaign.json') == PARENT_SHA, 'unexpected parent')
    out.mkdir(parents=True)
    frozen = out/'source'
    shutil.copytree(parent, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(parent/'campaign.json', frozen/'parent_campaign.json')
    manifest = outcome.read(parent/'campaign.json')
    cells = [c for c in manifest['cells'] if c['role'] in ('native', 'most_output')]
    write(frozen/'campaign.json', dict(schema_version=1, experiment_id=ID,
        parent_campaign_sha256=PARENT_SHA, cohorts=manifest['cohorts'], cells=cells,
        design='same-cohort runtime repeat; A/B/B/A; no new holdout'))
    (frozen/'DECISIONS.md').write_text(PROTOCOL)
    validate(frozen)
    unchanged = [p for p in parent.rglob('*') if p.is_file() and p.name not in ('campaign.json', 'DECISIONS.md')]
    outcome.require(all(p.read_bytes() == (frozen/p.relative_to(parent)).read_bytes() for p in unchanged), 'runtime/input changed')
    labels = json.loads(subprocess.check_output([sys.executable, '-B', str(frozen/'run_campaign.py'), '--list'], text=True))
    outcome.require(labels == [c['label'] for c in cells] and len(labels) == 4, 'launcher order differs')
    archive = out/'execution.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for p in sorted(frozen.iterdir()):
            bundle.add(p, arcname=p.name)
    write(out/'status.json', dict(status='PREPARED_UNRUN', uploaded=False, gpu_executions=0,
        archive_sha256=shared.digest(archive), labels=labels, parent_run=str(parent_run.resolve()),
        unchanged_parent_files={str(p.relative_to(parent)):shared.digest(p) for p in unchanged},
        scope=SCOPE, preparation_script_sha256=shared.digest(Path(__file__))))


def analyze(run):
    frozen = run/'frozen'
    if not frozen.exists():
        frozen = run.parent/'preparation/source'
    manifest, cohorts, inputs = validate(frozen)
    outcome.module('metrics', frozen/'metrics.py')
    outputs = Path(__file__).resolve().parents[2]/'outputs/admission_capacity'
    native = outcome.module('runtime_repeat_native_helpers', outputs/'20260908_native_preemption_r01/analyze_native_preemption.py')
    rows = [base.inspect_cell(run, c, native, frozen, cohorts[c['cohort_id']]) for c in manifest['cells']]
    qualified = all(c['full_episode_comparison_eligible'] for c in rows)
    signatures = [(c['engine_args'], {k:v for k,v in c['config'].items() if k not in ('completion_policy', 'rotation_victim_order')}) for c in rows] if qualified else []
    common = qualified and all(s == signatures[0] for s in signatures)
    index = {(c['block'],c['role']):c for c in rows}
    def pair(a, b, kind):
        return shared.compare_cells(a, b, common, native, kind, additional_config_exclusions=('rotation_victim_order',))
    pairs = [pair(index[(b,'native')], index[(b,'most_output')], 'within_block_policy') for b in (0,1)]
    repeats = [pair(index[(0,r)], index[(1,r)], 'same_role_across_blocks') for r in ('native','most_output')]
    eligible = bool(common and all(p['status'] == 'DESCRIPTIVE_MATCHED_PAIR' for p in pairs+repeats))
    status = 'MEASUREMENT_ONLY' if eligible else 'UNRUN' if all(c['status']=='UNRUN' for c in rows) else 'INVALID_NO_ACTION' if any(c['status']=='INVALID_NO_ACTION' for c in rows) else 'INVALID_EVIDENCE' if qualified or any(c['status']=='INVALID_EVIDENCE' for c in rows) else 'INCOMPLETE_CAMPAIGN'
    recovery = [base.recovery.cell(run, c['label'], native, c, policy=c['policy'], campaign=True) for c in rows] if eligible else []
    return dict(status=status, all_four_cells_qualified=qualified, common_execution_config=common,
        comparisons_eligible=eligible, campaign_manifest=manifest, cohort_inputs=inputs, cells=rows,
        comparisons=pairs, native_aa=[], same_role_repeats=repeats, recovery_accounting=recovery,
        scope=SCOPE, pairing_rule='All four must qualify; native/most per block, never replace or correct baseline.',
        cost_scope='Full wall retained; scheduler inclusive + engine non schedule + outside engine; no long-call subtraction.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare','analyze'))
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.mode == 'prepare':
        prepare(args.run_dir, args.output_dir)
    else:
        result = analyze(args.run_dir)
        args.output_dir.mkdir(parents=True)
        write(args.output_dir/'analysis.json', result)
        (args.output_dir/'report.md').write_text(shared.readable(result).replace('# Independent-cohort rotation controls', '# Same-cohort runtime repeat', 1))
        print(json.dumps(dict(status=result['status'], comparisons_eligible=result['comparisons_eligible'])))


if __name__ == '__main__':
    main()

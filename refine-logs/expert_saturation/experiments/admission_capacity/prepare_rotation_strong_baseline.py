#!/usr/bin/env python3
"""Freeze the fresh-document eight-cell strong-baseline comparison."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from prepare_rotation_victim_order import sha, write, replace_once

ORDER = ('native', 'headroom', 'most_output', 'native_aa')
PROTOCOL = '''# 持续驱逐排序：新文档上的强基线对照

PREPARED_UNRUN；输入、比较、八项顺序在本轮GPU数据前冻结。
问题：同实际KV预算，持续most_output是否相对native与fast completion_headroom改善最长输出暂停—完整服务量权衡？
沿用已测持续排序，不新增controller；首次交换方案已显示局部width2减少可被width1增加抵消。

## 固定资源、输入和动作

OLMoE BF16 / vLLM0.26 / 单RTX5090；实际KV16,089,350,144 bytes / 7671 usable blocks。
32请求、3072输入/1024输出、50ms steady到达、cap32、token budget1024。
新增cohort2取固定WikiText shard中原96文档之后的下一32篇足长完整源文章；执行其3072-token前缀。
原96文档逐字及token重现，并排除文档hash、输入前缀hash和源行区间重叠。凭据见fresh_inputs_receipt.json。
这是同语料、同长度/到达域的新文档检验，不是独立总体或新运行域。
四角色：native；native_aa（完全相同策略）；fast headroom；持续most_output轮转。
轮转保持等待30/交换冷却20/恢复驻留30/近完成保护0.90/最多8次缺席，以及完整历史资金、首新输出保护和最长缺席恢复。
most_output只按当时原生已生成输出数选择合格受害者；非未来信息，不计重算，不代表客户端消费。
每格独立引擎、共同预热、策略独立演进请求/KV/batch/输出；不共享未来轨迹。

## 八项顺序与分析

block0：native, headroom, most_output, native_aa。
block1：native_aa, most_output, headroom, native。
全八格完成、同输入/资源/计时合格、headroom与most实际动作合格后，逐block比较native→headroom、native→most、headroom→most。
两对native→native_aa、四对同角色跨block作为运行漂移保留。native始终为预先指定主基线，不以A/A替代、不减A/A、不把两重复差写成噪声界。
不能把8引擎或256次请求执行当8独立workload；同一32文档的请求差值也不是独立策略重复。

## 目标、成本与停止规则

主目标仍为最大ITL与完整吞吐，同时报告TTFT、平均完成、每请求损益、失败/完成、重算、恢复及输出变化。
wall = scheduler-inclusive + engine-non-schedule + outside-engine；decision是scheduler子集。
含重算的调用可能同时推进其它请求的新decode，不把整段算为纯GPU重算税。收尾宽度变化是观察到的成本，不能跨策略直接相加为因果saving。
若持续排序相对headroom仍有一致收益，保留其强简单策略资格；若有代价或变号，按完整成本报告，不改变主指标或删格。
若无实际干预，标INVALID_NO_ACTION并区分运行域与实现失败；若输入/资源/账本不合格，标INVALID_EVIDENCE。
本轮不追加参数/文档搜索；不以任意百分比判GO，不称硬暂停界、吞吐非劣、生产p99、质量等价、公平保证或已证明新颖性。
参考TTFT5s/平均TPOT0.2s不约束最大ITL；全通过时goodput退化为吞吐，不能据此证明长暂停SLO改善。
Oracle/相邻系统直接对照/第二模型/动态到达范围仍未测；结果仅NATIVE_SERVING、MEASUREMENT_ONLY。

## 执行与保留

现有connect.weste.seetacloud.com:23478。初始化及正式测量前检查GPU占用；忙碌或查询失败ABORT，不终止其它任务。
每格归档回传核验后推进下一格；所有真实尝试与失败保留，不自动重跑、替换canonical或覆盖raw，修正写addendum。
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--holdout-run', required=True, type=Path)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    source, out = args.holdout_run/'frozen', args.output_dir
    if out.exists(): raise FileExistsError(out)
    state = json.loads((args.holdout_run/'execution.json').read_text())
    if state['status'] != 'COMPLETE' or len(state['cells']) != 20:
        raise ValueError('requires completed twenty-cell baseline source')
    config = json.loads((args.inputs/'config.json').read_text())
    receipt = json.loads((args.inputs/'inputs_report.json').read_text())
    assert receipt['cohort_id'] == 'cohort2' and receipt['workload_sha256'] == config['workload_sha256']
    out.mkdir(parents=True); frozen = out/'source'
    shutil.copytree(source, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'results'))
    shutil.rmtree(frozen/'cohorts')
    prepared = frozen/'cohorts/cohort2/inputs_preparation/prepared/long'
    prepared.mkdir(parents=True)
    for name in ('config.json', 'workload.json'): shutil.copyfile(args.inputs/name, prepared/name)
    shutil.copyfile(args.inputs/'inputs_report.json', frozen/'fresh_inputs_receipt.json')
    for name in ('absence_rotation.py', 'rotation_native.py'):
        shutil.copyfile(Path(__file__).resolve().parent/name, frozen/name)
    cells = [dict(label=f'cohort2-block{b}-{r}', cohort_id='cohort2', block=b, role=r,
        completion_policy='rotate' if r == 'most_output' else 'native' if r == 'native_aa' else r,
        cap=32, victim_order='most_output' if r == 'most_output' else 'least_progress')
        for b, roles in ((0, ORDER), (1, tuple(reversed(ORDER)))) for r in roles]
    manifest = dict(schema_version=4, experiment_id='20260913_rotation_strong_baseline_r01',
        cohorts=[dict(id='cohort2', input_root='cohorts/cohort2', workload_sha256=config['workload_sha256'])], cells=cells,
        novelty_inputs=dict(receipt_path='fresh_inputs_receipt.json', receipt_sha256=sha(frozen/'fresh_inputs_receipt.json'), excluded_prior_requests=96),
        parent_campaign_manifest_sha256=sha(source/'campaign.json'), design='fresh32 documents; four roles, two reverse-order blocks')
    write(frozen/'campaign.json', manifest); (frozen/'DECISIONS.md').write_text(PROTOCOL)
    probe = replace_once((frozen/'run_probe.py').read_text(), "choices=['cohort0', 'cohort1']", "choices=['cohort2']")
    probe = replace_once(probe, '    args = parser.parse_args()', "    parser.add_argument('--victim-order', choices=['least_progress', 'most_output'], required=True)\n    args = parser.parse_args()\n    expected_order = 'most_output' if args.completion_policy == 'rotate' else 'least_progress'\n    if args.completion_policy not in ('native', 'headroom', 'rotate') or args.victim_order != expected_order or args.cap != 32 or args.domain != 'long' or args.gpu_memory_utilization != 0.9:\n        raise ValueError('frozen strong-baseline role/long/cap32/budget90 required')")
    probe = replace_once(probe, "preemption_mode='native_recompute', rotation_config=vars(RotationConfig()))", "preemption_mode='native_recompute', rotation_config=vars(RotationConfig()),\n                  rotation_victim_order=args.victim_order)")
    probe = replace_once(probe, "expected_requests=32, rotation_config=RotationConfig(**config['rotation_config']))", "expected_requests=32, rotation_config=RotationConfig(**config['rotation_config']),\n                victim_order=config['rotation_victim_order'])")
    (frozen/'run_probe.py').write_text(probe)
    launcher = replace_once((frozen/'run_campaign.py').read_text(), "'--output-dir', str(out / label)]", "'--victim-order', cell['victim_order'], '--output-dir', str(out / label)]")
    (frozen/'run_campaign.py').write_text(launcher.replace('frozen twenty-cell order', 'frozen eight-cell order'))
    for path in frozen.rglob('*.py'): compile(path.read_text(), str(path), 'exec')
    observed = json.loads(subprocess.check_output([sys.executable, '-B', str(frozen/'run_campaign.py'), '--list'], text=True))
    assert observed == [c['label'] for c in cells]
    unchanged = ('completion_headroom.py', 'native_capture.py', 'memory_telemetry.py', 'metrics.py', 'safe_static.py', 'run_one_cell.py')
    assert all(sha(source/n) == sha(frozen/n) for n in unchanged)
    warmups = {str(p.relative_to(frozen)):sha(p) for p in (frozen/'inputs_preparation').rglob('*.json')}
    assert all(sha(source/n) == digest for n,digest in warmups.items())
    from analyze_rotation_strong_baseline import validate_manifest
    validate_manifest(manifest, frozen)
    archive = out/'execution.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(frozen.iterdir()): bundle.add(path, arcname=path.name)
    write(out/'status.json', dict(status='PREPARED_UNRUN', uploaded=False, gpu_executions=0,
        archive_sha256=sha(archive), labels=observed, cohorts=manifest['cohorts'], unchanged_modules=list(unchanged),
        input_files_sha256={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*.json') if 'inputs_preparation' in p.parts},
        unchanged_warmup_files_sha256=warmups, source_files_sha256={p.name:sha(p) for p in frozen.glob('*.py')},
        novelty_inputs=manifest['novelty_inputs'], command=[sys.executable]+sys.argv,
        preparation_script_sha256=sha(Path(__file__)), created_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='PREPARED_UNRUN', cells=len(cells), archive_sha256=sha(archive))))


if __name__ == '__main__': main()

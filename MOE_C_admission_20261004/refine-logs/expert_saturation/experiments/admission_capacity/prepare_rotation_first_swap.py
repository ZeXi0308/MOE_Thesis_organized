#!/usr/bin/env python3
"""Freeze one-cohort first-exchange diagnosis from the completed victim campaign."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from prepare_rotation_victim_order import sha, write, replace_once

ROLES = ('least_progress', 'first_most_then_least', 'most_output')
PROTOCOL = '''# 首次驱逐选择与持续改序的因果诊断

PREPARED_UNRUN；六项顺序与解释规则在新GPU结果前冻结。
问题：一次不同的驱逐选择是否已足以压缩收尾，同时避免持续改序让早完成请求反复等待？
A least_progress：每次合格交换选择计算进度比例最小者。
B most_output：每次选择当前原生输出token数最多者。
C first_most_then_least：仅第一次实际成功强制交换使用B，之后使用A。
“首次”由已成功执行的强制交换数决定，不由绝对step、未来轨迹或事后指定request决定。
noop、资金不足的提案、自然抢占、原生恢复均不消耗首次机会。运行失败保留，不靠重启重置策略。

## 不变量与范围

OLMoE BF16 / vLLM0.26 / 单RTX5090；实际KV16,089,350,144 bytes / 7671 usable blocks。
32请求、3072输入/1024输出、50ms到达、cap32、token budget1024。
最长缺席恢复、等待30/交换冷却20/恢复驻留30、近完成保护0.90、最多8次缺席、完整历史资金和首新输出前保护不变。
每格独立引擎、相同预热、策略独立演进请求/KV/batch/输出。计数来自request.num_output_tokens，不含重算，不等于消费或质量。
仅复用既有cohort0的32篇已见文本；一cohort两顺序block六执行，不是六个独立workload或新holdout。
旧八项仅提供假说，不能替代本轮A/B/C完整请求比较；本轮不重跑native/headroom/safe29，不冒称完整公平或prior-art基线。

## 六项固定顺序

block0 A,C,B；block1 B,C,A。label及具体参数以campaign.json为准。
全六格完成且输入、资源、请求、计时与模式转换均合格后，逐block比较A→C、B→C、A→B，再保留三对同角色跨block差值。
记录每step的effective_victim_order与此前实际强制交换数；C首次成功交换为B，随后全为A，未应用提案不能推进计数。
若C无实际交换，记INVALID_NO_ACTION；若模式转换不符，记INVALID_EVIDENCE，不拿不生效实验判机制失败。

## 测量与判断

主目标仍为max-ITL与完整服务量；同时报告TTFT、平均完成、每请求损益、所有完成/失败、输出差异。
核对原末两请求的实际服务进度、完成次序、width1/2/4收尾、实际重算及互斥host成本。
wall = scheduler-inclusive + engine-non-schedule + outside-engine；decision是scheduler子集。重算调用也可包含其它请求的新decode。
若C保持B类收尾变化且减少对早完成请求代价，支持首次选择贡献；若无此结构或净代价仍在，首次选择不足，按实际后续链解释。
结构变化不等于显著性能收益；近零/变号保持未确认，不设事后GO门槛，不把两次运行差当噪声界，不从策略差中减A/A。
此轮不追加阈值/次数搜索；不宣称暂停硬界、质量、native非劣、公平保证、新颖性或方法GO。

## 执行与保留

现有connect.weste.seetacloud.com:23478；初始化与测量前检查GPU占用，失败即保留ABORT，不杀其它任务，不自动重跑。
每格归档回传核验后才推进下一格；所有结果保留，原始结果不覆盖，修正另写addendum。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--victim-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.victim_run/'frozen', args.output_dir
    if out.exists(): raise FileExistsError(out)
    state = json.loads((args.victim_run/'execution.json').read_text())
    if state['status'] != 'COMPLETE' or len(state['cells']) != 8:
        raise ValueError('requires completed eight-cell victim campaign')
    old = json.loads((source/'campaign.json').read_text())
    out.mkdir(parents=True); frozen = out/'source'
    shutil.copytree(source, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'results'))
    shutil.rmtree(frozen/'cohorts/cohort1')
    for name in ('absence_rotation.py', 'rotation_native.py'):
        shutil.copyfile(Path(__file__).resolve().parent/name, frozen/name)
    cells = [dict(label=f'cohort0-block{block}-{role}', cohort_id='cohort0', block=block,
                  role=role, completion_policy='rotate', cap=32, victim_order=role)
             for block, roles in ((0, ROLES), (1, tuple(reversed(ROLES)))) for role in roles]
    manifest = dict(schema_version=3, experiment_id='20260913_rotation_first_swap_r01',
                    cohorts=[old['cohorts'][0]], cells=cells,
                    parent_campaign_manifest_sha256=sha(source/'campaign.json'),
                    design='one reused cohort, three roles, two reversed-order blocks',
                    only_treatment_config_key='rotation_victim_order')
    write(frozen/'campaign.json', manifest); (frozen/'DECISIONS.md').write_text(PROTOCOL)
    probe = replace_once((frozen/'run_probe.py').read_text(),
        "choices=['least_progress', 'most_output']", "choices=['least_progress', 'most_output', 'first_most_then_least']")
    (frozen/'run_probe.py').write_text(probe)
    launcher = (frozen/'run_campaign.py').read_text().replace('frozen eight-cell order', 'frozen six-cell order')
    (frozen/'run_campaign.py').write_text(launcher)
    for path in frozen.rglob('*.py'): compile(path.read_text(), str(path), 'exec')
    observed = json.loads(subprocess.check_output([sys.executable, '-B', str(frozen/'run_campaign.py'), '--list'], text=True))
    assert observed == [c['label'] for c in cells]
    unchanged = ('completion_headroom.py', 'native_capture.py', 'memory_telemetry.py', 'metrics.py', 'safe_static.py', 'run_one_cell.py')
    assert all(sha(source/name) == sha(frozen/name) for name in unchanged)
    input_hashes = {str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*.json') if 'inputs_preparation' in p.parts}
    assert all(sha(source/name) == value for name, value in input_hashes.items())
    archive = out/'execution.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(frozen.iterdir()): bundle.add(path, arcname=path.name)
    write(out/'status.json', dict(status='PREPARED_UNRUN', uploaded=False, gpu_executions=0,
        archive_sha256=sha(archive), labels=observed, cohorts=manifest['cohorts'],
        unchanged_modules=list(unchanged), input_files_sha256=input_hashes,
        source_files_sha256={p.name:sha(p) for p in frozen.glob('*.py')},
        command=[sys.executable]+sys.argv, created_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='PREPARED_UNRUN', cells=len(cells), archive_sha256=sha(archive))))


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Freeze the two-rule victim-order ablation from the completed holdout source."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ORDER = [(0,0,'least_progress'), (0,0,'most_output'),
         (1,0,'most_output'), (1,0,'least_progress'),
         (0,1,'most_output'), (0,1,'least_progress'),
         (1,1,'least_progress'), (1,1,'most_output')]
PROTOCOL = '''# 驱逐服务量排序消融

PREPARED_UNRUN；本协议及八项顺序在本轮GPU结果前冻结。
问题：同KV与恢复规则下，驱逐对象排序如何改变最长暂停、完整请求吞吐与完成代价？
A least_progress：现有合格running集合内，计算进度比例最小者先驱逐。
B most_output：同一合格集合内，当前已产生输出token最多者先驱逐。同分均按request ID升序。
输出计数取原生request.num_output_tokens，不计重算位置，也不是客户端消费或质量指标。

## 共同条件

OLMoE BF16 / vLLM0.26 / RTX5090；实际KV 16,089,350,144 bytes、7,671 usable blocks；32请求、3072输入/1024输出、50ms到达，token budget1024、engine/admission cap32。最长缺席优先恢复；等待30、交换冷却20、恢复后驻留30、近完成保护0.90、每请求最多8次缺席；完整历史资金检查与恢复到首个新输出的保护相同。
复用20项实验的cohort0/1与旧预热输入，各格fresh engine；这是已看过文本上的探索性组件消融，不是新holdout或新运行域。旧20项只作历史依据，不替代本次A/B配对。
唯一可变项为rotation_victim_order；不能强制B复用A之后的动作时刻、KV、路由、请求集合或输出。

## 八项顺序与比较

cohort0/block0 A,B；cohort1/block0 B,A；cohort0/block1 B,A；cohort1/block1 A,B。按campaign.json执行，不因结果重抽、替换或删格。
全部8格完成且输入/资源/计时/动作资格成立后，比较4个同cohort/block A→B配对，另保留4个同角色跨block差值；不跨cohort配请求。
报告完整吞吐、平均完成、每请求max-ITL/TTFT及受损请求、全部完成/失败、抢占/恢复/重算与互斥host成本。近零差或变号保持未确认；重复差不是总体噪声界，不减去旧A/A，不设事后百分比GO阈值。
比较实际驱逐对象/恢复路径和首个动作差异；无选择差异只能说明该域未暴露排序作用。无实际轮转保留INVALID_NO_ACTION，区分无动作域与实现失败。
有利结果仅支持具体排序在此域的作用；不称公平保证、VTC/FastServe复现、质量等价、native吞吐非劣、显著性或方法GO。正式最近邻策略比较仍缺客户端/服务计费/触发语义。

## 资格与留存

GPU初始化及正式测量前检查其它计算进程；查询失败/占用即保留ABORT，不杀他人、不自动重跑。每格归档回传核验后才启动下一格，保留失败与不利结果，原件不覆盖。原始raw只读，修正另写addendum。
若结果不利，先用实际恢复和工作账本区分排序、重算、批宽收尾和实现开销；本轮不追加阈值/文档搜索。只有本次同底座消融完成后，再决定下一因果环节。
'''

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, data): path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
def replace_once(text, old, new):
    if text.count(old) != 1: raise ValueError('source replacement not unique: '+old)
    return text.replace(old, new)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--holdout-run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args=p.parse_args(); source=args.holdout_run/'frozen'; out=args.output_dir
    if out.exists(): raise FileExistsError(out)
    state=json.loads((args.holdout_run/'execution.json').read_text())
    if state['status']!='COMPLETE' or len(state['cells'])!=20:
        raise ValueError('requires completed holdout')
    old=json.loads((source/'campaign.json').read_text())
    out.mkdir(parents=True); frozen=out/'source'
    shutil.copytree(source, frozen, ignore=shutil.ignore_patterns('__pycache__','*.pyc','results'))
    module_root=Path(__file__).resolve().parent
    for name in ('absence_rotation.py','rotation_native.py'):
        shutil.copyfile(module_root/name,frozen/name)
    cells=[dict(label=f'cohort{c}-block{b}-{role}',cohort_id=f'cohort{c}',block=b,
                role=role,completion_policy='rotate',cap=32,victim_order=role) for c,b,role in ORDER]
    manifest=dict(schema_version=2,experiment_id='20260913_rotation_victim_order_r01',
                  cohorts=old['cohorts'],cells=cells,
                  parent_campaign_manifest_sha256=sha(source/'campaign.json'),
                  design='same-document exploratory victim-order ablation; balanced fixed AB/BA blocks',
                  only_treatment_config_key='rotation_victim_order')
    write(frozen/'campaign.json',manifest); (frozen/'DECISIONS.md').write_text(PROTOCOL)
    probe=(frozen/'run_probe.py').read_text()
    probe=replace_once(probe,"    args = parser.parse_args()", "    parser.add_argument('--victim-order', choices=['least_progress', 'most_output'], required=True)\n    args = parser.parse_args()\n    if args.completion_policy != 'rotate' or args.cap != 32 or args.domain != 'long' or args.gpu_memory_utilization != 0.9:\n        raise ValueError('frozen victim-order ablation requires rotate/long/cap32/budget90')")
    probe=replace_once(probe,"preemption_mode='native_recompute', rotation_config=vars(RotationConfig()))", "preemption_mode='native_recompute', rotation_config=vars(RotationConfig()),\n                  rotation_victim_order=args.victim_order)")
    probe=replace_once(probe,"expected_requests=32, rotation_config=RotationConfig(**config['rotation_config']))", "expected_requests=32, rotation_config=RotationConfig(**config['rotation_config']),\n                victim_order=config['rotation_victim_order'])")
    (frozen/'run_probe.py').write_text(probe)
    launcher=(frozen/'run_campaign.py').read_text()
    launcher=replace_once(launcher,"'--output-dir', str(out / label)]", "'--victim-order', cell['victim_order'], '--output-dir', str(out / label)]")
    launcher=launcher.replace('frozen twenty-cell order','frozen eight-cell order')
    (frozen/'run_campaign.py').write_text(launcher)
    for path in frozen.rglob('*.py'): compile(path.read_text(),str(path),'exec')
    observed=json.loads(subprocess.check_output([sys.executable,'-B',str(frozen/'run_campaign.py'),'--list'],text=True))
    assert observed==[c['label'] for c in cells]
    unchanged=[name for name in ('completion_headroom.py','native_capture.py','memory_telemetry.py','metrics.py','safe_static.py','run_one_cell.py') if sha(source/name)==sha(frozen/name)]
    assert len(unchanged)==6
    input_hashes={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*.json') if 'inputs_preparation' in p.parts}
    assert all(sha(source/name)==digest for name,digest in input_hashes.items())
    archive=out/'execution.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        for path in sorted(frozen.iterdir()): bundle.add(path,arcname=path.name)
    write(out/'status.json',dict(status='PREPARED_UNRUN',uploaded=False,gpu_executions=0,
        archive_sha256=sha(archive),labels=observed,cohorts=manifest['cohorts'],
        unchanged_modules=unchanged,input_files_sha256=input_hashes,
        source_files_sha256={p.name:sha(p) for p in frozen.glob('*.py')},
        command=[sys.executable]+sys.argv,created_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='PREPARED_UNRUN',cells=len(cells),archive_sha256=sha(archive))))

if __name__=='__main__': main()

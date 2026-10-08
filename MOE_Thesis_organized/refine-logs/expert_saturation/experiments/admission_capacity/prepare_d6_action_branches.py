"""Prepare the single-event three-branch diagnostic from the qualified KV package."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import tarfile

root=Path(__file__).resolve().parents[2]
o=root/'outputs/admission_capacity';source=o/'20260914_d6_action_state_r01'
p=o/'20260914_d6_action_branches_r01';pkg=p/'pkg'
if p.exists():raise FileExistsError(p)
shutil.copytree(source/'pkg',pkg,ignore=shutil.ignore_patterns('__pycache__','SHA256SUMS'))
shutil.copy2(source/'readback/results/repeat0/action-state.json',pkg/'reference_action_state.json')
def patch(name,old,new):
 f=pkg/name;s=f.read_text()
 if s.count(old)!=1:raise ValueError((name,old,s.count(old)))
 f.write_text(s.replace(old,new))
patch('native_capture.py','observation_end_s=now(), regime=regime,','host_clock_origin_perf_counter=origin, observation_end_s=now(), regime=regime,')
patch('rotation_native.py','state_callback=None):','state_callback=None, branch_action="least"):' )
patch('rotation_native.py','decisions, cohort, protected, output_at_start = [], None, None, None','decisions, cohort, protected, output_at_start = [], None, None, None\n    deferred_once = False')
patch('rotation_native.py','nonlocal cohort, protected, output_at_start, current, proposed_victim, held_before','nonlocal cohort, protected, output_at_start, current, proposed_victim, held_before, deferred_once')
patch('rotation_native.py','            state_callback()','            state_callback()\n            current["branch_action_start_perf_counter"] = time.perf_counter()')
patch('rotation_native.py','''                        protected, proposed_victim = target, victim
                        output_at_start = target.num_output_tokens''','''                        if branch_action == "defer" and not deferred_once:
                            tracker.last_swap_step = previous_swap
                            deferred_once = True
                            current["deferred_first_action"] = True
                        else:
                            protected, proposed_victim = target, victim
                            output_at_start = target.num_output_tokens''')
patch('run_probe.py',"    args = parser.parse_args()","    parser.add_argument('--branch-action', choices=['least','most','defer'], required=True)\n    args = parser.parse_args()\n    if args.completion_policy != 'rotate': raise ValueError('branch requires rotation')\n    args.victim_order = 'first_most_then_least' if args.branch_action == 'most' else 'least_progress'")
patch('run_probe.py','rotation_victim_order=args.victim_order,','rotation_victim_order=args.victim_order, branch_action=args.branch_action,')
patch('run_probe.py','victim_order=args.victim_order,\n                state_callback=', 'victim_order=args.victim_order, branch_action=args.branch_action,\n                state_callback=')
patch('run_probe.py','out/"action-state.json"))','out/"action-state.json", reference=root/"reference_action_state.json"))')
patch('recovery_action_state.py','block_size, output):','block_size, output, reference=None):')
patch('recovery_action_state.py',"    with output.open('x') as f:json.dump(result,f,indent=2)",'''    if reference is not None:
        baseline=json.loads(reference.read_text())
        def logical(s):
            s=json.loads(json.dumps(s))
            for r in s['requests'].values():r.pop('blocks')
            return s
        same_state=logical(state)==logical(baseline['state'])
        same_kv=all(layers[n]['requests']==baseline['layers'][n]['requests'] for n in layers)
        result['reference_match']=dict(logical_state=same_state,effective_kv=same_kv)
        if not (same_state and same_kv):result['status']='MISMATCH'
    with output.open('x') as f:json.dump(result,f,indent=2)
    if result['status']!='CAPTURED':raise RuntimeError('pre-action state differs from frozen reference')''')
cells=[dict(label=f'block{b}-{arm}',block=b,action=arm) for b,arms in [(0,['least','most','defer']),(1,['defer','most','least'])] for arm in arms]
protocol='''# d6单事件动作分支（探索性因果诊断）
问题：同一已核对的请求/有效KV前态，第一次victim选择或延迟一次如何影响恢复与完整未来？
复用旧32文档、d6 6656可用块、32cap、3072/1024、50ms、APC off。
least：第一次与后续均least。most：仅第一次实际成功交换most，其后least。defer：第一次已合格且可资助的交换暂不执行，恢复last_swap_step，下一调用重新决策，此后least。保护/冷却/驻留/近完成阈值均不变。
第329步前每臂真实采集有效KV与请求状态，必须匹配上一资格repeat0，失败保留MISMATCH且不做动作比较。前态检查不能证明整个引擎checkpoint或所有隐藏状态一致。
两block完全反序least/most/defer与defer/most/least，共六格；每格新引擎、相同预热、全结果保留，不自动重试。旧d2 first_most_then_least结果不被当作未测，本轮是高压单事件分支，不是新算法。
主要量：校验结束且动作尚未决定时到32请求各自完成的时间及其均值/总和、最后完成、目标首新token等待、受害请求剩余完成；报告完整raw，绝不把校验扰动后的整段ITL/吞吐当机制性能。只有完全位于动作后区间的token间隔可单独描述，跨校验间隔不掩盖而单列。
冻结短期预测：目标已有3274个token位置待恢复，32运行中的一次victim替换留下30个其它decode，每调用至多994目标位置，ceil(3274/994)=4；两victim释放207/213块均有充足完整恢复余量，4步内其它30请求至多各增长1块。因此least/most目标首新token预计第4次调用返回；defer预计第5次（含一步延迟）。这是基于已见状态的探索性模型，不是独立holdout预测，也不预测完整完成排序。
完整未来效应未知，不预设most有益；平均或wall变号/接近波动不得宣称可分辨或非劣。无质量、SLO、一般Oracle或方法GO。
GPU同一共同flock，每格忙碌/查询失败ABORT，完整组结束释放。当前UNRUN。
'''
(p/'DECISIONS.md').write_text(protocol);(pkg/'DECISIONS.md').write_text(protocol)
script=(pkg/'run.sh').read_text().replace('/root/d6-action-state-20260914-r01','/root/d6-action-branches-20260914-r01')
script=script.replace('for label in repeat0 repeat1; do','for label in block0-least block0-most block0-defer block1-defer block1-most block1-least; do\n action=${label#*-}')
script=script.replace('--victim-order least_progress','--victim-order least_progress --branch-action "$action"');(pkg/'run.sh').write_text(script)
for f in pkg.glob('*.py'):ast.parse(f.read_text())
files={str(f.relative_to(pkg)):hashlib.sha256(f.read_bytes()).hexdigest() for f in pkg.rglob('*') if f.is_file()}
(pkg/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in sorted(files.items())))
with tarfile.open(p/'package.tar.gz','w:gz') as t:t.add(pkg,arcname='pkg')
(p/'STATUS.json').write_text(json.dumps(dict(status='CPU_PREPARED_UNRUN',cells=cells,files=files,gpu_runs=0,uploaded=False,package_sha256=hashlib.sha256((p/'package.tar.gz').read_bytes()).hexdigest()),indent=2)+'\n')
print(p)

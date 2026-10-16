"""Extract the six selected historical decisions with immutable source receipts."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
B=ROOT/'B_recovery_order_20261004'
OUT=Path(__file__).resolve().parent
rows=[
 dict(arm='Full-tail reservation',heading='## 尾段预留固定1024输出 ABBA：排除输出量后，停止当前机制',
      metrics='tail_reservation/session-fixed-20261007-r01/tail-metrics.json',
      observation='ACK-to-allocation failures disappeared; whole-service effects changed sign across the two pairs.',
      explanation='Holding the target history did not prevent its preemption during residual computation; peer growth remained outside the reservation.'),
 dict(arm='Once versus repeat8',heading='### 2026-10-08 预算1/8实际对照：更短局部间隔，重复抢占与长请求代价',
      metrics=None,
      observation='All 16 bypasses produced output then re-preempted; selected short requests improved while bypassed long requests waited longer.',
      explanation='Short recovery pulses divided gaps but did not sustain service; fewer request-seconds of recovery did not imply faster completion.'),
 dict(arm='Bounded retry deferral',heading='## 修正版重试门控 r03：一次真实完成释放，反序零动作',
      metrics='recovery_retry_defer/session-20261008-r03/defer-metrics.json',
      observation='One 74 ms wait ended at a real completion; the second candidate run had no qualifying action.',
      explanation='The action population was narrow and mean differences preceded intervention; stable whole-service benefit was not established.'),
 dict(arm='Yield across ACK',heading='## 新主机ACK最多两轮实际结果：r05',
      metrics='recovery_start_yield/session-20261008-r05/yield-ack-metrics.json',
      observation='Two breaks crossed ACK, but the target emitted only one token before another preemption.',
      explanation='A different running request exhausted capacity through decode growth; earlier ACK access did not provide continuing headroom.'),
 dict(arm='Unique8 versus repeat8',heading='## 去重复动作价值探针：实际LOAD减少，但完整流畅度变坏（2026-10-08）',
      metrics='recovery_repeat_unique/session-20261008-r01/repeat-unique-metrics.json',
      observation='Removing repeated bypasses reduced LOAD traffic but increased maxgap P99 by 35.6 and 33.1 percent.',
      explanation='Repeated outputs had split long gaps; suppressing those useful service pulses traded away smoothness without stable completion gains.'),
 dict(arm='One-donor exchange',heading='## 受控长目标交换：实际一token服务，随后等待回迁（2026-10-09完整结果）',
      metrics='recovery_capacity_exchange/session-after-failure-20261009-r01/capacity-exchange-metrics.json',
      observation='One controlled exchange produced one target token, then re-preemption; donor flow increased by 4.31 s.',
      explanation='Initial T/R/G covered entry but missed later peer growth. Costs shifted to the donor and peers; one active run is not replicated evidence.')
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tex(s):return s.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
source=B/'RESULTS.md';text=source.read_text()
for row in rows:
 assert text.count(row['heading'])==1,row['heading']
 row['results_line']=text[:text.index(row['heading'])].count('\n')+1
 if row['metrics']:
  p=B/row['metrics'];assert p.exists();json.loads(p.read_text());row['metrics_sha256']=sha(p)
(OUT/'data.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),
 scope='Historical B fixed-output development studies, not new HORIZON runs or natural-generation quality evidence. Interpretations are limited to the documented interventions.',rows=rows),ensure_ascii=False,indent=2)+'\n')
lines=[r'\begin{table}[ht]',r'\centering\small',
 r'\caption{Historical narrow interventions and their limits (T2). These are fixed-output B studies, not HORIZON performance results. Source sections and metric hashes are retained in the reproduction data.}',
 r'\label{tab:historical}',r'\begin{tabular}{>{\raggedright\arraybackslash}p{.17\textwidth}>{\raggedright\arraybackslash}p{.34\textwidth}>{\raggedright\arraybackslash}p{.36\textwidth}}',r'\toprule',
 r'Intervention & Observation & Explanation and limit \\',r'\midrule']
for row in rows:lines.append(' & '.join(tex(row[k]) for k in ('arm','observation','explanation'))+r' \\[4pt]')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
(OUT/'table.tex').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(rows=len(rows),source_sha256=sha(source))))

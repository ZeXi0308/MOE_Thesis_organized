"""Action-level comparison from verified primary sources; not a reproduction claim."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROWS=[
 dict(name='MC-SF',url='https://arxiv.org/html/2502.07115v5',locator='Sections 2 and 4, equation 5',
      action='Keep admitted jobs running; select waiting jobs using predicted upper lengths and future memory feasibility.',
      boundary='Targets total completion latency in a non-preemptive round model. Our FIFO peak adapter omits shortest-first subset selection; it is not full MC-SF.'),
 dict(name='WAIT',url='https://arxiv.org/abs/2504.11320v4',locator='v4 abstract; detailed thresholds not reproduced',
      action='Gate admission with fluid-guided thresholds; Nested WAIT regulates decode segments and adds a memory safety buffer.',
      boundary='Known-length and unknown-length variants target stability and latency. Memory headroom is prior art; our measurements concern native offload recovery.'),
 dict(name='Andes',url='https://arxiv.org/abs/2404.16283',locator='Abstract',
      action='Prioritize and preempt at token boundaries using expected user-experience gain and GPU resource demand.',
      boundary='Targets smooth streaming and prompt first output. Our fixed maxgap/stall metrics do not reproduce its reading-aware utility.'),
 dict(name='FastServe',url='https://arxiv.org/abs/2305.05920',locator='Abstract',
      action='Use skip-join multilevel feedback queues initialized from prompt length, with token-level preemption and proactive host/GPU state movement.',
      boundary='Targets job latency through priority and execution quanta. HORIZON retains native quanta and changes admission/recovery/victim decisions.'),
 dict(name='FastSwitch',url='https://arxiv.org/html/2411.18424v1',locator='Sections 3.1--3.3',
      action='Group KV blocks, adapt synchronous/asynchronous swaps, and reuse valid CPU KV copies to reduce switching cost.',
      boundary='Changes layout and transfer execution for fairness-aware serving. Our fixed native transfer path cannot reproduce its system-level benefit.'),
 dict(name='UniBoost',url='https://arxiv.org/html/2606.18431v1',locator='Sections 3.2--3.4, especially 3.3 MemGuard',
      action='Unify prefill/decode priorities from arrival and attained work; quantize revisions and protect a dispatched request until its next service threshold.',
      boundary='MemGuard already provides minimum-run protection. Our step-based lease and peer-growth gate are a different implementation, not proof of novelty or superiority.'),
 dict(name='SuperInfer',url='https://arxiv.org/html/2601.20309v1',locator='Sections 4.2--4.3',
      action='Rotate requests by SLO-derived virtual lag under a transfer budget; overlap full-duplex KV movement with computation.',
      boundary='Designed for GH200 HBM/Grace memory and C2C transfers. Its proactive rotary scheduling and transfer engine are not reproduced by our single-GPU adapter.'),
 dict(name='Llumnix',url='https://www.usenix.org/conference/osdi24/presentation/sun-biao',locator='OSDI 2024 abstract and paper introduction',
      action='Reschedule running requests across model instances using live migration for load balancing, isolation, and priority.',
      boundary='Requires multiple instances and migration. Our experiment has one serving GPU; its remaining loss does not test distributed rescheduling.'),
 dict(name='vLLM',url='https://arxiv.org/abs/2309.06180',locator='PagedAttention paper abstract; runtime independently pinned in FACTS.md',
      action='Allocate KV in pages and share blocks to reduce fragmentation and duplicate storage.',
      boundary='The original paper motivates memory management. Native victim/LOAD/ACK behavior here is established from pinned 0.26 source, not attributed to the 2023 paper.')]

def escape(s):
 for old,new in [('&',r'\&'),('%',r'\%'),('_',r'\_')]:s=s.replace(old,new)
 return s

def main():
 (OUT/'data.json').write_text(json.dumps(dict(checked_date='2026-10-16',rows=ROWS,
  semantics='Primary-source action comparison. Locators state the evidence depth. No unimplemented method is presented as a measured baseline.'),indent=2)+'\n')
 lines=[r'\begingroup\small\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}',
 r'\begin{longtable}{>{\raggedright\arraybackslash}p{.14\textwidth}>{\raggedright\arraybackslash}p{.38\textwidth}>{\raggedright\arraybackslash}p{.39\textwidth}}',
 r'\caption{Prior-work actions and comparison boundaries. Names link to the primary sources; versions and locators are recorded in T3/data.json.}\label{tab:prior}\\',
 r'\toprule Method & State and action & Boundary of this comparison \\ \midrule\endfirsthead',
 r'\toprule Method & State and action & Boundary of this comparison \\ \midrule\endhead']
 for r in ROWS:lines.append(r'\href{'+r['url']+'}{'+escape(r['name'])+'} & '+escape(r['action'])+' & '+escape(r['boundary'])+r' \\')
 lines.extend([r'\bottomrule\end{longtable}',r'\endgroup'])
 (OUT/'table.tex').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(rows=len(ROWS))))
if __name__=='__main__':main()

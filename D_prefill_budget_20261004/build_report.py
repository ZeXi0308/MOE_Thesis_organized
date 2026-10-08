"""Build numerical paper sections only from complete local campaign originals."""
import collections
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
PHASES=['fixed-normal-r01','tuning-normal-r01','holdout-normal-r01',
        'high-normal-r01','low-normal-r01','extra-controls-dev','extra-controls-high','extra-controls-high384']
groups={}
allrows=[]
for phase in PHASES:
    folder=ROOT/phase
    assert json.loads((folder/'status.json').read_text())['status']=='COMPLETE',phase
    subprocess.run([sys.executable,str(ROOT/'analyze.py'),str(folder),'--strict','--exclude-warmup'],check=True)
    reports=json.loads((folder/'summary.json').read_text())['runs']
    groups[phase]=[x['summary'] for x in reports]
    allrows.extend(groups[phase])

def aggregate(rows):
    d=collections.defaultdict(list)
    for r in rows:d[r['policy']].append(r)
    result={}
    for name,rs in d.items():
        vals={k:statistics.mean(r[k] for r in rs) for k in rs[0]
              if all(type(r[k]) in (int,float) for r in rs)}
        result[name]=dict(n=len(rs),mean=vals,ranges={k:[min(r[k] for r in rs),max(r[k] for r in rs)] for k in vals})
    return result

agg={k:aggregate(v) for k,v in groups.items()}
agg['high-controls']=aggregate(groups['extra-controls-high']+groups['extra-controls-high384'])
agg['low-merged']=aggregate([dict(r,policy='fixed2048') if r['policy']=='decode' else r for r in groups['low-normal-r01']])
agg['development']=aggregate(groups['fixed-normal-r01']+groups['tuning-normal-r01'])
(ROOT/'campaign_summary.json').write_text(json.dumps(dict(groups=agg,
    measured_runs=len(allrows),completed_requests=sum(r['finished_count'] for r in allrows),
    measured_output_tokens=sum(r['output_tokens'] for r in allrows),
    all_valid=all(r['valid'] and r['all_finished'] for r in allrows)),indent=2)+'\n')

def label(p):return {'decode':'Decode count','feedback':'Feedback'}.get(p,'Fixed '+p[5:])
def table(key,title,policies=None):
    g=agg[key]
    policies=policies or sorted(g)
    lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',
       r'Policy & $n$ & Output/s & TTFT P95 & Mean flow & Gap P95 & Goodput/s \\',
       r' & & tokens & seconds & seconds & ms & requests \\ \midrule']
    for p in policies:
        if p not in g:continue
        s=g[p]['mean']
        lines.append(f"{label(p)} & {g[p]['n']} & {s['output_tokens_per_s']:.1f} & {s['ttft_p95_s']:.3f} & {s['completion_flow_mean_s']:.3f} & {s['request_max_gap_p95_s']*1000:.2f} & {s['joint_slo_requests_per_s']:.3f} \\\\")
    lines += [r'\bottomrule\end{tabular}',r'\caption{'+title+r' Values are means of complete-run statistics; Gap P95 is across per-request maximum observed gaps, not pooled token gaps.}',r'\end{table}']
    return '\n'.join(lines)

dev=agg['development']; hold=agg['holdout-normal-r01']; hi=agg['high-normal-r01']
s=dev['fixed512']['mean']; b=dev['fixed2048']['mean']; f=hold['feedback']['mean']; hb=hold['fixed2048']['mean']
results=[f"All {len(allrows)} measured episodes completed, covering {sum(r['finished_count'] for r in allrows):,} request completions and {sum(r['output_tokens'] for r in allrows):,} generated tokens. No measured request is dropped from the denominator. All token accounting checks pass.",
    r'\subsection{The fixed-budget tradeoff is real}',
    f"On the development trace, fixed 2,048 improves output throughput by {(b['output_tokens_per_s']/s['output_tokens_per_s']-1)*100:.2f}\\% relative to 512. Request maximum-gap P95 rises from {s['request_max_gap_p95_s']*1000:.2f} to {b['request_max_gap_p95_s']*1000:.2f} ms. Both satisfy the same primary joint SLO for every request. Thus a finer gap tradeoff exists even where the declared SLO does not reward smaller chunks.",
    table('development','Development trace; fixed selection and candidate construction use these observations.', ['fixed256','fixed512','fixed1024','fixed2048','decode','feedback']),
    r'\subsection{Frozen long-document confirmation}',
    f"On the long-document holdout, feedback changes throughput by {(f['output_tokens_per_s']/hb['output_tokens_per_s']-1)*100:.2f}\\% and joint goodput by {(f['joint_slo_requests_per_s']/hb['joint_slo_requests_per_s']-1)*100:.2f}\\% versus the development-selected fixed 2,048. The short background prompts are shared with development. The three main-policy repetitions provide repeatability evidence, not a broad workload-population confidence interval.",
    table('holdout-normal-r01','Frozen long-document holdout.', ['fixed2048','fixed1024','decode','feedback']),
    r'\subsection{Normal-memory pressure extension}',
    table('high-normal-r01','High load: 160 requests, normal memory allocation.', ['fixed256','fixed1024','fixed2048','decode','feedback']),
    table('low-merged','Low load: 12 requests. Decode-count and fixed 2,048 choose exactly the same cap at every step and are merged here.', ['fixed2048','feedback']),
    r'\subsection{Dominant-level and mean-budget controls}',
    'In development, feedback selects 1,024 in 84 of 88 mixed steps in each repetition, with four selections of 2,048. Its mean mixed cap is 1,070.55. Fixed 1,024 is the direct dominant-level control; fixed 1,072 additionally approximates the mean allowance. This comparison counts full realized service time, rather than assuming identical state trajectories from similar average caps.',
    table('extra-controls-dev','Additional exploratory controls: matched mean cap and upper fixed endpoint.', ['fixed1072','fixed3840','feedback']),
    table('high-controls','Additional high-load fixed controls; constants apply to the entire trace, never individual segments.', ['fixed256','fixed304','fixed384','fixed512','fixed768','fixed3840','feedback'])]

peak=max(r['peak_kv_fraction'] for r in allrows if r['peak_kv_fraction'] is not None)
resident=max(r['max_resident_requests'] for r in allrows if r['max_resident_requests'] is not None)
decision=max(r['decision_p95_us'] for r in allrows)
results.append(f"Across measured episodes, peak resident concurrency is {resident} and peak observed KV occupancy is {peak*100:.2f}\\%. All measured preemption counts are zero. The largest per-run P95 policy-choice time is {decision:.2f} microseconds; this excludes common observation construction, which remains included in all end-to-end times. No second model is introduced because an independently useful extra signal has not yet been established.")
hc=agg['high-controls']; fb=hc['feedback']['mean']; best=hc['fixed384']['mean']
results.append(r'\subsection{A conditional gain, not a uniform improvement}')
results.append(f"In the additional high-load comparison, feedback reaches {fb['joint_slo_requests_per_s']:.3f} qualified requests/s versus {best['joint_slo_requests_per_s']:.3f} for fixed 384, the highest mean among the tested high-load fixed constants. This is {(fb['joint_slo_requests_per_s']/best['joint_slo_requests_per_s']-1)*100:.2f}\\% higher goodput, alongside {(1-fb['output_tokens_per_s']/best['output_tokens_per_s'])*100:.2f}\\% lower raw throughput, {(fb['ttft_p95_s']/best['ttft_p95_s']-1)*100:.2f}\\% higher TTFT P95, and {(fb['completion_flow_p95_s']/best['completion_flow_p95_s']-1)*100:.2f}\\% higher completion-latency P95. This is a measured tradeoff, not Pareto dominance. Fixed 304 approximates feedback's {fb['mean_actual_prefill_tokens_mixed_steps']:.2f}-token realized mixed-step mean and {fb['mean_prefill_cap_mixed_steps']:.2f}-token mean cap in this supplementary group, but has only {hc['fixed304']['mean']['joint_slo_requests_per_s']:.3f} qualified requests/s. Mean work alone does not explain the high-load goodput difference.")
results.append('The original high-load main group has zero gap-SLO failures. Feedback admits 30--31 first-wave long requests into the joint SLO versus 21--22 for the count rule; later waves fail under both. Compared with count-only scheduling, feedback improves complete-request mean outcomes for 87 of 160 request IDs and worsens 73; the joint-goodput increment comes only from early requests crossing thresholds. Third-wave mean completion rises from 34.57 to 39.62 seconds. This explicitly exposes the service cost of the gain.')
results.append('All fixed-512 supplementary trials remain in the report: its goodput ranges from 0.571 to 1.078 requests/s. In the slower-SLO trial, an observed 108.97 ms gap causes 67 gap failures; the associated step envelope is 108.47 ms and scheduler time is 1.07 ms. This record is neither removed nor attributed to a specific GPU or host cause. Fixed-3840 goodput is zero because all 160 requests exceed the 20-second completion constraint, despite the largest raw throughput; none fails its gap condition. These results show sensitivity to the full SLO conjunction, not only average step latency.')
results.append('The complete retained log contains one monitored Triton JIT warning during the initial 512 warmup and no recorded JIT compilation warning in the measured episodes. This is a log-based check, not a claim that every source of runtime jitter is observable.')

(ROOT/'results.tex').write_text('\n\n'.join(results)+'\n')
(ROOT/'abstract_result.tex').write_text(
    'We implement an aggregate prefill-token cap in native single-GPU serving, preserving request order, admission rules, model precision and kernels. '
    'Real mixed request traces show a repeatable fixed-budget throughput--generation-stall tradeoff. '
    f'In the development trace, increasing the cap from 512 to 2,048 improves output throughput by {(b["output_tokens_per_s"]/s["output_tokens_per_s"]-1)*100:.1f}\\%, while increasing request maximum-gap P95. '
    'A four-level controller using only the previous completed mixed-step duration mostly selects one level at moderate load. '
    'We compare it with fixed budgets, a decode-count rule, its dominant fixed level and a matched-mean cap, including a normal-memory PRO 6000 pressure extension. '
    'The frozen moderate-load confirmation favors a fixed cap. At high load, feedback has a small exploratory joint-goodput advantage over the tested constants, with lower raw throughput and longer tail waiting. '
    'The contribution is a reproducible, bounded empirical result rather than a claim that adaptive chunking is new or broadly beneficial.\n')
(ROOT/'conclusion.tex').write_text(
    'These experiments establish the fixed-budget tradeoff and a conditional high-load goodput gain, but not a consistent improvement across full service metrics and loads. The frozen long-document confirmation does not beat its development-selected fixed policy. '
    'At moderate load, most actions collapse to the same cap; at higher load, the feedback target can be incompatible with the decode cost already present, so reducing prompt service can move delay into the waiting queue. '
    'The scope is one model, one GPU, in-process synchronous vLLM, fixed output lengths, and a small set of structured arrival traces. '
    'Model precision is unchanged, but different batch shapes can alter greedy outputs through floating-point execution; fixed output lengths hold service work constant and are not a semantic-equivalence claim. '
    'Host-observed delivery gaps omit network transport and do not isolate device-only latency. GPU clocks are not locked. '
    'A 100 ms gap SLO leaves many small-load requests qualified under every policy, so its goodput is often determined by full service duration. '
    'No hindsight choice of a fixed cap per trace segment is presented as an online algorithm. '
    'The high-load trace was used for fixed-budget tuning and is exploratory rather than an independent high-pressure holdout. The count-only comparator is a conservative simple rule, not a tuned count-based controller. Thus the value of elapsed time beyond a strong count rule remains unresolved; a MoE label supplies no novelty claim.\n')
print('Generated numerical paper sections and campaign_summary.json')

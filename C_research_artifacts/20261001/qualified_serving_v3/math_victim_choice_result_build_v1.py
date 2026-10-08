#!/usr/bin/env python3
"""Aggregate the completed full-running victim action experiment."""
from pathlib import Path
import json,hashlib,time,statistics
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_text())
def sha(n):return hashlib.sha256((P/n).read_bytes()).hexdigest()
arms={};mechanisms={};pairs={}
for mode in ('fixed','bidkv','maxfree'):
 n='math_victim_choice_'+mode+'_analysis_v1.json';a=read(n)
 assert a['integrity']=='VALID' and a['counts']['planned']==a['counts']['completed']==1024
 arms[mode]=dict(file=n,sha256=sha(n),**{k:a[k] for k in ('integrity','issues','counts','timing','episode','pressure_counts')},work=a['attribution']['totals'])
 if mode!='fixed':
  n='math_victim_choice_'+mode+'_evidence_v1.json';e=read(n);assert e['integrity']=='VALID' and not e['issues']
  p=json.loads((Path(a['run'])/'victim-choice-policy.json').read_text());d=p['decisions']
  delta=[x['selected_releasable_blocks']-x['native_tail_candidate']['releasable_blocks'] for x in d]
  nums=[x['candidate_count'] for x in d]
  mechanisms[mode]=dict(file=n,sha256=sha(n),counts=e['counts'],step_offset=e['policy_step_minus_observer_call_id'],
   policy_hook_wall_s=p['policy_hook_wall_s'],cost_scope=p['cost_scope'],
   candidate_count=dict(min=min(nums),median=statistics.median(nums),max=max(nums)),
   non_single_decode_candidate_events=sum(x['candidate_non_single_decode_count']>0 for x in d),
   excluded_prefix_events=sum(x['excluded_unscheduled_prefix']>0 for x in d),
   same_state_selected_minus_native_tail_release=dict(total=sum(delta),min=min(delta),max=max(delta),all_positive=all(x>0 for x in delta),
    scope='Per-decision immediate release only; not whole-policy saved blocks or counterfactual service benefit.'))
for suffix in ('fixed_within','fixed_vsbidkv','fixed_vsmaxfree','bidkv_vsmaxfree'):
 n='math_victim_choice_'+suffix+'_pair_v1.json';c=read(n)
 assert c['comparison_integrity'] in ('MATCHED','MATCHED_WITH_POLICY_CHANGE') and not c['issues']
 pairs[suffix]=dict(file=n,sha256=sha(n),**{k:c[k] for k in ('comparison_integrity','episode_changes','count_changes','timing_changes','per_request_directions','identical_output_subset_directions','output_consistency','transitions')})
qualification={}
candidate=arms['maxfree']
for control in ('fixed','bidkv'):
 base=arms[control]
 ratio=dict(mean_flow=candidate['timing']['completion_s']['mean']/base['timing']['completion_s']['mean'],
  slo_goodput=candidate['episode']['slo_goodput_requests_per_s']/base['episode']['slo_goodput_requests_per_s'],
  output_token_rate=candidate['episode']['output_tokens_per_s']/base['episode']['output_tokens_per_s'],
  worst_gap=candidate['timing']['max_host_gap_s']['max']/base['timing']['max_host_gap_s']['max'])
 guards=ratio['mean_flow']<=1.05 and ratio['output_token_rate']>=.97
 qualification[control]=dict(ratios=ratio,efficiency_guardrails_pass=guards,
  flow_or_goodput_signal=guards and (ratio['mean_flow']<=.97 or ratio['slo_goodput']>=1.03),
  worst_gap_tradeoff_signal=guards and ratio['worst_gap']<=.8,
  correct_count_delta=candidate['counts']['correct']-base['counts']['correct'],
  natural_eos_count_delta=candidate['counts']['natural_eos']-base['counts']['natural_eos'])
signal=all(x['flow_or_goodput_signal'] for x in qualification.values()) or all(x['worst_gap_tradeoff_signal'] for x in qualification.values())
assert not signal
h=mechanisms['maxfree']['policy_hook_wall_s']
discount=dict(assumption='Subtract all measured maxfree hook wall from every completion and from episode duration, while holding its executed workload and outcomes fixed. Deliberately optimistic bookkeeping sensitivity, not a rerun, not an oracle and not a system guarantee.',
 hook_wall_removed_s=h,optimistically_discounted_mean_flow_s=candidate['timing']['completion_s']['mean']-h,
 optimistically_discounted_duration_s=candidate['episode']['duration_s']-h,
 optimistically_discounted_worst_gap_s=candidate['timing']['max_host_gap_s']['max']-h)
discount['mean_flow_ratio_to_fixed']=discount['optimistically_discounted_mean_flow_s']/arms['fixed']['timing']['completion_s']['mean']
discount['slo_goodput_ratio_to_fixed']=(candidate['counts']['slo_pass']/discount['optimistically_discounted_duration_s'])/arms['fixed']['episode']['slo_goodput_requests_per_s']
snap=read('math_victim_choice_terminal_snapshot_v1.json')
assert snap['parent'] is None and len(snap['arms'])==3 and all(x['launcher-receipt.json']['status']==x['native/status.json']['status']=='COMPLETE' for x in snap['arms'].values())
result=dict(schema='c-math-victim-choice-result-v1',created_unix_s=time.time(),goal_status='ACTIVE_NOT_ACHIEVED',paper_status='NO_PAPER_GO',previous_turn_classification='PROGRESS',
 protocol='math_victim_choice_protocol_v1.json',protocol_sha256=sha('math_victim_choice_protocol_v1.json'),evidence_type='NATIVE_SERVING_INPROCESS_FULL_REQUEST_DEVELOPMENT',
 scope='Same1024 GSM8K development requests, all arrivals0, model/runtime/resources/fixed48 held fixed. Three independent natural generation episodes.',
 new_valid_episodes=3,total_valid_math1024_episodes=26,arms=arms,mechanisms=mechanisms,comparisons=pairs,
 predeclared_pilot_qualification=qualification,pilot_signal_qualifies=signal,optimistic_fixed_trajectory_hook_discount=discount,
 decision=dict(status='SCOPED_NO_GO_FOR_MAXIMUM_PHYSICAL_RELEASE_VICTIM_RULE',
 exact_formulation='On native RUNNING allocation None, maximize immediately releasable distinct physical blocks across scheduled prefix/current/suffix, native tail tie; native rollback/self-break; APC ON and fixed48 restoration.',
 strongest_baseline='Fixed48 with native FCFS tail gives lowest measured mean completion; default BidKV-full score is an additional nearest-method component control.',
 action_real='All62 BidKV and63 maximum-free choices changed native tail, executed native preemption and matched predicted physical release.41/50 chosen victims were already scheduled; native rollback was verified.',
 finding='Fewer preemptions need not mean less recomputation or faster service:78→62/63 events while recompute12122→13207/13466 and mean completion29.131→29.509/30.341s.',
 nuance='Both scoring arms improve the historical SLO pass count963→976 and p95 host gap, but worst gap worsens and mean completion rises. BidKV diagnostic SLO-goodput+0.665% is below pilot threshold and within the magnitude of same-policy duration drift; no dominance/significance claim.',
 failure_category='Valid action, local release proxy failed to yield qualifying whole-request benefit; additional exact-physical scoring tax for maximum-free.',
 stop='Do not tune weights/ties or optimize ownership counters to rescue this maximum-free pilot. Do not generalize to all victim policies or all arrival regimes.',
 what_not_measured='Independent arrivals, held-out inputs, repeated policy effects, Ascend BidKV full system, second model, future-information oracle and business SLO.',
 reopen='A new workload/arrival or cache-ownership regime with demonstrated full-request residual beyond tuned fixed48 and nearest published controls; source repair if this experiment is later shown invalid.',
 next_smallest_experiment='Stop adding victim scores. Use one predeclared two-burst arrival trace on the same usable math task and directly compare native restoration, fixed48 and existing dynamic-runway restoration. This tests the observed useful admission/restore action under time-varying demand; no rescue by threshold or seed search.'),
 runtime=dict(status='ALL_THREE_COMPLETE_PARENT_EXITED',launcher_pid=78334,start_ticks='1207346496',active_job=None,
 terminal_snapshot='math_victim_choice_terminal_snapshot_v1.json',archive='math_victim_choice_raw_v1_20261002.tar.xz',
 archive_sha256=sha('math_victim_choice_raw_v1_20261002.tar.xz'),archive_bytes=(P/'math_victim_choice_raw_v1_20261002.tar.xz').stat().st_size,
 verified_restored_large_logs=9),
 limitations=['Single development episode per new policy; no statistical or held-out claim.',
  'Different natural outputs in247/254requests versus fixed control; identical-output subsets are descriptive rather than causal.',
  'BidKV score component adapted to the local vLLM backend with APC and fixed48; not a full paper/system reproduction.',
  'Unique prefix-free prompts and selected single-token decode qualify this APC rollback probe; chunked-prefill victim behavior untested.',
  'Historical TTFT20s/max-host-gap4s thresholds are diagnostics; p95 gap improvement is reported but not substituted for the predeclared worst-gap qualification.',
  'Measured full-request effects include policy execution, observer and actual generated work; no sum of local release counts is claimed as speedup.'])
with (P/'math_victim_choice_result_v1.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(status='COMPLETE',decision=result['decision']['status'],qualification=qualification,discount=discount)))

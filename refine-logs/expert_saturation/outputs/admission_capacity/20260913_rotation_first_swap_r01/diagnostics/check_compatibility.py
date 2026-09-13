#!/usr/bin/env python3
"""Compare retained V diagnostics and exercise CPU-only readiness/order gates."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import diagnose_paths as d

here = Path(__file__).resolve().parent
v = here.parents[1]/'20260913_rotation_victim_order_r01'
result = d.read(here/'v_compatibility/diagnostics.json')
old_paths = d.read(v/'analysis/paths.json')
old_tail = d.read(v/'analysis/tail_paths.json')
checks = []
assert result['experiment_id']=='20260913_rotation_victim_order_r01' and len(result['cells'])==8
for label, cell in result['cells'].items():
    old=old_paths['cells'][label]
    fields=('step','victim','resume','victim_output_tokens','victim_computed_tokens')
    assert [tuple(r[k] for k in fields) for r in cell['forced_actions']]==[tuple(r[k] for k in fields) for r in old['rotations']]
    paths=cell['recovery_accounting']['recoveries']
    assert [(r['preempt_step'],r['request_id'],r['first_schedule_step'],r['first_new_token_step'],r['gap_s'],r['recomputed_positions']) for r in paths]==[(r['preempt_step'],r['request_id'],r['resume_step'],r['first_new_output_step'],r['pause_s'],r['recomputed_tokens_to_first_new_output']) for r in old['recovery_paths']]
checks.append('all_eight_V_forced_choices_recovery_alignment_pause_and_recompute_equal')
for prior in old_tail:
    for old_role,role in [('A','least_progress'),('B','most_output')]:
        label=f"{prior['cohort']}-block{prior['block']}-{role}";cell=result['cells'][label]
        widths=prior['widths'][old_role];two=cell['pure_decode_widths']['2']
        assert {k:r['calls'] for k,r in cell['pure_decode_widths'].items()}==widths['pure_decode_width_counts']
        assert two['calls']==widths['pure_decode_width2_calls']
        assert two['engine_duration_s']==widths['width2_engine_duration_s']
        assert two['first_output_counts']==widths['width2_first_output_counts']
        assert two['request_sets']==widths['width2_request_sets']
        for row in prior['requests']:assert cell['requests'][row['request_id']]==row[old_role]
checks.append('all_V_width_counts_width2_times_progress_request_sets_and_completion_equal')
for pair in result['comparisons']:
    assert pair['width_changes']['2']['calls_change']==-79
    assert pair['first_choice_difference']['index']==0 and pair['first_action_prestate_equal']
checks.append('four_V_pairs_first_choice_836_and_width2_delta_minus79')
assert d.read(here/'prepared_gate/diagnostics.json')['comparisons']==[]
checks.append('F_missing_qualified_primary_has_no_comparisons')
primary=d.read(v/'analysis/analysis.json');partial=copy.deepcopy(primary);partial['cells'].pop()
original=d.read
with patch.object(d,'read',side_effect=lambda p: partial if p==v/'analysis/analysis.json' else original(p)):
    qualified,blocked=d.gate(v,v/'analysis/analysis.json');assert qualified is None and blocked['comparisons']==[]
checks.append('partial_primary_rejected_before_numerical_diagnostics')
# Minimal synthetic CPU state checks effective modes; this is not a C GPU run.
def state(output):return dict(prompt_tokens=3072,output_tokens=output,computed_tokens=3071+output,block_counts=[205])
lo,hi,target='internal-low','internal-high','internal-target'
states={lo:state(100),hi:state(200),target:state(50)}
def decision(step,mode,victim):
    return dict(step=step,effective_victim_order=mode,proposal=dict(action='rotate',victim_id=victim,resume_id=target),
        forced_preempted=[victim],preempted=[victim],resumed=[])
raw=dict(internal_to_source={lo:'source-low',hi:'source-high',target:'source-target'},memory_trace=[
    dict(attempted_step=step,before=dict(requests=states,running_ids=[lo,hi],pool=dict(free_blocks=0))) for step in (0,40)])
trace=[decision(0,'most_output',hi),decision(40,'least_progress',lo)]
assert [r['victim'] for r in d.actions(raw,trace,'first_most_then_least')]==['source-high','source-low']
checks.append('CPU_only_C_candidates_use_recorded_most_then_least_modes')
report=dict(status='PASS_CPU_COMPATIBILITY_ONLY',checks=checks,new_F_GPU_evidence=False,
    compatibility_source_experiment=result['experiment_id'],diagnostic_source_sha256=d.digest(here/'diagnose_paths.py'),
    legacy_inputs={str(p):d.digest(p) for p in (v/'analysis/paths.json',v/'analysis/tail_paths.json',v/'analysis/analysis.json')},
    scope='One complete compatibility pass over retained V A/B. F has no qualified GPU result. C ordering check is a tiny synthetic CPU fixture, not replayed or fabricated C GPU evidence.')
with (here/'compatibility_checks.json').open('x') as out:json.dump(report,out,indent=2);out.write('\n')
print(json.dumps(report,indent=2))

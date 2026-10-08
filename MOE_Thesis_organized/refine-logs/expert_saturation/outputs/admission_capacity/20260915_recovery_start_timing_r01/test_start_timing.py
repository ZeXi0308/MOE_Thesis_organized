"""CPU selector/prepare behavior only; no native adapter execution or GPU."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument('--pkg', type=Path, default=Path(
    '/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/'
    'admission_capacity/20260915_repeated_kv_service_r01/pkg'))
pkg = parser.parse_args().pkg.resolve()
sys.path.insert(0, str(pkg))
from absence_rotation import AbsenceRotation, RequestView, RotationConfig
from staged_save_contract import RequestState, prepare

default = RotationConfig()
early = replace(default, min_absence_steps=0)
assert default.min_absence_steps == 30
assert {k for k, v in asdict(default).items() if v != asdict(early)[k]} == {'min_absence_steps'}
assert (early.min_steps_between_swaps, early.min_residency_steps,
        early.protect_progress_fraction) == (20, 30, 0.9)
step = 100
victim = RequestState('victim', 95, 64, 32, 128, 'RUNNING', tuple(range(6)))
target = RequestState('target', 0, 80, 16, 128, 'PREEMPTED', ())


def decide(cfg, *, current=victim, waiting=target, last_swap=80, resident=70):
    tracker = AbsenceRotation(config=cfg, victim_order='most_output',
        absent_since={'target': step}, absence_count={'target': 1},
        resident_since={'victim': resident}, last_swap_step=last_swap)
    row = RequestView(current.request_id, current.computed, current.prompt,
                      current.prompt + current.max_output, current.output)
    decision = tracker.decide(step, [row], [waiting.request_id], 1,
                             {waiting.request_id: waiting.remaining_blocks})
    assert len(tracker.decisions) == 1
    return decision


baseline, immediate = decide(default), decide(early)
assert baseline.action == 'noop' and baseline.reason == 'absence 0 below threshold'
assert immediate.action == 'rotate' and immediate.absence_steps == 0
plan = prepare(step, victim, target, 1)
assert plan.saved_tokens == 80 and len(plan.source_blocks) == 5

cooldown = decide(early, last_swap=81)
assert cooldown.action == 'noop' and cooldown.reason == 'swap cooldown'
residency = decide(early, resident=71)
assert residency.action == 'noop' and residency.reason == 'no eligible victim'
near_done = replace(victim, computed=180, output=117, blocks=tuple(range(12)))
progress = decide(early, current=near_done)
assert progress.action == 'noop' and progress.reason == 'no eligible victim'

unfunded = replace(target, prompt=112)
funding = decide(early, waiting=unfunded)
assert funding.action == 'rotate'  # Selector proposes; adapter prepare still qualifies.
try:
    prepare(step, victim, unfunded, 1)
except ValueError as error:
    funding_error = str(error)
    assert funding_error == 'victim cannot fund target full history'
else:
    raise AssertionError('Zero absence threshold bypassed prepare funding qualification')

print(json.dumps(dict(status='PASS', package=str(pkg), save='on in both planned arms',
    scope='Actual CPU selector and prepare; save execution/native adapter wiring/GPU remain untested.',
    config_default=asdict(default), config_explicit_zero=asdict(early), step=step,
    same_state_decisions=dict(default30=asdict(baseline), explicit0=asdict(immediate)),
    early_plan_saved_tokens=plan.saved_tokens,
    preserved_guards=dict(cooldown_19=cooldown.reason, residency_29=residency.reason,
                         progress_ge_0_9=progress.reason, insufficient_funding=funding_error)),
    indent=2))

"""One bounded cap-order pulse with the parent's unchanged normal-arrival loop."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base

DESIGN = json.loads((ROOT / 'design.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PulseController:
    """Observe only current scheduler state; change only the pure-prefill cap."""
    def __init__(self, name, engine):
        assert name in ('pulse_hl', 'pulse_lh', 'fixed1024', 'fixed2048'), name
        self.name, self.engine = name, engine
        self.low, self.high = DESIGN['legal_caps']
        self.common, self.half = DESIGN['common_cap'], DESIGN['half_steps']
        self.trigger_step = self.trigger_time = self.trigger_snapshot = None
        self.pending = self.native_result = self.awaiting_forward = None
        self.forward_pulse_steps = self.forward_prefill = self.forward_decode = self.forward_up_steps = 0
        self.decision_count = self.attached_steps = self.requested_ups = 0
        self.pulse_steps = self.pulse_prefill = self.pulse_decode = self.actual_up_steps = 0
        self.underfilled = []

    def external(self, rid):
        return self.engine.output_processor.request_states[rid].external_req_id

    def snapshot(self, scheduler, rows, decision_time):
        records = []
        slo = DESIGN['primary_slo']
        for rid, request in scheduler.requests.items():
            external = self.external(rid)
            row = rows[external]
            times = row['token_times_s']
            age = decision_time - row['arrival_s']
            first, last = (times[0], times[-1]) if times else (None, None)
            closed_gap = max((b-a for a, b in zip(times, times[1:])), default=0.)
            open_gap = decision_time-last if last is not None else None
            records.append(dict(request_id=external,
                computed_tokens=request.num_computed_tokens,
                prompt_tokens=request.num_prompt_tokens,
                observed_output_tokens=len(row['output_token_ids']),
                arrival_s=row['arrival_s'], add_s=row['add_s'],
                first_output_s=first, last_output_s=last,
                max_closed_gap_s=closed_gap, open_gap_s=open_gap, age_s=age,
                ttft_failed_observed=(age if first is None else first-row['arrival_s']) > slo['ttft_s'],
                gap_failed_observed=closed_gap > slo['gap_s'] or (open_gap is not None and open_gap > slo['gap_s']),
                completion_failed_observed=age > slo['completion_s']))
        return dict(decision_time_s=decision_time,
            running_order=[self.external(r.request_id) for r in scheduler.running],
            waiting_order=[self.external(r.request_id) for r in scheduler.waiting],
            requests=records)

    def choose(self, decode_count, scheduler, rows, step_index, now):
        decision_time = now()
        backlog = sum(max(0, r.num_prompt_tokens-r.num_computed_tokens)
                      for r in scheduler.requests.values())
        context = sum(r.num_computed_tokens for r in scheduler.running
                      if r.num_computed_tokens >= r.num_prompt_tokens)
        guard = (decode_count >= DESIGN['trigger']['decode_count_at_least']
                 and backlog >= DESIGN['trigger']['prefill_backlog_at_least'])
        formal = self.name.startswith('pulse_')
        first_hit = formal and self.trigger_step is None and guard
        capture_us = 0.
        if first_hit:
            self.trigger_step, self.trigger_time = step_index, decision_time
            tick = time.perf_counter()
            self.trigger_snapshot = self.snapshot(scheduler, rows, decision_time)
            capture_us = (time.perf_counter()-tick)*1e6
        offset = None if self.trigger_step is None else step_index-self.trigger_step
        in_pulse = offset is not None and 0 <= offset < 2*self.half
        recommendations = dict(pulse_hl=self.common, pulse_lh=self.common)
        if in_pulse:
            recommendations = dict(pulse_hl=self.high if offset < self.half else self.low,
                                   pulse_lh=self.low if offset < self.half else self.high)
        if formal:
            cap = recommendations[self.name]
            phase = 'pulse' if in_pulse else ('prefix' if self.trigger_step is None else 'suffix')
            reason = 'bounded_pulse' if in_pulse else ('guard_not_yet_met' if self.trigger_step is None else 'pulse_exhausted_no_retrigger')
        else:
            cap, phase, reason = int(self.name.removeprefix('fixed')), 'warm_fixed', 'fixed_warmup'
        self.pending = dict(step_index=step_index, decision_time_s=decision_time,
            decode_count=decode_count, decode_context_sum=context,
            prefill_backlog_tokens=backlog, guard=guard, guard_first_hit=first_hit,
            pulse_offset=offset if in_pulse else None, phase=phase,
            legal_caps=list(DESIGN['legal_caps']), baseline_cap=self.common,
            recommendations=recommendations, requested_cap=cap,
            decision_reason=reason, trigger_snapshot_us=capture_us,
            execution_status='REQUESTED_NOT_RETURNED',
            native_schedule_completed=False, forward_completed=False)
        self.native_result = None
        self.decision_count += 1
        if in_pulse and cap > self.common:
            self.requested_ups += 1
        return cap

    def native_failed(self, exc):
        self.pending.update(execution_status='NATIVE_RAISED', native_error=repr(exc))

    def native_returned(self, out, before):
        self.pending.update(execution_status='NATIVE_RETURNED_NOT_ATTACHED',
                            native_schedule_completed=True)
        self.native_result = (out, before)

    def attach(self, step, rows):
        record = self.pending
        pt, dt, cap = step['prefill_tokens'], step['decode_tokens'], record['requested_cap']
        reason = None
        if pt < cap:
            reason = ('backlog_below_cap' if record['prefill_backlog_tokens'] < cap
                      else 'native_underfill_unresolved')
        record.update(execution_status='NATIVE_SCHEDULED_FORWARD_UNCONFIRMED',
                      native_schedule_completed=True, forward_completed=False,
                      actual_prefill_tokens=pt,
                      actual_decode_tokens=dt, unused_budget_tokens=cap-pt,
                      unused_budget_reason=reason)
        step['pulse_decision'] = record
        self.awaiting_forward = step
        step['completed_prefill_ids'] = [r['request_id'] for r in step['requests']
            if r['prefill_tokens'] > 0 and r['computed_start'] < rows[r['request_id']]['prompt_tokens']
            <= r['computed_start']+r['prefill_tokens']]
        self.attached_steps += 1
        if record['phase'] == 'pulse':
            self.pulse_steps += 1
            self.pulse_prefill += pt
            self.pulse_decode += dt
            self.actual_up_steps += int(pt > self.common)
            if pt < cap:
                self.underfilled.append(dict(step_index=record['step_index'], pulse_offset=record['pulse_offset'],
                                             requested_cap=cap, actual_prefill_tokens=pt, reason=reason))
        self.pending = self.native_result = None

    def observe(self, elapsed, prefill, decode):
        if self.awaiting_forward is not None:
            record = self.awaiting_forward['pulse_decision']
            record.update(execution_status='FORWARD_COMPLETED', forward_completed=True,
                          forward_elapsed_s=elapsed)
            if record['phase'] == 'pulse':
                self.forward_pulse_steps += 1
                self.forward_prefill += prefill
                self.forward_decode += decode
                self.forward_up_steps += int(prefill > self.common)
            self.awaiting_forward = None

    def summary(self):
        pending = None if self.pending is None else dict(self.pending)
        if pending is not None and self.native_result is not None:
            out, before = self.native_result
            pending['native_scheduled_tokens'] = [dict(request_id=before[rid][2], tokens=n,
                computed_tokens_before=before[rid][0], prompt_tokens=before[rid][1])
                for rid, n in out.num_scheduled_tokens.items()]
            pending['native_preempted_ids'] = list(out.preempted_req_ids or [])
        pending_forward = None
        if self.awaiting_forward is not None:
            pending_forward = dict(self.awaiting_forward['pulse_decision'],
                                   unconfirmed_reason='engine_step_success_not_observed')
        status = ('WARM_FIXED' if not self.name.startswith('pulse_') else
                  'NO_TRIGGER' if self.trigger_step is None else
                  'COMPLETE' if self.forward_pulse_steps == 2*self.half else 'INCOMPLETE')
        return dict(policy=self.name, status=status, trigger_step=self.trigger_step,
            trigger_time_s=self.trigger_time, trigger_snapshot=self.trigger_snapshot,
            decision_count=self.decision_count, attached_steps=self.attached_steps,
            requested_ups=self.requested_ups, scheduled_pulse_steps=self.pulse_steps,
            scheduled_prefill_tokens=self.pulse_prefill, scheduled_decode_tokens=self.pulse_decode,
            executed_pulse_steps=self.forward_pulse_steps,
            executed_prefill_tokens=self.forward_prefill, executed_decode_tokens=self.forward_decode,
            actual_up_steps=self.actual_up_steps, actual_up_steps_scope='native_scheduled_prefill_above_common_cap',
            completed_forward_up_steps=self.forward_up_steps,
            underfilled_pulse_steps=self.underfilled, pending_decision=pending,
            pending_forward_decision=pending_forward, frozen_design_sha256=sha(ROOT/'design.json'),
            source_sha256={name: sha(ROOT/name) for name in DESIGN['source_sha256']})


def derived_episode():
    """Exact-once instrumentation edits; the entire arrival/output loop stays literal."""
    source = inspect.getsource(base.episode)
    replacements = [
        ('policy = Policy(policy_name, target_ms)', 'policy = PulseController(policy_name, engine)'),
        ('budget = policy.choose(len(decoding))',
         'budget = policy.choose(len(decoding), scheduler, rows, len(steps), now)'),
        ('        out = native(*args, **kwargs)\n',
         '        try:\n            out = native(*args, **kwargs)\n'
         '        except BaseException as exc:\n            policy.native_failed(exc)\n            raise\n'
         '        policy.native_returned(out, before)\n'),
        ('        return out\n', '        policy.attach(steps[-1], rows)\n        return out\n'),
        ("        dump(path/'raw.json',data)\n",
         "        data['pulse'] = policy.summary()\n        dump(path/'raw.json',data)\n")]
    for old, new in replacements:
        assert source.count(old) == 1, ('Parent episode changed; refusing ambiguous replacement', old)
        source = source.replace(old, new, 1)
    namespace = dict(base.__dict__, PulseController=PulseController)
    exec(compile(source, str(ROOT/'run_timing.py')+'::derived_episode', 'exec'), namespace)
    return namespace['episode'], source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--wait-lock', type=float, default=0)
    args = p.parse_args()
    assert DESIGN['status_at_freeze'] == 'PRE_GPU_FROZEN', 'Freeze design and source hashes before GPU use'
    assert DESIGN['legal_caps'] == [1024, 2048] and DESIGN['common_cap'] == 1024
    assert DESIGN['half_steps'] == 16 and DESIGN['action_steps'] == 32
    assert DESIGN['timeout_s'] == 180 and DESIGN['primary_slo'] == dict(ttft_s=4, gap_s=.1, completion_s=20)
    assert DESIGN['policies'] == ['pulse_hl', 'pulse_lh', 'pulse_lh', 'pulse_hl']
    assert DESIGN['warm_policies'] == ['fixed1024', 'fixed2048']
    for name, expected in DESIGN['source_sha256'].items():
        assert sha(ROOT/name) == expected, ('Frozen source mismatch', name)
    assert {'run_timing.py', 'launch.sh'} <= set(DESIGN['source_sha256']), 'Missing probe source hashes'
    workload = (ROOT/DESIGN['workload']).resolve()
    assert sha(workload) == DESIGN['workload_sha256'], 'Frozen workload mismatch'
    base.episode, source = derived_episode()
    original_dump = base.dump
    def dump_with_probe(path, value):
        if path.name == 'protocol.json':
            value = dict(value, experiment_kind='NORMAL_ARRIVAL_BOUNDED_CAP_ORDER_PROBE',
                         frozen_timing_design=DESIGN, frozen_timing_design_sha256=sha(ROOT/'design.json'))
            (path.parent/'derived_episode.py').write_text(source)
        original_dump(path, value)
    base.dump = dump_with_probe
    sys.argv = [str(ROOT.parent/'run_confirmation.py'), '--output', str(args.output.resolve()),
                '--workload', str(workload), '--policies', ','.join(DESIGN['policies']),
                '--warm-policies', ','.join(DESIGN['warm_policies']), '--wait-lock', str(args.wait_lock)]
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())

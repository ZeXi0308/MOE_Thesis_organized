"""Niyama artifact chunk component adapted to native vLLM, NOT full Niyama.

Only the aggregate pure-prefill cap changes. Native order, admission, recovery,
KV allocation and all decodes remain intact. Declared fixed output counts are
input metadata here, not observations of future natural EOS. Source:
microsoft/sarathi-serve@72339db5b120614033b317880ad3a05a5e4c9415.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base

TOTAL_TIERS = [128] + list(range(256, 2049, 256)) + [2552]
LOW_MEMORY_TIERS = list(range(128, 513, 128))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_by_slack(slack, decode, decode_context, prefill_context, tiers, predict):
    """Same source tiers/margin/fallback; enumerate to handle fitted nonmonotonicity.

    The transferred two-regime cost fit can decrease at 512->768. Enumerating
    ten tiers preserves the source maximum-feasible objective without assuming
    monotonicity or weakening the baseline through a binary-search miss.
    """
    best = tiers[0]
    best_prediction = predict(max(0, best-decode), decode, decode_context, prefill_context)
    evaluated = [(best, best_prediction)]
    if slack <= 0:
        return max(0, best-decode), best, best_prediction, evaluated, 'nonpositive_slack_minimum_total_tier'
    found = False
    for total in tiers:
        prediction = (best_prediction if total == tiers[0] else
                      predict(max(0, total-decode), decode, decode_context, prefill_context))
        if total != tiers[0]:
            evaluated.append((total, prediction))
        if 1.2*prediction <= slack:
            best, best_prediction, found = total, prediction, True
    return max(0, best-decode), best, best_prediction, evaluated, ('maximum_feasible_enumerated_tier' if found else 'no_feasible_tier_minimum_fallback')


class ComponentController:
    design = None
    calibration = None

    def __init__(self, name, engine):
        assert name in ('niyama_component', 'fixed1024'), name
        self.name, self.engine = name, engine
        self.pending = self.awaiting_forward = None
        self.decisions = self.completed = 0
        self.changed_cap = self.actual_above_fixed = self.zero_prefill_steps = 0

    def external(self, request):
        return self.engine.output_processor.request_states[request.request_id].external_req_id

    def predict(self, p, d, cd, cp):
        # Model schema is normalized at load, no online fitting or GPU sampling.
        model = self.calibration['runtime_models']['le512' if p+d <= 512 else 'gt512']
        return model['intercept_s'] + sum(a*b for a, b in zip(model['coefficients'], (p, d, cd, cp)))

    def choose(self, decode_count, scheduler, rows, step_index, now):
        decision_time = now()
        decoders = [r for r in scheduler.running if r.num_computed_tokens >= r.num_prompt_tokens]
        assert len(decoders) == decode_count
        backlog = sum(max(0, r.num_prompt_tokens-r.num_computed_tokens) for r in scheduler.requests.values())
        prefills = [r for r in scheduler.running if r.num_computed_tokens < r.num_prompt_tokens]
        if not prefills:
            prefills = [r for r in scheduler.waiting if r.num_computed_tokens < r.num_prompt_tokens]
        head = prefills[0] if prefills else None
        cp = head.num_computed_tokens if head is not None else 0
        # Source context includes sampled output token not yet consumed by vLLM.
        cd = sum(r.num_computed_tokens+1 for r in decoders)
        pool = scheduler.kv_cache_manager.block_pool
        free = pool.get_num_free_blocks()
        block_size = self.engine.vllm_config.cache_config.block_size
        memory_budget = (free*block_size)//32
        tiers = LOW_MEMORY_TIERS if memory_budget < 1000 else TOTAL_TIERS
        min_slack, constraint = None, None
        selected_total = prediction = None
        evaluated = []
        if self.name == 'fixed1024':
            cap, reason = 1024, 'fixed_baseline'
        else:
            tau = self.design['batch_token_deadline_s']
            for request in decoders:
                external = self.external(request)
                row = rows[external]
                times = row['token_times_s']
                assert times and times[-1] <= decision_time
                observed = len(row['output_token_ids'])
                assert observed == len(times)
                assert request.num_computed_tokens+1 == row['prompt_tokens']+observed
                supplied_length = row['max_tokens']
                internal_prefill_target = self.design['primary_slo']['completion_s']-supplied_length*tau
                actual_prefill = times[0]-row['arrival_s']
                deadline = row['arrival_s']+max(actual_prefill, internal_prefill_target)+(observed+1)*tau
                slack = deadline-decision_time
                if min_slack is None or slack < min_slack:
                    min_slack = slack
                    constraint = dict(request_id=external, observed_outputs=observed,
                        declared_output_tokens=supplied_length, internal_prefill_target_s=internal_prefill_target,
                        actual_prefill_s=actual_prefill, deadline_s=deadline)
            if backlog:
                source_slack = 1e18 if min_slack is None else min_slack
                cap, selected_total, prediction, evaluated, reason = choose_by_slack(
                    source_slack, decode_count, cd, cp, tiers, self.predict)
            else:
                # Source retains native remaining chunk budget when prefill queue empty.
                cap, reason = max(0, 4096-decode_count), 'no_prefill_backlog_native_remaining_budget'
        self.pending = dict(step_index=step_index, decision_time_s=decision_time,
            decode_count=decode_count, decode_context_source=cd,
            head_prefill_context_estimate=cp, head_prefill_id=None if head is None else self.external(head),
            prefill_backlog_tokens=backlog, free_kv_blocks=free, memory_budget_tokens=memory_budget,
            legal_total_tiers=tiers, legal_prefill_caps=[max(0, t-decode_count) for t in tiers],
            min_slack_s=min_slack, limiting_request=constraint, evaluated_total_predictions=evaluated,
            baseline_cap=1024, candidate_cap=cap if self.name == 'niyama_component' else None,
            requested_cap=cap, selected_total_tier=selected_total, predicted_host_step_s=prediction,
            reason=reason, execution_status='REQUESTED', forward_completed=False)
        self.decisions += 1
        self.changed_cap += int(backlog > 0 and cap != 1024)
        return cap

    def native_failed(self, exc):
        self.pending.update(execution_status='NATIVE_RAISED', error=repr(exc))

    def native_returned(self, out, before):
        self.pending.update(execution_status='NATIVE_RETURNED_FORWARD_UNCONFIRMED',
            native_scheduled_tokens=[dict(request_id=before[rid][2], tokens=n,
                computed_tokens_before=before[rid][0], prompt_tokens=before[rid][1])
                for rid, n in out.num_scheduled_tokens.items()],
            native_preempted_ids=list(out.preempted_req_ids or []))

    def attach(self, step, rows):
        record = self.pending
        p, d = step['prefill_tokens'], step['decode_tokens']
        actual_cp = sum(x['computed_start'] for x in step['requests'] if x['prefill_tokens'] > 0)
        cap = record['requested_cap']
        reason = None
        if p < cap:
            reason = 'backlog_below_cap' if record['prefill_backlog_tokens'] < cap else 'native_underfill_unresolved'
        record.update(execution_status='NATIVE_SCHEDULED_FORWARD_UNCONFIRMED',
            actual_prefill_tokens=p, actual_decode_tokens=d, actual_prefill_context=actual_cp,
            unused_budget_tokens=cap-p, unused_budget_reason=reason)
        # Full native details already exist in step.requests on successful attach.
        record.pop('native_scheduled_tokens', None)
        step['component_decision'] = record
        self.awaiting_forward = step
        self.pending = None

    def observe(self, elapsed, p, d):
        assert self.awaiting_forward is not None
        record = self.awaiting_forward['component_decision']
        record.update(execution_status='FORWARD_COMPLETED', forward_completed=True,
                      observed_host_step_s=elapsed)
        self.completed += 1
        self.actual_above_fixed += int(p > 1024)
        self.zero_prefill_steps += int(record['prefill_backlog_tokens'] > 0 and p == 0)
        self.awaiting_forward = None

    def summary(self):
        return dict(policy=self.name, decision_count=self.decisions, completed_forward_count=self.completed,
            backlog_steps_requested_cap_differs_fixed1024=self.changed_cap,
            actual_prefill_above_fixed1024=self.actual_above_fixed,
            backlog_steps_zero_actual_prefill=self.zero_prefill_steps,
            pending_decision=self.pending,
            pending_forward=None if self.awaiting_forward is None else self.awaiting_forward['component_decision'])


def derived_episode():
    source = inspect.getsource(base.episode)
    replacements = [
        ('policy = Policy(policy_name, target_ms)', 'policy = ComponentController(policy_name, engine)'),
        ('budget = policy.choose(len(decoding))', 'budget = policy.choose(len(decoding), scheduler, rows, len(steps), now)'),
        ('        out = native(*args, **kwargs)\n',
         '        try:\n            out = native(*args, **kwargs)\n'
         '        except BaseException as exc:\n            policy.native_failed(exc)\n            raise\n'
         '        policy.native_returned(out, before)\n'),
        ('        return out\n', '        policy.attach(steps[-1], rows)\n        return out\n'),
        ("        dump(path/'raw.json',data)\n", "        data['component'] = policy.summary()\n        dump(path/'raw.json',data)\n")]
    for old, new in replacements:
        assert source.count(old) == 1, ('Parent source changed', old)
        source = source.replace(old, new, 1)
    namespace = dict(base.__dict__, ComponentController=ComponentController)
    exec(compile(source, str(ROOT/'run_component.py')+'::derived_episode', 'exec'), namespace)
    return namespace['episode'], source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-lock', type=float, default=0)
    parser.add_argument('--check-only', action='store_true', help='Verify frozen local inputs without GPU initialization')
    args = parser.parse_args()
    design = json.loads((ROOT/'design.json').read_text())
    assert design['status_at_freeze'] == 'PRE_GPU_FROZEN'
    assert design['primary_slo'] == dict(ttft_s=4, gap_s=.1, completion_s=20)
    assert design['batch_token_deadline_s'] == (20-4)/512
    for name, digest in design['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    workload = (ROOT/design['workload']).resolve()
    assert sha(workload) == design['workload_sha256']
    ComponentController.design = design
    ComponentController.calibration = json.loads((ROOT/'calibration.json').read_text())
    assert sha(ROOT/'calibration.json') == design['calibration_sha256']
    assert not ComponentController.calibration['fit_errors']
    assert set(ComponentController.calibration['runtime_models']) == {'le512', 'gt512'}
    base.episode, source = derived_episode()
    if args.check_only:
        print(json.dumps(dict(status='FROZEN_INPUTS_VERIFIED_NO_GPU',
            resource_check='Not performed in check-only mode',
            design_sha256=sha(ROOT/'design.json'))))
        return 0
    assert base.UUID == design['gpu_uuid'], 'Explicit authorized physical GPU required'
    lock = Path('/root/autodl-tmp/moe-research-gpu.lock')
    assert lock.stat().st_ino == design['shared_lock_inode']
    assert shutil.disk_usage('/tmp').free >= design['root_free_bytes_required'], 'Root filesystem below frozen 4GiB reserve; no GPU initialized'
    original_dump = base.dump
    def dump(path, value):
        if path.name == 'status.json' and value.get('status') == 'INITIALIZING':
            free = shutil.disk_usage('/tmp').free
            if free < design['root_free_bytes_required']:
                original_dump(path, dict(status='RESOURCE_BLOCKED_NO_GPU_INITIALIZED',
                    pid=os.getpid(), root_free_bytes=free, required=design['root_free_bytes_required']))
                raise RuntimeError('Root reserve fell below 4GiB while waiting for common lock')
        if path.name == 'protocol.json':
            value = dict(value, experiment_kind='NIYAMA_CHUNK_COMPONENT_DEVELOPMENT',
                         frozen_component_design=design, frozen_component_design_sha256=sha(ROOT/'design.json'))
            (path.parent/'derived_episode.py').write_text(source)
            original_dump(path.parent/'calibration.json', ComponentController.calibration)
        original_dump(path, value)
    base.dump = dump
    sys.argv = [str(ROOT.parent/'run_confirmation.py'), '--output', str(args.output.resolve()),
        '--workload', str(workload), '--policies', ','.join(design['policies']),
        '--warm-policies', ','.join(design['warm_policies']), '--wait-lock', str(args.wait_lock)]
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())

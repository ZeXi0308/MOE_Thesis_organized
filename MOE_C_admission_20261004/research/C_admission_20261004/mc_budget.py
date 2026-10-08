"""Production-guarded adaptation of an existing future-round peak principle.

Future-round FIFO/declared-upper-length accounting is prior work (MC-Benchmark,
arXiv:2502.07115v5), not a new algorithm. Only an ordinary budget denial may be
relaxed, after checking this pinned synchronous runtime's current physical state.
Natural EOS can reduce occupancy; no future realized output length is used.
"""
import hashlib
from functools import partial
import inspect
from pathlib import Path
import time

import admission_probe as native
import declared_budget


EXTRA_SOURCE_SHA = {
    'request.py': '92124fbad28cda49bd06fa12c2c4fd5f53fc9381ddb4dc35f275c5ccfbd27378',
    'utils.py': '85e82eae555a03497ad2ac1540ed562a6c36fc26185aa6233725c914816aa1b3',
    'offloading/scheduler.py': '89ac26a80fbc29b9bcaa5a0daba88fb6309247f9bb053e48d2d61efa7d9d66f1',
}


def peak_envelope(rows, p, B, block_size=16):
    """Exact peak of the declared discrete envelope at 0, each r-1, and B.

    h=0 already includes this step's old decode slot allocation. An old request
    contributes ceil((n+h)/block_size) for h<r and zero from h=r onwards. The new
    request retains ceil((p+min(h,B))/block_size) even after B: its prefill may be
    delayed, so its final declared footprint is never credited as a future free.
    Between old releases/new plateau the footprint is nondecreasing. Residue
    counts evaluate each necessary checkpoint without a quadratic request scan.
    """
    if any(type(x) is not int or x <= 0 for x in (p, B, block_size)):
        raise ValueError('p, B and block_size must be positive integers')
    rows = list(rows)
    if any(type(row['n']) is not int or row['n'] <= 0 or
           type(row['r']) is not int or row['r'] <= 0 for row in rows):
        raise ValueError('Each old row needs positive integer n and r')
    points = sorted({0, B, *(row['r']-1 for row in rows)})
    remaining = sorted(rows, key=lambda row: row['r'])
    residues = [0]*block_size
    quotients = 0
    for row in rows:
        quotients += row['n']//block_size
        residues[row['n'] % block_size] += 1
    cursor, live = 0, len(rows)
    checkpoints = []
    for h in points:
        while cursor < len(remaining) and remaining[cursor]['r'] <= h:
            n = remaining[cursor]['n']
            quotients -= n//block_size
            residues[n % block_size] -= 1
            cursor, live = cursor+1, live-1
        horizon_pages, horizon_residue = divmod(h, block_size)
        # For u>0, ceil((a+u)/b) is 1 plus [a>b-u].  For u=0,
        # it is [a>0].  Release updates above keep sum(residues)==live.
        residue_pages = (live-residues[0] if horizon_residue == 0 else
                         live+sum(residues[block_size-horizon_residue+1:]))
        old = quotients + live*horizon_pages + residue_pages
        new = (p+min(h, B)+block_size-1)//block_size
        checkpoints.append(dict(h=h, blocks=old+new))
    peak = max(checkpoints, key=lambda row: row['blocks'])
    return dict(peak_blocks=peak['blocks'], peak_h=peak['h'], checkpoints=checkpoints)


def _validate_parameters(use_future_peak, maximum_declared_blocks, budget_blocks,
                         allow_scheduled_prefill=False):
    if type(use_future_peak) is not bool:
        raise ValueError('use_future_peak must be a boolean')
    if type(allow_scheduled_prefill) is not bool:
        raise ValueError('allow_scheduled_prefill must be a boolean')
    if allow_scheduled_prefill and not use_future_peak:
        raise ValueError('Scheduled-prefill adaptation requires use_future_peak=True')
    if maximum_declared_blocks is not None and (type(maximum_declared_blocks) is not int or
                                                maximum_declared_blocks < budget_blocks):
        raise ValueError('maximum_declared_blocks must be an integer at least the physical budget')
    if not use_future_peak and maximum_declared_blocks is None:
        raise ValueError('Static borrowing requires an explicit maximum_declared_blocks')


class Gate(declared_budget.Gate):
    def __init__(self, scheduler, cap=256, budget_blocks=32768, *, use_future_peak=True,
                 maximum_declared_blocks=None, allow_scheduled_prefill=False):
        _validate_parameters(use_future_peak, maximum_declared_blocks, budget_blocks,
                             allow_scheduled_prefill)
        super().__init__(scheduler, cap=cap, budget_blocks=budget_blocks)
        self.use_future_peak, self.maximum_declared_blocks = use_future_peak, maximum_declared_blocks
        self.allow_scheduled_prefill = allow_scheduled_prefill
        self.mode = ('mc_budget_phase' if allow_scheduled_prefill else
                     'mc_budget_static' if not use_future_peak else
                     'mc_budget_capped' if maximum_declared_blocks is not None else 'mc_budget')
        self.mc_authorizations = {}
        self.mc = dict(attempted_evaluations=0, eligible_evaluations=0,
            requested_relaxations=0, successful_relaxations=0,
            guard_fallback_counts={}, peak_rejections=0, peak_evaluations=0,
            declared_ceiling_rejections=0, first_eligible=None,
            scheduled_prefill_requested_relaxations=0,
            scheduled_prefill_successful_relaxations=0,
            first_scheduled_prefill_relaxation=None)

    @staticmethod
    def _request_guard(req):
        params = req.sampling_params
        if (params is None or not hasattr(params, 'min_tokens') or
                not 0 <= params.min_tokens <= req.max_tokens):
            return 'sampling_minimum'
        if (getattr(req, 'structured_output_request', None) is not None or
                getattr(params, 'structured_outputs', None) is not None or
                req.pooling_params is not None or req.resumable or req.has_encoder_inputs):
            return 'non_plain_generation'
        return None

    def _old_rows(self, req, row):
        failure = self._request_guard(req)
        if failure:
            return failure, [], None
        running = self.s.running
        if (self.s.num_waiting_for_streaming_input or len(running) != len(self.admitted_ids) or
                {r.request_id for r in running} != self.admitted_ids):
            return 'admitted_not_all_running', [], None
        manager = self.s.kv_cache_manager.coordinator.single_type_managers[0]
        rows, physical = [], 0
        for old in running:
            failure = self._request_guard(old)
            if failure:
                return failure, rows, None
            n, output = old.num_tokens, old.num_output_tokens
            remaining = old.max_tokens-output
            computed = old.num_computed_tokens
            scheduled = self.scheduled_tokens.get(old.request_id, 0)
            scheduled_prefill = (self.allow_scheduled_prefill and output == 0 and
                n == old.num_prompt_tokens and computed < old.num_prompt_tokens and
                scheduled > 0 and computed+scheduled == n)
            complete_decode = computed == n-1 and computed >= old.num_prompt_tokens
            if (old.status.name != 'RUNNING' or old.is_finished() or remaining <= 0 or
                    n != old.num_prompt_tokens+output or
                    not (complete_decode or scheduled_prefill)):
                return 'not_complete_prompt_decode', rows, None
            if (old.num_output_placeholders or old.spec_token_ids or
                    getattr(old, 'num_in_flight_tokens', 0) or
                    getattr(old, 'async_tokens_to_discard', 0) or
                    old.next_decode_eligible_step > self.s.current_step or
                    (not scheduled_prefill and scheduled != 1)):
                return 'not_synchronous_one_token', rows, None
            pages = len(manager.req_to_blocks.get(old.request_id, ()))
            if pages != (n+15)//16:
                return 'physical_page_geometry', rows, None
            physical += pages
            old_row = dict(request_id=old.request_id, n=n, r=remaining, allocated_blocks=pages)
            if self.allow_scheduled_prefill:
                old_row.update(scheduled_prefill=scheduled_prefill, computed=computed,
                    output=output, prompt=old.num_prompt_tokens, max_tokens=old.max_tokens,
                    scheduled_tokens=scheduled, status=old.status.name,
                    preemptions=old.num_preemptions)
            rows.append(old_row)
        if self.s.kv_cache_manager.block_pool.get_num_free_blocks() != row['free_blocks']:
            return 'physical_free_changed', rows, physical
        if row['free_blocks']+physical != self.budget_blocks:
            return 'unaccounted_physical_pages', rows, physical
        return None, rows, physical

    def before_allocate(self, req, *, token_budget, **kwargs):
        check = super().before_allocate(req, token_budget=token_budget, **kwargs)
        row = check['row']
        if row is None:
            return check
        row.update(ordinary_budget_allowed=row['final_allowed'], mc_attempted=False,
            mc_eligible=None, mc_guard_reason=None, mc_peak_blocks=None, mc_peak_h=None,
            mc_old_physical_blocks=None, changed_by_mc_budget=False,
            use_future_peak=self.use_future_peak, maximum_declared_blocks=self.maximum_declared_blocks,
            allow_scheduled_prefill=self.allow_scheduled_prefill,
            mc_scheduled_prefill_count=None, changed_by_scheduled_prefill=False,
            mc_peak_computed=False, declared_ceiling_allowed=None, declared_ceiling_denied=False,
            mc_limit_reason=None)
        if not (row['reason'] == 'declared_budget' and row['baseline_allowed']):
            return check
        row['mc_attempted'] = True
        self.mc['attempted_evaluations'] += 1
        if (kwargs['num_new_tokens'] <= 0 or kwargs.get('delay_cache_blocks', False) or
                kwargs.get('num_external_computed_tokens', 0)):
            failure, rows, physical = 'not_new_local_prefill', [], None
        else:
            failure, rows, physical = self._old_rows(req, row)
        row.update(mc_eligible=failure is None, mc_guard_reason=failure,
                   mc_old_physical_blocks=physical)
        if failure:
            counts = self.mc['guard_fallback_counts']
            counts[failure] = counts.get(failure, 0)+1
            return check
        self.mc['eligible_evaluations'] += 1
        scheduled_prefill_count = sum(old.get('scheduled_prefill', False) for old in rows)
        row['mc_scheduled_prefill_count'] = scheduled_prefill_count
        ceiling_allowed = (self.maximum_declared_blocks is None or
            row['budget_after_if_admitted_blocks'] <= self.maximum_declared_blocks)
        row.update(declared_ceiling_allowed=ceiling_allowed, declared_ceiling_denied=not ceiling_allowed)
        envelope = None
        if ceiling_allowed and self.use_future_peak:
            envelope = peak_envelope(rows, req.num_prompt_tokens, req.max_tokens)
            self.mc['peak_evaluations'] += 1
            row.update(mc_peak_computed=True, mc_peak_blocks=envelope['peak_blocks'],
                       mc_peak_h=envelope['peak_h'])
        event = None
        if (self.mc['first_eligible'] is None or
                (scheduled_prefill_count and self.mc['first_scheduled_prefill_relaxation'] is None)):
            event = dict(decision_index=len(self.decisions)-1,
                request_id=req.request_id, t=row['t'], old_rows=rows,
                new_prompt_tokens=req.num_prompt_tokens, new_max_tokens=req.max_tokens,
                free_blocks=row['free_blocks'], envelope=envelope,
                use_future_peak=self.use_future_peak, maximum_declared_blocks=self.maximum_declared_blocks,
                allow_scheduled_prefill=self.allow_scheduled_prefill,
                mc_scheduled_prefill_count=scheduled_prefill_count,
                token_budget=token_budget, num_new_tokens=kwargs['num_new_tokens'],
                declared_ceiling_allowed=ceiling_allowed, mc_peak_computed=row['mc_peak_computed'])
        if self.mc['first_eligible'] is None:
            self.mc['first_eligible'] = event
        if not ceiling_allowed:
            self.mc['declared_ceiling_rejections'] += 1
            row['mc_limit_reason'] = 'declared_ceiling'
            return check
        if envelope is not None and envelope['peak_blocks'] > self.budget_blocks:
            self.mc['peak_rejections'] += 1
            row['mc_limit_reason'] = 'future_peak'
            return check
        self.mc['requested_relaxations'] += 1
        if scheduled_prefill_count:
            self.mc['scheduled_prefill_requested_relaxations'] += 1
            if self.mc['first_scheduled_prefill_relaxation'] is None:
                self.mc['first_scheduled_prefill_relaxation'] = event
        row.update(denied=False, reason='mc_budget_allow' if self.use_future_peak else 'mc_budget_static_allow', final_allowed=True,
            changed_by_declared_budget=False, changed_by_mc_budget=True,
            changed_by_scheduled_prefill=bool(scheduled_prefill_count))
        # Undo only the ordinary-budget barrier created by this very head row.
        assert self.barrier == 'declared_budget' and self.barrier_decision_index == len(self.decisions)-1
        self.barrier = self.barrier_deadline = self.barrier_decision_index = None
        return check

    def after_allocate(self, req, check, result):
        super().after_allocate(req, check, result)
        if result is not None and check['row'] is not None and check['row']['changed_by_mc_budget']:
            self.mc_authorizations[req.request_id] = check['row']

    def admitted(self, req):
        native.Gate.admitted(self, req)
        if req.request_id in self.charges:
            return
        required, before = self.declared_pages(req), self.budget_used_blocks
        authorization = self.mc_authorizations.pop(req.request_id, None)
        assert before+required <= self.budget_blocks or authorization is not None
        assert self.maximum_declared_blocks is None or before+required <= self.maximum_declared_blocks
        if authorization is not None:
            assert authorization['native_allocation_result'] is True
            self.mc['successful_relaxations'] += 1
            if authorization['changed_by_scheduled_prefill']:
                self.mc['scheduled_prefill_successful_relaxations'] += 1
        self.charges[req.request_id] = required
        self.budget_used_blocks += required
        self.budget_peak_blocks = max(self.budget_peak_blocks, self.budget_used_blocks)
        self.budget_events.append(dict(t=time.perf_counter()-self.origin, request_id=req.request_id,
            action='charge', budget_required_blocks=required, budget_before_blocks=before,
            budget_after_blocks=self.budget_used_blocks, mc_relaxation=authorization is not None,
            scheduled_prefill_relaxation=bool(authorization and authorization['changed_by_scheduled_prefill'])))

    def report(self):
        result = super().report()
        result.update(budget_overrides=self.mc['successful_relaxations'],
            use_future_peak=self.use_future_peak, maximum_declared_blocks=self.maximum_declared_blocks,
            allow_scheduled_prefill=self.allow_scheduled_prefill,
            mc_budget=dict(self.mc, source_sha256=EXTRA_SOURCE_SHA,
                use_future_peak=self.use_future_peak, maximum_declared_blocks=self.maximum_declared_blocks,
                allow_scheduled_prefill=self.allow_scheduled_prefill,
                prior_work='MC-Benchmark arXiv:2502.07115v5; future-round FIFO/upper-length peak principle',
                semantics='Production-guarded adaptation, not a new algorithm. Ordinary full declared '
                    'charges remain recorded and may exceed the pool after an authorized relaxation. '
                    'Only the guarded physical peak controls that relaxation; native fit/cap remain '
                    'mandatory. Failures use the original FIFO budget barrier and scan-break rule. '
                    'Requested relaxations are decisions; successful relaxations are actual native '
                    'allocation followed by admission. Starts are host schedule-return boundaries. '
                    'h=0 includes old decode slots allocated this step; old requests disappear at '
                    'h=r after their last declared output. New request plateau is never freed in '
                    'the envelope. Pending store reuse can flush and cost wall time; no time gain '
                    'or quality equality follows from round accounting.'))
        if self.maximum_declared_blocks is not None:
            result['mc_budget']['semantics'] = (
                'Structure-matched borrowed-admission comparison, not a new algorithm. Both modes '
                'first allow the original full-declared sum within the physical budget. Only an '
                'ordinary budget denial with native fit/cap permission can attempt the identical '
                'old-request/physical-accounting guards and shared maximum_declared_blocks. '
                + ('The capped MC arm additionally computes the future-round envelope and requires '
                   'its peak within the physical budget. ' if self.use_future_peak else
                   'The static arm does not call peak_envelope or record an estimated peak; it '
                   'allows once the common guards and declared ceiling pass. ') +
                'Neither mode changes native allocation, running/victim/recovery behavior or adds '
                'an age override. Virtual full charges remain recorded and cannot exceed the '
                'shared declared ceiling. mc_eligible denotes the common runtime guards only; '
                'declared_ceiling_denied and mc_limit_reason distinguish limit refusals. '
                'Requested relaxations are decisions, successful relaxations require actual native '
                'allocation followed by admission. FIFO has no counterfactual fit or peak. '
                'No uncomputed static peak, service benefit or quality equality is inferred.')
        if self.allow_scheduled_prefill:
            result['mc_budget']['semantics'] = (
                'Scheduled-prefill qualification adaptation of MC-Benchmark, not a new algorithm. '
                'In addition to strict one-token decodes, an old zero-output request may qualify '
                'only when its entire remaining prompt is scheduled in this current step. Its '
                'h=0 footprint is the prompt, and its declared last output is produced at h=B-1; '
                'it disappears at h=B. Actual per-request pages and all synchronous runtime guards '
                'remain checked. The original ordinary-budget path stays unchanged; borrowing '
                'requires the optional declared ceiling, native fit/cap and the physical future '
                'peak to pass. Truly partial prefills fall back '
                'to the ordinary FIFO budget barrier. changed_by_scheduled_prefill marks a requested '
                'allow that strict qualification would refuse, not an allocation or execution. '
                'The first_scheduled_prefill_relaxation decision_index identifies its native '
                'allocation result; successful counters require allocation followed by admission. '
                'No future realized output, service benefit or quality equality is inferred.')
        result['semantics'] = result['mc_budget']['semantics']
        return result


def install(scheduler, cap=256, budget_blocks=32768, *, use_future_peak=True,
            maximum_declared_blocks=None, allow_scheduled_prefill=False):
    _validate_parameters(use_future_peak, maximum_declared_blocks, budget_blocks,
                         allow_scheduled_prefill)
    from vllm.v1.request import Request
    from vllm.v1.core.sched.utils import check_stop
    connector_scheduler = scheduler.connector.connector_scheduler
    for obj, name in ((Request, 'request.py'), (check_stop, 'utils.py'),
                      (type(connector_scheduler), 'offloading/scheduler.py')):
        assert hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest() == EXTRA_SOURCE_SHA[name]
    assert scheduler.max_num_scheduled_tokens == 1024 and cap <= 256
    assert not scheduler.defer_block_free and not scheduler.use_pp
    return declared_budget.install(scheduler, cap, budget_blocks,
        gate_factory=partial(Gate, use_future_peak=use_future_peak,
                             maximum_declared_blocks=maximum_declared_blocks,
                             allow_scheduled_prefill=allow_scheduled_prefill))

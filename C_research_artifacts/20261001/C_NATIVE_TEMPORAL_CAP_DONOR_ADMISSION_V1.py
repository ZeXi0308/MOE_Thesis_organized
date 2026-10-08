"""Cap-only, single-donor FIFO admission; a partial CacheOPT-inspired ablation.

This is not CacheOPT. The native allocator remains the sole owner of physical
KV blocks. Here ``active`` is committed *logical quota*; ``full_targets`` is
each request's full prompt-plus-cap upper bound. One partial receiver may have
less committed quota than its target until its donor actually finishes.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

from C_NATIVE_MAX_BOUND_ADMISSION import CAPACITY, MaxBoundAdmission, blocks
from C_NATIVE_RETIREMENT_ADMISSION_V1 import decode_ready

BLOCK_SIZE = 16
GUARD_TOKENS = 1
MAX_SEQUENCES = 32


def settle_finished_quota(active: dict[str, int], full_targets: dict[str, int],
                          promise: dict | None, request_id: str) -> tuple[dict | None, list[dict]]:
    """Settle one *already native-freed* request; no physical allocator calls.

    This is deliberately separate from the hook so donor/receiver free order can
    be checked on CPU. It mutates the two quota ledgers only after validating
    the transition and returns the next outstanding promise plus event records.
    """
    if request_id not in active or request_id not in full_targets:
        raise ValueError("native free lacked an active logical reservation")
    quota, target = active[request_id], full_targets[request_id]
    if type(quota) is not int or type(target) is not int or not 0 < quota <= target:
        raise ValueError("invalid committed quota or full upper bound")
    before = sum(active.values())
    events = []
    if promise is not None:
        donor, receiver = promise["donor_id"], promise["receiver_id"]
        shortfall = promise["transfer_blocks"]
        if request_id == donor:
            if (receiver not in active or receiver not in full_targets
                    or quota != target or shortfall > quota
                    or active[receiver] + shortfall != full_targets[receiver]):
                raise ValueError("donor release cannot complete its receiver quota")
        elif request_id == receiver:
            if quota != promise["receiver_initial_blocks"] or quota + shortfall != target:
                raise ValueError("receiver cancellation would release the wrong quota")
        elif quota != target:
            raise ValueError("unrelated resident has a partial quota")
    elif quota != target:
        raise ValueError("partial receiver lacks an outstanding promise")

    del active[request_id]
    del full_targets[request_id]
    if promise is not None and request_id == promise["donor_id"]:
        receiver = promise["receiver_id"]
        active[receiver] += promise["transfer_blocks"]
        events.append(dict(event="transfer", donor_id=request_id,
                           receiver_id=receiver, blocks=promise["transfer_blocks"],
                           receiver_committed_blocks=active[receiver]))
        promise = None
    elif promise is not None and request_id == promise["receiver_id"]:
        events.append(dict(event="cancel_promise", receiver_id=request_id,
                           donor_id=promise["donor_id"],
                           untransferred_blocks=promise["transfer_blocks"]))
        promise = None
    events.append(dict(event="release", request_id=request_id, blocks=quota,
                       full_upper_bound_blocks=target))
    if sum(active.values()) > before or sum(active.values()) > CAPACITY:
        raise ValueError("quota increased or exceeded capacity after native free")
    return promise, events


class TemporalCapDonorAdmission(MaxBoundAdmission):
    def __init__(self, engine):
        super().__init__(engine)
        scheduler = self.scheduler
        cfg = engine.vllm_config
        if not (scheduler.max_num_scheduled_tokens == 1024
                and scheduler.num_sampled_tokens_per_step == 1
                and not scheduler.need_mamba_block_aligned_split
                and cfg.scheduler_config.async_scheduling is False
                and cfg.parallel_config.pipeline_parallel_size == 1
                and cfg.parallel_config.data_parallel_size == 1):
            raise ValueError("donor rule requires synchronous single-token single-GPU execution")
        self.full_targets: dict[str, int] = {}
        self.promise: dict | None = None
        self.never_donor: set[str] = set()
        self.protected: set[str] = set()
        self.expected_outputs: dict[str, int] = {}
        self.eligible_calls = self.promises = self.transfers = self.cancellations = 0
        self.full_bound_sum_peak = self.physical_blocks_peak = 0
        self.decision_seconds = self.bookkeeping_seconds = 0.0

    def check_ledger(self, running: list) -> None:
        ids = {request.request_id for request in running}
        if set(self.active) != ids or set(self.full_targets) != ids:
            self.fail("quota/full-target ledger differs from native running requests")
        if sum(self.active.values()) > CAPACITY:
            self.fail("committed logical quota exceeds capacity")
        for request in running:
            rid = request.request_id
            if self.full_targets[rid] != blocks(request):
                self.fail("resident full upper bound changed")
            if not 0 < self.active[rid] <= self.full_targets[rid]:
                self.fail("resident committed quota is invalid")
        partial = {rid for rid in ids if self.active[rid] < self.full_targets[rid]}
        if self.promise is None:
            if partial:
                self.fail("partial receiver has no donor promise")
        else:
            donor, receiver = self.promise["donor_id"], self.promise["receiver_id"]
            if (partial != {receiver} or donor == receiver or donor not in ids
                    or receiver not in ids
                    or donor in self.never_donor or receiver not in self.never_donor
                    or self.active[donor] != self.full_targets[donor]
                    or self.active[receiver] != self.promise["receiver_initial_blocks"]
                    or self.active[receiver] + self.promise["transfer_blocks"]
                        != self.full_targets[receiver]):
                self.fail("outstanding single-donor promise ledger differs")

    def plan_fifo(self, running: list, waiting: list) -> list[dict]:
        """Ordinary full-bound FIFO prefix, then at most one partial head."""
        if self.promise is not None:
            return []
        free = CAPACITY - sum(self.active.values())
        selected = []
        for request in waiting:
            if len(running) + len(selected) >= MAX_SEQUENCES:
                break
            full = blocks(request)
            if free >= full:
                selected.append(dict(request=request, mode="full_bound",
                                     committed_blocks=full, full_blocks=full))
                free -= full
                continue
            runway = BLOCK_SIZE * free - request.num_prompt_tokens
            if runway < 0:
                break
            shortfall = full - free
            donor = next((resident for resident in running
                          if (resident.request_id not in self.never_donor
                              and resident.max_tokens - resident.num_output_tokens
                              + GUARD_TOKENS <= runway
                              and blocks(resident) >= shortfall)), None)
            if donor is not None:
                selected.append(dict(request=request, mode="single_donor",
                    committed_blocks=free, full_blocks=full,
                    donor_id=donor.request_id, donor_full_blocks=blocks(donor),
                    donor_horizon_calls=donor.max_tokens - donor.num_output_tokens,
                    transfer_blocks=shortfall, receiver_runway_tokens=runway))
            break  # Never bypass a blocked FIFO head or form another promise.
        return selected

    def schedule(self, *args, **kwargs):
        started = perf_counter()
        scheduler = self.scheduler
        self.schedule_calls += 1
        running, waiting = list(scheduler.running), list(scheduler.waiting)
        before_ids = {request.request_id for request in running}
        if (scheduler.skipped_waiting or scheduler.num_waiting_for_streaming_input
                or any(request.status != self.RequestStatus.WAITING for request in waiting)):
            self.fail("unmodeled native waiting queue or request status")
        self.check_ledger(running)
        for rid, expected in self.expected_outputs.items():
            if rid in before_ids and scheduler.requests[rid].num_output_tokens != expected:
                self.fail("protected decode failed its one-output progress condition")
        self.protected.intersection_update(before_ids)
        if [request.request_id for request in running[:len(self.protected)]] != [
                request.request_id for request in running
                if request.request_id in self.protected]:
            self.fail("protected decoders no longer form the running prefix")

        eligible = all(decode_ready(request) for request in running)
        selected = []
        if eligible and self.promise is None:
            self.eligible_calls += 1
            self.protected = set(before_ids)
            selected = self.plan_fifo(running, waiting)
        self.hold_calls += len(selected) < len(waiting)
        expected = {request.request_id: request.num_output_tokens + 1
                    for request in running if request.request_id in self.protected}
        if any(not decode_ready(request)
               or request.next_decode_eligible_step > scheduler.current_step + 1
               for request in running if request.request_id in self.protected):
            self.fail("protected prefix is not eligible for one-token decode")

        old_cap = scheduler.max_num_running_reqs
        scheduler.max_num_running_reqs = len(running) + len(selected)
        decision_s = perf_counter() - started
        self.decision_seconds += decision_s
        try:
            result = self.original_schedule(*args, **kwargs)
        finally:
            scheduler.max_num_running_reqs = old_cap
        bookkeeping_start = perf_counter()
        after_ids = {request.request_id for request in scheduler.running}
        new_ids = after_ids - before_ids
        planned_ids = [item["request"].request_id for item in selected]
        new_order = [request.request_id for request in scheduler.running
                     if request.request_id in new_ids]
        self.preemptions += len(result.preempted_req_ids or ())
        if (result.preempted_req_ids or not before_ids.issubset(after_ids)
                or new_order != planned_ids[:len(new_order)]
                or any(result.num_scheduled_tokens.get(rid) != 1
                       for rid in self.protected)):
            self.fail("native execution violated FIFO admission or protected progress")
        self.expected_outputs = expected
        for item in selected:
            request = item["request"]
            rid = request.request_id
            if rid not in new_ids:
                break
            committed, full = item["committed_blocks"], item["full_blocks"]
            self.active[rid] = committed
            self.full_targets[rid] = full
            event = dict(event="admit", request_id=rid, mode=item["mode"],
                         blocks=committed, full_upper_bound_blocks=full,
                         schedule_call=self.schedule_calls)
            if item["mode"] == "single_donor":
                if self.promise is not None:
                    self.fail("second donor promise attempted")
                self.never_donor.add(rid)
                self.promise = dict(donor_id=item["donor_id"], receiver_id=rid,
                    receiver_initial_blocks=committed, receiver_full_blocks=full,
                    donor_full_blocks=item["donor_full_blocks"],
                    donor_horizon_calls=item["donor_horizon_calls"],
                    transfer_blocks=item["transfer_blocks"],
                    receiver_runway_tokens=item["receiver_runway_tokens"],
                    admitted_call=self.schedule_calls)
                self.promises += 1
                self.events.append(dict(event="promise", **self.promise))
            self.events.append(event)
        self.check_ledger(list(scheduler.running))
        logical = sum(self.active.values())
        full_sum = sum(self.full_targets.values())
        used = CAPACITY - scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        if not 0 <= used <= logical <= CAPACITY:
            self.fail("physical KV or logical quota exceeded the committed capacity")
        self.reserved_blocks_peak = max(self.reserved_blocks_peak, logical)
        self.full_bound_sum_peak = max(self.full_bound_sum_peak, full_sum)
        self.physical_blocks_peak = max(self.physical_blocks_peak, used)
        self.steps.append(dict(call=self.schedule_calls,
            running_before=len(running), waiting_before=len(waiting),
            eligible_all_decode=eligible, protected_decode_requests=len(self.protected),
            selected=len(selected), newly_running=len(new_ids),
            committed_quota_blocks=logical, full_upper_bound_blocks=full_sum,
            physical_blocks_after_schedule=used, promise_outstanding=self.promise is not None,
            native_preemptions=len(result.preempted_req_ids or ()), decision_s=decision_s))
        self.bookkeeping_seconds += perf_counter() - bookkeeping_start
        return result

    def free(self, request, *args, **kwargs):
        # Native first: this hook observes actual completion and actual KV free.
        result = self.original_free(request, *args, **kwargs)
        started = perf_counter()
        rid = request.request_id
        if (request.status not in (self.RequestStatus.FINISHED_STOPPED,
                                   self.RequestStatus.FINISHED_LENGTH_CAPPED)
                or rid in self.scheduler.requests):
            self.fail("quota release lacked genuine completed native free")
        previous = self.promise
        try:
            self.promise, events = settle_finished_quota(
                self.active, self.full_targets, self.promise, rid)
        except ValueError as error:
            self.fail(f"logical donor/receiver settlement failed: {error}")
        self.transfers += int(previous is not None and rid == previous["donor_id"])
        self.cancellations += int(previous is not None and rid == previous["receiver_id"])
        for event in events:
            if event["event"] == "release":
                event["finish_status"] = str(request.status)
            self.events.append(event)
        used = CAPACITY - self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        if not 0 <= used <= sum(self.active.values()) <= CAPACITY:
            self.fail("physical KV exceeds remaining logical quota after native free")
        self.bookkeeping_seconds += perf_counter() - started
        return result

    def receipt(self) -> dict:
        result = super().receipt()
        logical_peak = result.pop("reserved_blocks_peak")
        drained = (result["drained"] and not self.full_targets and self.promise is None
                   and self.promises == self.transfers + self.cancellations)
        result.update(status="DRAINED" if result["status"] == "DRAINED" and drained else "ERROR",
            drained=drained, policy="cap_only_single_donor_fifo",
            formula="full-bound FIFO prefix; one guarded known-cap donor for a blocked head",
            logical_quota_peak_blocks=logical_peak,
            full_bound_sum_peak_blocks=self.full_bound_sum_peak,
            physical_blocks_peak=self.physical_blocks_peak,
            eligible_calls=self.eligible_calls, promises=self.promises,
            transfers=self.transfers, cancellations=self.cancellations,
            outstanding_promise=dict(self.promise) if self.promise else None,
            receiver_ids_ineligible_as_donors=sorted(self.never_donor),
            full_upper_bounds=dict(self.full_targets),
            decision_seconds=self.decision_seconds,
            bookkeeping_seconds=self.bookkeeping_seconds,
            policy_wall_seconds=self.decision_seconds + self.bookkeeping_seconds,
            progress_condition="All residents pure decode at admission; protected prior running prefix schedules and produces one token per call until cap/EOS; guard=1; violation aborts.",
            scope="Partial CacheOPT-inspired cap-only donor ablation; no predictor, SLO allocation, embedding, global reserve, or preemption selection")
        return result


@contextmanager
def install(engine):
    gate = TemporalCapDonorAdmission(engine)
    scheduler = gate.scheduler
    try:
        scheduler.schedule = gate.schedule
        scheduler._free_request = gate.free
        yield gate
    finally:
        if gate.had_schedule:
            scheduler.schedule = gate.original_schedule
        else:
            del scheduler.schedule
        if gate.had_free:
            scheduler._free_request = gate.original_free
        else:
            del scheduler._free_request

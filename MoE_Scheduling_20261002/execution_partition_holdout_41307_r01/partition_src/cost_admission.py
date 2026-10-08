"""Myopic admission heuristic, not a measured counterfactual or global optimum.

Both costs are request-seconds, but their horizons differ: one waiting step
versus estimated extra work across the new prompt. Existing prefill proceeds.
"""
import json
import math
from pathlib import Path
from admission_feedback import apply_nonpreemptive_limit


class CostAdmission:
    def __init__(self, profile_path, *, policy, maximum, token_budget,
                 training_median_prompt, output_contract=32, alpha=0.2):
        if policy not in ("model", "age_gate") or not 0 < alpha <= 1:
            raise ValueError("expected model/age_gate and 0 < alpha <= 1")
        if any(type(v) is not int or v < 1 for v in (maximum, token_budget, output_contract)):
            raise ValueError("positive integer capacity/budget/output contract required")
        if not math.isfinite(training_median_prompt) or training_median_prompt <= 0:
            raise ValueError("training-only median prompt length required")
        profile = json.loads(Path(profile_path).read_text())
        self.decode, self.mixed = {}, {}
        for kind, table in (("pure_decode", self.decode), ("mixed", self.mixed)):
            steps = [s for s in profile["steps"] if s["kind"] == kind]
            for width in sorted({s["decode_requests"] for s in steps}):
                values = sorted(s["engine_wall_ms"] for s in steps if s["decode_requests"] == width)
                table[width] = float(values[(len(values) - 1) // 2]) / 1000  # Profile's nearest-rank p50.
        self.mixed_default = float(profile["by_kind"]["mixed"]["engine_wall_ms"]["p50"]) / 1000
        if 1 not in self.decode or any(not math.isfinite(v) or v <= 0
                for v in [*self.decode.values(), *self.mixed.values(), self.mixed_default]):
            raise ValueError("invalid or missing training service medians")
        self.policy, self.maximum, self.budget, self.alpha = policy, maximum, token_budget, alpha
        self.age_guard_s = math.ceil(training_median_prompt / token_budget) * self.mixed_default + output_contract * self.decode[1]
        self.last_available_s, self.decisions, self.observations = None, [], []
        self.training = dict(profile_path=str(profile_path), source=profile.get("source"),
            median_prompt_tokens=training_median_prompt, output_contract=output_contract,
            token_budget=token_budget, frozen_age_guard_s=self.age_guard_s)

    def _estimate(self, table, width, default):
        if not table:
            return default, None
        key = min(table, key=lambda k: (abs(k - width), k))
        return table[key], key  # Nearest observed width, explicitly logged.

    def before_schedule(self, scheduler, rows, internal_to_source, now_s):
        if not math.isfinite(now_s) or (self.last_available_s is not None and self.last_available_s > now_s):
            raise ValueError("decision precedes completed observation availability")
        running = list(scheduler.running)
        waiting = list(scheduler.skipped_waiting) + list(scheduler.waiting)
        active, queued = len(running), len(waiting)
        decode = sum(r.num_computed_tokens >= r.num_prompt_tokens for r in running)
        prefilling = active - decode
        td, decode_key = self._estimate(self.decode, max(1, decode), self.decode[1])
        if not decode:
            td = 0.0
        tm, mixed_key = self._estimate(self.mixed, decode, self.mixed_default)
        fallback = None
        if getattr(scheduler, "num_waiting_for_streaming_input", 0) or any(
                getattr(r.status, "name", str(r.status)) != "WAITING"
                or getattr(r, "num_preemptions", 0) for r in waiting):
            fallback = "recovery_or_blocked_waiting_native_fallback"
        ages = []
        for request in waiting:
            source = internal_to_source.get(request.request_id)
            stamp = rows.get(source, {}).get("engine_add_return_s")
            if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp > now_s:
                fallback = fallback or "unknown_or_future_queue_timestamp_native_fallback"
            else:
                ages.append(now_s - stamp)
        queue = scheduler._select_waiting_queue_for_scheduling() if queued else None
        head = queue.peek_request() if queue else None
        remaining = max(0, head.num_prompt_tokens - head.num_computed_tokens) if head else 0
        chunks = math.ceil(remaining / self.budget)
        wait_cost, add_cost = queued * td, decode * max(tm - td, 0.0) * chunks
        oldest = max(ages) if ages else None
        admit, reason = False, "no_waiting"
        if fallback:
            reason = fallback
        elif queued and active < self.maximum:
            if active == 0:
                admit, reason = True, "empty_must_progress"
            elif oldest is not None and oldest >= self.age_guard_s:
                admit, reason = True, "shared_age_guard"
            elif self.policy == "age_gate":
                reason = "age_gate_hold"
            else:
                admit = wait_cost >= add_cost
                reason = "myopic_cost_admit" if admit else "myopic_cost_hold"
        elif queued:
            reason = "engine_capacity_full"
        target = self.maximum if fallback else min(self.maximum, max(1, active + int(admit)))
        limit = apply_nonpreemptive_limit(scheduler, target, self.maximum)
        decision = dict(index=len(self.decisions), policy=self.policy, decision_s=now_s,
            signal_available_s=self.last_available_s, active=active, decode_requests=decode,
            running_prefill_requests=prefilling, waiting_requests=queued, oldest_queue_age_s=oldest,
            head_request_id=head.request_id if head else None, next_prompt_remaining=remaining,
            estimated_prompt_chunks=chunks, token_budget=self.budget,
            T_decode_hat_s=td, T_mixed_hat_s=tm, decode_lookup_width=decode_key,
            mixed_lookup_width=mixed_key, waiting_one_step_cost_request_s=wait_cost,
            new_prefill_added_cost_request_s=add_cost, age_guard_s=self.age_guard_s,
            admit_one=admit, target_cap=limit, effective_scheduler_limit=limit, fallback=fallback, reason=reason,
            hold_is_pure_decode_assumption_applicable=prefilling == 0,
            estimate_scope="Myopic unequal-horizon estimate; new prompt gets full budget; not a counterfactual.")
        self.decisions.append(decision)
        return decision

    def observe(self, step_wall_s, pager_new_records, available_s):
        if not math.isfinite(step_wall_s) or step_wall_s <= 0 or not math.isfinite(available_s):
            raise ValueError("invalid completed-step observation")
        if self.last_available_s is not None and available_s < self.last_available_s:
            raise ValueError("observation availability moved backwards")
        self.last_available_s = available_s
        calls = [r for r in pager_new_records if r.get("measurement") and not r.get("validation_run")]
        update = dict(available_s=available_s, step_wall_s=step_wall_s, used=False,
                      decision_index=len(self.decisions) - 1)
        if calls:
            context = calls[0].get("context", {})
            metadata = context.get("rows") or []
            valid = (context.get("row_request_order_verified") is True and metadata
                and context.get("valid_row_start") == 0 and context.get("valid_row_stop") == len(metadata)
                and all(r.get("status") == "complete" and r.get("context") == context for r in calls)
                and all(type(r.get("computed_position")) is int and type(r.get("prompt_tokens")) is int
                        and isinstance(r.get("internal_request_id"), str) for r in metadata))
            if valid:
                decoded = {r["internal_request_id"] for r in metadata if r["computed_position"] >= r["prompt_tokens"]}
                prefill = sum(r["computed_position"] < r["prompt_tokens"] for r in metadata)
                width = len(decoded)
                if width:
                    table = self.mixed if prefill else self.decode
                    prior, _ = self._estimate(table, width, self.mixed_default)
                    table[width] = (1 - self.alpha) * prior + self.alpha * step_wall_s
                    update.update(used=True, kind="mixed" if prefill else "pure_decode", width=width,
                        prior_s=prior, updated_s=table[width], step_id=context.get("step_id"))
        self.observations.append(update)
        return update

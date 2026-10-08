"""Online calibration and runtime integration for legal MoE suffix cancellation.

Predicted savings are decision inputs, never substituted for episode timing.
All observation, planning, gathering, and sampler work remains inside capture.
"""
from collections import defaultdict
import re
import time

from suffix_policy import plan_suffix
from suffix_runtime import SuffixStepState, install_suffix_runtime


class SuffixController:
    def __init__(self, runner, runtime, *, mode, decision_layer=0):
        self.state = SuffixStepState()
        self.runtime = runtime
        self.mode = mode
        self.decision_layer = decision_layer
        self.num_layers = len(runtime.layers)
        if mode not in ("off", "fixed0", "fixed1", "conservative", "horizon"):
            raise ValueError("unknown suffix policy")
        if type(decision_layer) is not int or not 0 <= decision_layer < self.num_layers:
            raise ValueError("decision layer must index a loaded MoE layer")
        self.max_draft_len = runner.num_spec_tokens
        uninstall_runtime = install_suffix_runtime(runner, self.state)
        self.records = []
        self.observations = []
        self.success = defaultdict(int)
        self.trials = defaultdict(int)
        self.warmup_success = defaultdict(int)
        self.warmup_trials = defaultdict(int)
        self.acceptance_calibrations = []
        self.frozen_survival = None
        self.frozen_phase = None
        self.price_s = None
        self.copy_seconds_per_byte = None
        self.overhead_s = 0.0
        original = runner._bookkeeping_sync
        had_bookkeeping = "_bookkeeping_sync" in vars(runner)

        def bookkeeping(*args, **kwargs):
            result = original(*args, **kwargs)
            token_lists, req_ids = result[2], result[4]
            if req_ids != self.state.req_ids:
                raise ValueError("bookkeeping and suffix request order differ")
            accepted = []
            for tokens, keep in zip(token_lists, self.state.keep_drafts):
                if not tokens:
                    accepted.append(None)
                    continue
                count = len(tokens) - 1
                if not 0 <= count <= keep:
                    raise ValueError("sampler committed outside the kept prefix")
                accepted.append(count)
                # Truncation is censoring: never label an unverified tail failed.
                for j in range(1, keep + 1):
                    self.trials[j] += 1
                    self.success[j] += int(count >= j)
                    if not self.runtime.measurement:
                        self.warmup_trials[j] += 1
                        self.warmup_success[j] += int(count >= j)
            self.state.steps[-1]["accepted_before_stop_filter"] = accepted
            self.state.steps[-1]["sampler_token_counts_before_stop_filter"] = [len(t) for t in token_lists]
            return result

        runner._bookkeeping_sync = bookkeeping
        runtime.suffix_controller = self

        def uninstall():
            if getattr(runner, "_bookkeeping_sync", None) is bookkeeping:
                if had_bookkeeping:
                    runner._bookkeeping_sync = original
                else:
                    del runner._bookkeeping_sync
            if getattr(runtime, "suffix_controller", None) is self:
                del runtime.suffix_controller
            uninstall_runtime()

        self.uninstall = uninstall

    def _acceptance_survival(self):
        phase = self.runtime.context.get("phase")
        if self.runtime.measurement and (self.frozen_survival is None or self.frozen_phase != phase):
            values = [(self.warmup_success[j] + 1) / (self.warmup_trials[j] + 2)
                      for j in range(1, self.max_draft_len + 1)]
            for j in range(1, len(values)):
                values[j] = min(values[j], values[j - 1])
            self.frozen_survival = values
            self.frozen_phase = phase
            self.acceptance_calibrations.append(dict(
                phase=phase, step_id=self.state.step_id,
                warmup_success=dict(self.warmup_success), warmup_trials=dict(self.warmup_trials),
                survival=list(values), source="uncut_warmup_for_adaptive_modes",
                measurement_updates_used=False))
        if self.runtime.measurement:
            return list(self.frozen_survival)
        values = [(self.warmup_success[j] + 1) / (self.warmup_trials[j] + 2)
                  for j in range(1, self.max_draft_len + 1)]
        for j in range(1, len(values)):
            values[j] = min(values[j], values[j - 1])
        return values

    def select(self, rows, resident, layer_name, expert_bytes):
        state = self.state
        if not state.active or state.sampled:
            return None, None
        if getattr(self, "compact_active", False):
            if len(rows) != len(self.compact_original_indices):
                raise ValueError("compacted MoE rows lost their original-row mapping")
            # The single decision has already run; all current rows survive.
            return None, None
        # Freeze on the first measured MoE call, including prefill, before any
        # measured output can enter calibration. Later outcomes remain trace
        # observations only: route-dependent cutting must not bias this prior.
        survival = self._acceptance_survival()
        match = re.search(r"layers\.(\d+)", layer_name)
        if match is None:
            raise ValueError("unknown model layer name")
        layer_index = int(match[1])
        receipt = None
        if layer_index == self.decision_layer and any(state.eligible):
            started = time.perf_counter()
            positions = [i for i, eligible in enumerate(state.eligible) if eligible]
            ids = [state.req_ids[i] for i in positions]
            counts = {state.req_ids[i]: state.original_draft_counts[i] for i in positions}
            ready = self.price_s is not None and self.copy_seconds_per_byte is not None
            mode = self.mode if ready or self.mode in ("off", "fixed0", "fixed1") else "off"
            if not self.runtime.measurement and mode in ("conservative", "horizon"):
                mode = "off"
            fixed = int(mode[-1]) if mode.startswith("fixed") else None
            plan = plan_suffix(
                row_experts=rows, resident_experts=resident,
                row_starts={state.req_ids[i]: state.row_starts[i] for i in positions},
                original_draft_counts=counts,
                current_keep={state.req_ids[i]: state.keep_drafts[i] for i in positions},
                expert_bytes=expert_bytes,
                bandwidth_bytes_per_s=1 / self.copy_seconds_per_byte if ready else 1.0,
                acceptance_survival={rid: survival[:counts[rid]] for rid in ids},
                future_token_price_s=self.price_s or 0.0,
                overhead_s=self.overhead_s,
                mode="fixed" if fixed is not None else mode,
                remaining_layer_scale=self.num_layers - layer_index,
                fixed_keep=fixed)
            keep = [plan["keep_drafts"].get(rid, old)
                    for rid, old in zip(state.req_ids, state.keep_drafts)]
            state.set_keep_drafts(keep, layer=layer_name)
            elapsed = time.perf_counter() - started
            receipt = dict(plan["receipt"], step_id=state.step_id,
                           phase=self.runtime.context.get("phase"), layer=layer_name,
                           policy_mode=self.mode, calibration_ready=ready,
                           decision_host_s=elapsed, price_s=self.price_s,
                           copy_seconds_per_byte=self.copy_seconds_per_byte,
                           survival=survival, acceptance_frozen=self.runtime.measurement,
                           acceptance_source="warmup")
            self.records.append(receipt)
            self.overhead_s = max(self.overhead_s, elapsed)
        alive = state.alive_indices(len(rows))
        return (alive if len(alive) < len(rows) else None), receipt

    def observe_service(self, duration_s, committed_tokens, pager_records):
        """Called after native engine.step and host receipts, before next step."""
        load_s = load_bytes = 0
        # Match this engine call's groups by identity, rather than an index into
        # the retained event list. flush_records() clears that list, and warmup
        # capture may omit this callback; neither should mix old DMA into today.
        pending = {id(group): group for record in pager_records for group in record["groups"]
                   if group.get("weight_copy_bytes", 0) > 0}
        load_groups = len(pending)
        if pending:
            for group, (begin, end) in reversed(self.runtime.events):
                if id(group) not in pending:
                    continue
                # Native synchronous sample completed these default-stream events.
                load_s += begin.elapsed_time(end) / 1000
                load_bytes += group["weight_copy_bytes"]
                del pending[id(group)]
                if not pending:
                    break
        if pending:
            raise ValueError("current pager load groups lack matching CUDA events")
        pure_decode = (bool(pager_records) and self.state.active and all(
            c >= p for c, p in zip(self.state.computed_starts, self.state.prompt_tokens)))
        if pure_decode and committed_tokens:
            value = duration_s / committed_tokens
            self.price_s = value if self.price_s is None else .8 * self.price_s + .2 * value
        if load_bytes and load_s > 0:
            value = load_s / load_bytes
            self.copy_seconds_per_byte = value if self.copy_seconds_per_byte is None else (
                .8 * self.copy_seconds_per_byte + .2 * value)
        self.observations.append(dict(step_id=self.state.step_id,
            phase=self.runtime.context.get("phase"), duration_s=duration_s,
            committed_tokens=committed_tokens, pure_decode=pure_decode,
            observed_load_span_s=load_s, observed_load_bytes=load_bytes, observed_load_groups=load_groups,
            price_s=self.price_s, copy_seconds_per_byte=self.copy_seconds_per_byte))

    def report(self):
        return dict(mode=self.mode, decision_layer=self.decision_layer,
            state_steps=self.state.steps, cancellations=self.state.events,
            decisions=self.records, observations=self.observations,
            acceptance_calibrations=self.acceptance_calibrations,
            interpretation="Current-layer set savings are exact under one-load expert execution; horizon savings are predictions only. Load spans are a calibrated proxy, not proof of marginal critical-path time.")

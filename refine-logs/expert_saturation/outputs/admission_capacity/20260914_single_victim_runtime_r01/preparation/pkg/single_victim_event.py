"""One predeclared victim substitution with a fail-closed current-state match."""
from copy import deepcopy
from dataclasses import asdict, replace

from absence_rotation import AbsenceRotation


class SingleVictimEvent:
    """Observe one selector call; optionally replace its victim exactly once."""

    def __init__(self, contract, enabled, request_state_provider=None):
        if type(enabled) is not bool or not isinstance(contract, dict):
            raise ValueError("contract must be a dict and enabled must be bool")
        if request_state_provider is not None and not callable(request_state_provider):
            raise ValueError("request_state_provider must be callable")
        if (contract.get("schema_version") != 1
                or contract.get("status") != "CPU_QUALIFIED_EVENT_CONTRACT"):
            raise ValueError("unsupported or unqualified event contract")
        for key in ("source", "event", "source_ids", "expected"):
            if key not in contract:
                raise ValueError(f"event contract missing {key}")
        ids = contract["source_ids"]
        if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
                or any(type(rid) is not str or not rid for rid in ids)):
            raise ValueError("source_ids must be unique nonempty strings")
        if any(a != b and b.startswith(a + "-") for a in ids for b in ids):
            raise ValueError("source-id prefixes are ambiguous")
        event = contract["event"]
        if (type(event.get("step")) is not int or event["step"] < 0
                or any(event.get(k) not in ids for k in
                       ("target_source_id", "default_victim_source_id",
                        "replacement_source_id"))):
            raise ValueError("invalid event identity")
        self.contract, self.enabled = deepcopy(contract), enabled
        self.request_state_provider = request_state_provider
        self._made = False
        self._result = dict(schema_version=1, status="NOT_REACHED", enabled=enabled,
            event=deepcopy(event), applied_count=0, attempt_state=None, mismatches=[],
            original_proposal=None, returned_proposal=None, repeat_attempt_count=0,
            requires_native_forced_event_reconciliation=True)

    def _normalize(self, request_id):
        matches = [rid for rid in self.contract["source_ids"]
                   if request_id == rid or request_id.startswith(f"measured/{rid}-")]
        if len(matches) != 1:
            raise ValueError(f"request id has {len(matches)} source-prefix matches: {request_id}")
        return matches[0]

    def _map(self, values):
        if values is None:
            return None
        result = {}
        for request_id, value in values.items():
            source_id = self._normalize(request_id)
            if source_id in result:
                raise ValueError(f"duplicate normalized request id: {source_id}")
            result[source_id] = value
        return dict(sorted(result.items()))

    def _proposal(self, proposal):
        result = asdict(proposal)
        for key in ("resume_id", "victim_id"):
            if result[key] is not None:
                result[key] = self._normalize(result[key])
        return result

    def _attempt(self, tracker, prior, step, running, waiting, free, needed, released):
        request_states = None if self.request_state_provider is None else self._map(
            self.request_state_provider())
        fields = {"computed_tokens", "prompt_tokens", "output_tokens", "max_tokens",
                  "num_tokens", "owned_blocks"}
        if request_states is not None and any(set(row) != fields or
                any(type(value) is not int or value < 0 for value in row.values())
                for row in request_states.values()):
            raise ValueError("request_state_provider returned invalid fields")
        return dict(step=step, running_order=[self._normalize(r.request_id) for r in running],
            request_view_count=len(running), request_views=[dict(
                request_id=self._normalize(r.request_id),
                num_computed_tokens=r.num_computed_tokens,
                num_prompt_tokens=r.num_prompt_tokens,
                max_total_tokens=r.max_total_tokens,
                num_output_tokens=r.num_output_tokens) for r in running],
            waiting_order=[self._normalize(rid) for rid in waiting], free_blocks=free,
            blocks_needed=self._map(needed), released_blocks=self._map(released),
            request_states=request_states,
            tracker=dict(victim_order=tracker.victim_order,
                effective_victim_order=prior["effective_victim_order"],
                absent_since=self._map(prior["absent_since"]),
                absence_count=self._map(prior["absence_count"]),
                resident_since=self._map(prior["resident_since"]),
                last_swap_step=prior["last_swap_step"],
                applied_rotations=prior["applied_rotations"],
                decision_count=prior["decision_count"]))

    def factory(self, config, *, victim_order):
        if self._made:
            raise ValueError("SingleVictimEvent factory may be used once")
        self._made = True
        return _EventTracker(config, victim_order=victim_order, observer=self)

    def snapshot(self):
        return deepcopy(self._result)


class _EventTracker(AbsenceRotation):
    def __init__(self, *args, observer, **kwargs):
        super().__init__(*args, **kwargs)
        self._observer = observer

    def decide(self, step, running, waiting_ids, free_blocks, blocks_needed=None, *,
               released_blocks=None):
        observer = self._observer
        if step != observer.contract["event"]["step"]:
            return super().decide(step, running, waiting_ids, free_blocks, blocks_needed,
                                  released_blocks=released_blocks)
        prior = dict(absent_since=dict(self.absent_since),
                     absence_count=dict(self.absence_count),
                     resident_since=dict(self.resident_since),
                     last_swap_step=self.last_swap_step,
                     applied_rotations=self.applied_rotations,
                     decision_count=len(self.decisions),
                     effective_victim_order=self.effective_victim_order)
        original = super().decide(step, running, waiting_ids, free_blocks, blocks_needed,
                                  released_blocks=released_blocks)
        if observer._result["attempt_state"] is not None:
            observer._result["repeat_attempt_count"] += 1
            return original
        try:
            attempt = observer._attempt(self, prior, step, running, waiting_ids,
                                        free_blocks, blocks_needed, released_blocks)
            original_view = observer._proposal(original)
        except Exception as error:
            observer._result.update(status="UNMATCHED",
                attempt_state={"observation_error": f"{type(error).__name__}: {error}"})
            return original
        observer._result.update(attempt_state=attempt, original_proposal=original_view,
                                returned_proposal=deepcopy(original_view))
        mismatches = [dict(field=key, expected=value, actual=attempt.get(key))
                      for key, value in observer.contract["expected"].items()
                      if attempt.get(key) != value]
        event = observer.contract["event"]
        expected_pair = ("rotate", event["target_source_id"],
                         event["default_victim_source_id"])
        actual_pair = (original_view["action"], original_view["resume_id"],
                       original_view["victim_id"])
        if actual_pair != expected_pair:
            mismatches.append(dict(field="original_proposal_pair",
                                   expected=expected_pair, actual=actual_pair))
        if mismatches:
            observer._result.update(status="UNMATCHED", mismatches=mismatches)
            return original
        if not observer.enabled:
            observer._result["status"] = "MATCHED_DISABLED"
            return original
        replacements = [r for r in running
                        if observer._normalize(r.request_id) == event["replacement_source_id"]]
        target_need = (blocks_needed or {}).get(original.resume_id)
        funded = (released_blocks is not None and type(target_need) is int
                  and original.resume_id in (blocks_needed or {})
                  and len(replacements) == 1
                  and replacements[0].request_id in released_blocks
                  and free_blocks + released_blocks[replacements[0].request_id] >= target_need)
        guarded = (len(replacements) == 1
                   and replacements[0].progress < self.config.protect_progress_fraction
                   and self.absence_count.get(replacements[0].request_id, 0)
                       < self.config.max_absences_per_request
                   and step - self.resident_since.get(replacements[0].request_id, -10 ** 9)
                       >= self.config.min_residency_steps)
        if not funded or not guarded:
            observer._result.update(status="UNMATCHED", mismatches=[dict(
                field="replacement_guard_and_funding", expected=True,
                actual=dict(guarded=guarded, funded=funded))])
            return original
        returned = replace(original, victim_id=replacements[0].request_id)
        self.decisions[-1] = returned
        observer._result.update(status="APPLIED", applied_count=1,
                                returned_proposal=observer._proposal(returned))
        return returned

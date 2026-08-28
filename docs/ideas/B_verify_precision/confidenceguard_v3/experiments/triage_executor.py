"""Policy-independent same-state trajectory executor for TriageAudit v2."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from triage_policy import AuditState, TriagePolicyError, cvar
from triage_runtime import cache_sha256, execute_same_state_step, execute_single_action_step, per_step_kl, tensor_sha256


class ExecutorError(RuntimeError):
    pass


def execute_policy_trajectory(
    *,
    policy: str,
    initial_cache: Any,
    decode_tokens: torch.Tensor,
    reference_logits: torch.Tensor,
    high_forward: Callable[[torch.Tensor, Any], Any],
    low_forward: Callable[[torch.Tensor, Any], Any],
    discrepancy_threshold: float,
    period: int | None = None,
    phase: int | None = None,
    max_unaudited_steps: int = 8,
    lockout_following_steps: int = 3,
    collect_diagnostics: bool = True,
    fingerprint_final_cache: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if decode_tokens.ndim != 2 or decode_tokens.shape[0] != 1:
        raise ExecutorError("decode_tokens must have shape [1,steps]")
    if reference_logits.ndim != 2 or reference_logits.shape[0] != decode_tokens.shape[1]:
        raise ExecutorError("reference_logits must have shape [steps,vocab]")
    if discrepancy_threshold < 0 or not np.isfinite(discrepancy_threshold):
        raise ExecutorError("discrepancy threshold must be finite and non-negative")
    periodic = policy not in {"always_bf16", "always_low"}
    if periodic:
        if period is None or phase is None:
            raise ExecutorError("periodic policy requires period and phase")
        state = AuditState(
            period=period,
            phase=phase,
            max_unaudited_steps=max_unaudited_steps,
            lockout_following_steps=lockout_following_steps,
        )
    else:
        state = None
    cache = initial_cache
    step_rows: list[dict[str, object]] = []
    served_kl: list[float] = []
    dangerous_count = 0
    protected_dangerous = 0
    diagnostic_forward_calls = 0
    diagnostic_high_forward_calls = 0
    diagnostic_low_forward_calls = 0
    diagnostic_clone_events = 0
    simple_high_calls = 0
    simple_low_calls = 0
    for step_index in range(decode_tokens.shape[1]):
        token = decode_tokens[:, step_index : step_index + 1]
        if policy == "always_bf16":
            decision = "high"
        elif policy == "always_low":
            decision = "low"
        else:
            decision = state.decision(step_index)
        if decision == "audit":
            probe = execute_same_state_step(
                cache,
                token,
                high_forward=high_forward,
                low_forward=low_forward,
                served_action="low",
            )
            selected = state.record_audit(probe.discrepancy, discrepancy_threshold)
            probe = replace(probe, served_action=selected)
            served = probe.served
            discrepancy: float | None = probe.discrepancy
            candidate_calls = 2
            candidate_clones = 2
            diagnostic_calls = 0
            diagnostic_clones = 0
            diagnostic_high_calls = 0
            diagnostic_low_calls = 0
        elif decision in {"low", "lockout_high"} and state is not None:
            state.record_single(decision)
            selected = "high" if decision == "lockout_high" else "low"
            if collect_diagnostics:
                probe = execute_same_state_step(
                    cache,
                    token,
                    high_forward=high_forward,
                    low_forward=low_forward,
                    served_action=selected,
                )
                served = probe.served
                discrepancy = probe.discrepancy
            else:
                served = execute_single_action_step(
                    cache,
                    token,
                    forward=high_forward if selected == "high" else low_forward,
                )
                discrepancy = None
            candidate_calls = 1
            candidate_clones = 0
            diagnostic_calls = int(collect_diagnostics)
            diagnostic_clones = 2 * int(collect_diagnostics)
            diagnostic_high_calls = int(collect_diagnostics and selected == "low")
            diagnostic_low_calls = int(collect_diagnostics and selected == "high")
        else:
            selected = decision
            if collect_diagnostics:
                probe = execute_same_state_step(
                    cache,
                    token,
                    high_forward=high_forward,
                    low_forward=low_forward,
                    served_action=selected,
                )
                served = probe.served
                discrepancy = probe.discrepancy
            else:
                served = execute_single_action_step(
                    cache,
                    token,
                    forward=high_forward if selected == "high" else low_forward,
                )
                discrepancy = None
            candidate_calls = 1
            candidate_clones = 0
            diagnostic_calls = int(collect_diagnostics)
            diagnostic_clones = 2 * int(collect_diagnostics)
            diagnostic_high_calls = int(collect_diagnostics and selected == "low")
            diagnostic_low_calls = int(collect_diagnostics and selected == "high")
            simple_high_calls += int(selected == "high")
            simple_low_calls += int(selected == "low")
        diagnostic_forward_calls += diagnostic_calls
        diagnostic_high_forward_calls += diagnostic_high_calls
        diagnostic_low_forward_calls += diagnostic_low_calls
        diagnostic_clone_events += diagnostic_clones
        cache = served.cache
        logits = served.logits[:, -1, :]
        quality_kl = per_step_kl(reference_logits[step_index : step_index + 1], logits)
        served_kl.append(quality_kl)
        dangerous = discrepancy > discrepancy_threshold if discrepancy is not None else None
        dangerous_count += int(bool(dangerous))
        protected_dangerous += int(bool(dangerous) and selected == "high")
        step_rows.append(
            {
                "step": step_index,
                "decision": decision,
                "served_action": selected,
                "same_state_discrepancy": discrepancy,
                "dangerous": dangerous,
                "served_quality_kl": quality_kl,
                "served_logits_sha256": tensor_sha256(logits),
                "candidate_forward_calls": candidate_calls,
                "candidate_clone_events": candidate_clones,
                "diagnostic_forward_calls": diagnostic_calls,
                "diagnostic_high_forward_calls": diagnostic_high_calls,
                "diagnostic_low_forward_calls": diagnostic_low_calls,
                "diagnostic_clone_events": diagnostic_clones,
            }
        )
    counters = state.counters() if state is not None else {
        "audit_events": 0,
        "high_forward_calls": simple_high_calls,
        "low_forward_calls": simple_low_calls,
        "total_candidate_forward_calls": simple_high_calls + simple_low_calls,
        "cache_clone_events": 0,
        "served_high_steps": simple_high_calls,
        "served_low_steps": simple_low_calls,
        "lockout_steps": 0,
    }
    # Independent closure against per-step rows catches counter bugs in the
    # policy state machine or executor integration.
    if counters["total_candidate_forward_calls"] != sum(int(row["candidate_forward_calls"]) for row in step_rows):
        raise ExecutorError("candidate forward counter does not close")
    if counters["cache_clone_events"] != sum(int(row["candidate_clone_events"]) for row in step_rows):
        raise ExecutorError("candidate clone counter does not close")
    quality = np.asarray(served_kl, dtype=np.float64)
    summary: dict[str, object] = {
        "policy": policy,
        "decode_steps": int(len(quality)),
        "document_mean_kl": float(quality.mean()),
        "document_cvar90_kl": cvar(quality, 0.1),
        "document_p95_kl": float(np.quantile(quality, 0.95)),
        "dangerous_steps": dangerous_count,
        "dangerous_step_recall": (
            protected_dangerous / dangerous_count if dangerous_count else 1.0
        ) if collect_diagnostics else None,
        "threshold_violation_fraction": (
            (dangerous_count - protected_dangerous) / len(quality)
        ) if collect_diagnostics else None,
        "diagnostic_forward_calls": diagnostic_forward_calls,
        "diagnostic_high_forward_calls": diagnostic_high_forward_calls,
        "diagnostic_low_forward_calls": diagnostic_low_forward_calls,
        "diagnostic_clone_events": diagnostic_clone_events,
        "physical_high_forward_calls": counters["high_forward_calls"] + diagnostic_high_forward_calls,
        "physical_low_forward_calls": counters["low_forward_calls"] + diagnostic_low_forward_calls,
        "final_cache_sha256": cache_sha256(cache) if fingerprint_final_cache else None,
        **counters,
    }
    return summary, step_rows

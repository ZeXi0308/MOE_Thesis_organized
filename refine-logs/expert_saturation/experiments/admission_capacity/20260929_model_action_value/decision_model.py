"""CPU-only commit decision on A's recovery state semantics.

The only override is direct resume when the already selected target can be
funded without evicting the planned victim.  This is a simple-rule repair,
not a time predictor or an independent scheduling method.  Native allocation,
connector retirement, and output receipts must still validate the proposal.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from service_window_model import Request as WindowRequest
from staged_save_contract import Plan, RequestState, commit_reason


@dataclass(frozen=True)
class StateSnapshot:
    state_version: int
    schedule_step: int
    observed_at_ms: float
    plan: Plan
    target: RequestState | None
    victim: RequestState | None
    target_window: WindowRequest | None
    victim_window: WindowRequest | None
    free_gpu_blocks: int | None
    free_sequence_slots: int | None
    block_size_tokens: int | None
    running_order: tuple[str, ...] | None
    waiting_order: tuple[str, ...] | None
    request_identity_stable: bool | None
    all_running_ownership_valid: bool | None
    # The native pool must supply current physical IDs and reference counts.
    victim_physical_ids: tuple[int, ...] | None
    victim_refcounts: tuple[int, ...] | None
    queue_clear: bool | None
    connector_retire_safe: bool | None
    target_transfer_clear: bool | None
    pending_load_ids: tuple[str, ...] | None
    # An accepted prepare may already have registered or completed a store.
    # Only *additional* host staging is tested here; old work is never erased.
    host_used_bytes: int | None = None
    host_budget_bytes: int | None = None
    new_host_staging_bytes: int | None = None


@dataclass(frozen=True)
class CostBoundary:
    target_need_blocks: int | None
    free_after_direct_blocks: int | None
    planned_victim_held_blocks: int | None
    planned_victim_saved_prefix_tokens: int
    target_output_age_ms: float | None
    victim_output_age_ms: float | None
    exposed_marginal_ms: None = None  # No calibrated action-specific critical path.
    future_eos_distance: None = None


@dataclass(frozen=True)
class Proposal:
    action: str  # DIRECT_RESUME | KEEP_PLANNED_SWAP | CANCEL_PLAN
    reason: str
    target_id: str
    victim_ids: tuple[str, ...]
    snapshot: StateSnapshot
    cost: CostBoundary


def _window_matches(native: RequestState, window: WindowRequest) -> bool:
    return (window.name == native.request_id
            and window.history_tokens == native.prompt + native.output
            and window.history_tokens > 0
            and window.computed_tokens == native.computed
            and 0 <= window.computed_tokens <= window.history_tokens
            and window.gpu_blocks == len(native.blocks)
            and window.computed_tokens <= 16 * window.gpu_blocks
            and isfinite(window.output_age_ms) and window.output_age_ms >= 0
            and not window.waiting_for_remote_kv
            and (window.remaining_output_cap is None
                 or window.remaining_output_cap == native.max_output - native.output > 0))


def propose(s: StateSnapshot) -> Proposal:
    """Propose an action from present state; never infer future output or time.

    KEEP leaves the qualified old commit to the native adapter.  CANCEL means
    the old intent itself became stale and must not be committed.  A direct
    proposal still needs an immediate native validation at the commit boundary.
    """
    victim_id = s.plan.victim.request_id
    target_id = s.plan.target.request_id
    need = s.target.remaining_blocks if s.target is not None else None
    free_after = (s.free_gpu_blocks - need if s.free_gpu_blocks is not None
                  and need is not None else None)
    boundary = CostBoundary(
        target_need_blocks=need,
        free_after_direct_blocks=free_after,
        planned_victim_held_blocks=len(s.victim.blocks) if s.victim else None,
        planned_victim_saved_prefix_tokens=s.plan.saved_tokens,
        target_output_age_ms=s.target_window.output_age_ms if s.target_window else None,
        victim_output_age_ms=s.victim_window.output_age_ms if s.victim_window else None,
    )

    def result(action: str, reason: str) -> Proposal:
        return Proposal(action, reason, target_id,
                        (victim_id,) if action == 'KEEP_PLANNED_SWAP' else (), s, boundary)

    if s.free_gpu_blocks is None or s.free_gpu_blocks < 0:
        return result('CANCEL_PLAN', 'UNKNOWN_OR_INVALID_GPU_FREE')
    if not isfinite(s.observed_at_ms) or s.observed_at_ms < 0:
        return result('CANCEL_PLAN', 'INVALID_OBSERVATION_TIME')
    reason = commit_reason(s.plan, s.schedule_step, s.victim, s.target,
                           s.free_gpu_blocks, None, save_enabled=False)
    if reason != 'READY':
        return result('CANCEL_PLAN', reason)
    if s.victim is None or s.target is None:  # also checked by commit_reason
        return result('CANCEL_PLAN', 'MISSING_REQUEST')
    if s.victim.output >= s.victim.max_output or s.target.output >= s.target.max_output:
        return result('CANCEL_PLAN', 'REQUEST_AT_DECLARED_CAP')
    if s.target_window is None or s.victim_window is None:
        return result('KEEP_PLANNED_SWAP', 'UNKNOWN_WINDOW_STATE')
    if not _window_matches(s.target, s.target_window) or not _window_matches(s.victim, s.victim_window):
        return result('KEEP_PLANNED_SWAP', 'WINDOW_STATE_MISMATCH_OR_PENDING_LOAD')
    if s.target_window.computed_tokens != 0 or s.target_window.gpu_blocks != 0:
        return result('KEEP_PLANNED_SWAP', 'TARGET_ALREADY_OWNS_GPU_KV')
    if s.block_size_tokens != 16:
        return result('KEEP_PLANNED_SWAP', 'UNQUALIFIED_BLOCK_SIZE')
    if (s.running_order is None or s.waiting_order is None
            or s.running_order.count(victim_id) != 1
            or target_id in s.running_order
            or s.waiting_order.count(target_id) != 1):
        return result('KEEP_PLANNED_SWAP', 'UNKNOWN_OR_CHANGED_QUEUE_ORDER')
    if s.request_identity_stable is not True or s.all_running_ownership_valid is not True:
        return result('KEEP_PLANNED_SWAP', 'IDENTITY_OR_POOL_OWNERSHIP_UNQUALIFIED')
    if s.pending_load_ids is None or target_id in s.pending_load_ids or victim_id in s.pending_load_ids:
        return result('KEEP_PLANNED_SWAP', 'UNKNOWN_OR_PENDING_NATIVE_LOAD')
    if (s.queue_clear is not True or s.connector_retire_safe is not True
            or s.target_transfer_clear is not True):
        return result('KEEP_PLANNED_SWAP', 'QUEUE_OR_CONNECTOR_UNQUALIFIED')
    if s.free_sequence_slots is None or s.free_sequence_slots < 1:
        return result('KEEP_PLANNED_SWAP', 'NO_VERIFIED_SEQUENCE_SLOT')
    if s.victim_physical_ids is None or s.victim_refcounts is None:
        return result('KEEP_PLANNED_SWAP', 'UNKNOWN_PHYSICAL_OWNERSHIP')
    if (s.victim_physical_ids != s.victim.blocks
            or len(set(s.victim_physical_ids)) != len(s.victim_physical_ids)
            or len(s.victim_refcounts) != len(s.victim_physical_ids)
            or any(ref != 1 for ref in s.victim_refcounts)):
        return result('KEEP_PLANNED_SWAP', 'SHARED_OR_CHANGED_BLOCK_OWNERSHIP')
    if s.new_host_staging_bytes is None or s.new_host_staging_bytes < 0:
        return result('KEEP_PLANNED_SWAP', 'UNKNOWN_NEW_HOST_STAGING')
    if s.new_host_staging_bytes:
        if (s.host_used_bytes is None or s.host_budget_bytes is None
                or min(s.host_used_bytes, s.host_budget_bytes) < 0
                or s.host_used_bytes + s.new_host_staging_bytes > s.host_budget_bytes):
            return result('KEEP_PLANNED_SWAP', 'HOST_STAGING_UNFUNDED')
    # The native allocator checks actual legal allocation again.  We do not
    # reserve every running request's future growth or assume an EOS release.
    if s.free_gpu_blocks < need:
        return result('KEEP_PLANNED_SWAP', 'TARGET_NEEDS_PLANNED_VICTIM')
    return result('DIRECT_RESUME', 'CURRENT_FREE_AND_SLOT_FUND_TARGET')


def validate_for_commit(proposal: Proposal, current: StateSnapshot) -> Proposal:
    """Reject changed versions, owners, queues, or resources before native commit."""
    def discrete(snapshot: StateSnapshot) -> StateSnapshot:
        # Clock/age advances without a resource or request transition.  An
        # actual output receipt must also change native output counters.
        target = (replace(snapshot.target_window, output_age_ms=0)
                  if snapshot.target_window is not None else None)
        victim = (replace(snapshot.victim_window, output_age_ms=0)
                  if snapshot.victim_window is not None else None)
        return replace(snapshot, observed_at_ms=0,
                       target_window=target, victim_window=victim)

    if discrete(current) != discrete(proposal.snapshot):
        return Proposal('CANCEL_PLAN', 'STALE_SNAPSHOT', proposal.target_id,
                        (), current,
                        propose(current).cost)
    fresh = propose(current)
    return fresh if fresh.action == proposal.action else Proposal(
        'CANCEL_PLAN', 'DECISION_CHANGED', proposal.target_id,
        (), current, fresh.cost)


def count_only_prediction(free: int, need: int, victim_blocks: int) -> dict:
    """A partial old-log certificate; missing slot/connector data stay unknown."""
    if min(free, need, victim_blocks) < 0:
        raise ValueError('negative block count')
    return dict(free=free, native_need=need, planned_victim_blocks=victim_blocks,
                direct_kv_funded=free >= need, free_after_direct=free - need,
                actual_direct_legal='UNKNOWN', exposed_time_saving_ms=None,
                full_request_gain='UNRUN')

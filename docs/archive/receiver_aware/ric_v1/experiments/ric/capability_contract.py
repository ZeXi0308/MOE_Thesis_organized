"""Frozen, outcome-independent RIC capability-probe execution contract."""

from __future__ import annotations

from typing import Any, Mapping


EXECUTION_ORDER_RULE = "config_trial_mod_4_orthogonal_counterbalance"

_RELEASE_ORDERS = {
    "early_then_barrier": ("streaming", "full_layer_barrier"),
    "barrier_then_early": ("full_layer_barrier", "streaming"),
}
_POLICY_ORDERS = {
    "baseline_then_candidate": (
        "baseline_nonclosing_first",
        "candidate_closing_first",
    ),
    "candidate_then_baseline": (
        "candidate_closing_first",
        "baseline_nonclosing_first",
    ),
}


class CapabilityContractError(ValueError):
    """The frozen counterbalance table is missing, malformed, or drifted."""


def capability_execution_order(
    config: Mapping[str, Any], trial: int
) -> tuple[tuple[str, str], ...]:
    """Return the exact outer-release/inner-policy order for one trial.

    The function deliberately consumes the frozen config instead of deriving a
    parity shortcut.  Validating the complete four-row table on every call
    prevents a producer and consumer from sharing the same incomplete schedule.
    """

    if type(trial) is not int or trial < 0:
        raise CapabilityContractError("capability trial must be a non-negative int")
    try:
        schedule = config["capability_probes"]["counterbalance_schedule"]
        period = schedule["period"]
        rows = schedule["trial_mod_4"]
    except (KeyError, TypeError) as exc:
        raise CapabilityContractError("counterbalance schedule is missing") from exc
    if type(period) is not int or period != 4 or not isinstance(rows, Mapping):
        raise CapabilityContractError("counterbalance period must be exactly four")
    if set(rows) != {"0", "1", "2", "3"}:
        raise CapabilityContractError("counterbalance table must contain rows 0..3")

    expected = {
        "0": ("early_then_barrier", "baseline_then_candidate"),
        "1": ("barrier_then_early", "candidate_then_baseline"),
        "2": ("early_then_barrier", "candidate_then_baseline"),
        "3": ("barrier_then_early", "baseline_then_candidate"),
    }
    normalized: dict[str, tuple[str, str]] = {}
    for key, wanted in expected.items():
        raw = rows.get(key)
        if not isinstance(raw, list) or len(raw) != 2:
            raise CapabilityContractError(
                f"counterbalance row {key} must contain two order labels"
            )
        normalized[key] = (str(raw[0]), str(raw[1]))
        if normalized[key] != wanted:
            raise CapabilityContractError(
                f"counterbalance row {key} differs from the frozen mod-4 table"
            )

    release_label, policy_label = normalized[str(trial % period)]
    releases = _RELEASE_ORDERS[release_label]
    policies = _POLICY_ORDERS[policy_label]
    return tuple((release, policy) for release in releases for policy in policies)

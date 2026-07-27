from __future__ import annotations

from pathlib import Path

import numpy as np

from analyze_r0a import analyze_rows, paired_bootstrap
from r0a_artifacts import load_config


CONFIG_PATH = Path(__file__).parent / "configs/r0a_5090_v1.json"


def _rows(*, free: float, full: float, flips: int, non_tie_flips: int) -> list[dict]:
    rows = []
    for index in range(32):
        digest = f"{index:064x}"
        common = {"text_sha256": digest, "prompt_length": 2048, "completed_steps": 32}
        rows.extend(
            [
                {
                    **common,
                    "target": "k_only",
                    "arm": "free",
                    "mean_kl": free + index * 1e-7,
                    "route_metrics": {
                        "set_flip_count": flips,
                        "non_tie_set_flip_count": non_tie_flips,
                        "non_tie_cell_count": 512,
                        "route_cell_count": 512,
                    },
                },
                {**common, "target": "k_only", "arm": "set_locked", "mean_kl": full + 1e-5},
                {**common, "target": "k_only", "arm": "fully_locked", "mean_kl": full},
                {**common, "target": "identity", "arm": "identity_free", "mean_kl": 1e-9},
            ]
        )
    return rows


def test_paired_bootstrap_uses_ratio_of_means() -> None:
    free = np.asarray([1.0, 9.0])
    router = np.asarray([0.9, 0.9])
    result = paired_bootstrap(free, router, replicates=100, seed=1, confidence=0.95)
    assert abs(result["router_share_ratio_of_means"] - 0.18) < 1e-12
    assert result["router_share_ratio_of_means"] != np.mean(router / free)


def test_synthetic_primary_pass() -> None:
    config = load_config(CONFIG_PATH)
    decision = analyze_rows(
        config,
        _rows(free=5e-4, full=1e-4, flips=8, non_tie_flips=8),
        {"status": "PASS"},
        require_complete=False,
    )
    assert decision["decision_code"] == "PASS_R0A_ROUTE_MEDIATED_KV_EFFECT_R0B_ONLY"


def test_total_effect_no_go_has_priority() -> None:
    config = load_config(CONFIG_PATH)
    decision = analyze_rows(
        config,
        _rows(free=5e-5, full=1e-5, flips=8, non_tie_flips=8),
        {"status": "PASS"},
        require_complete=False,
    )
    assert decision["decision_code"] == "NO_GO_TOTAL_KV_EFFECT_TOO_SMALL"


def test_rare_route_change_no_go() -> None:
    config = load_config(CONFIG_PATH)
    decision = analyze_rows(
        config,
        _rows(free=5e-4, full=1e-4, flips=4, non_tie_flips=4),
        {"status": "PASS"},
        require_complete=False,
    )
    assert decision["decision_code"] == "NO_GO_ROUTESET_CHANGE_TOO_RARE"


def test_integrity_failure_overrides_scientific_pass() -> None:
    config = load_config(CONFIG_PATH)
    decision = analyze_rows(
        config,
        _rows(free=5e-4, full=1e-4, flips=8, non_tie_flips=8),
        {"status": "FAIL", "decision_code": "INVALID_QUANTIZATION_PATH"},
        require_complete=False,
    )
    assert decision["decision_code"] == "INVALID_QUANTIZATION_PATH"


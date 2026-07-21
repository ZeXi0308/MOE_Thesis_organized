"""Deterministic toy checks for ``run_masscover_ep_p0.py``.

Run directly; pytest is intentionally not required:

    ./.venv/bin/python experiments/idea_a_mac/test_masscover_ep_p0.py
"""


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

from __future__ import annotations

import numpy as np
import pandas as pd

from run_masscover_ep_p0 import (
    build_risk_matrix,
    c1_cross_domain_fraction,
    cvar,
    cvar_greedy_order,
    residual_after,
    static_order,
    summarize_residual,
)


ASSERTIONS = 0


def check(condition: bool, message: str) -> None:
    global ASSERTIONS
    assert condition, message
    ASSERTIONS += 1


def toy_routes() -> pd.DataFrame:
    rows = [
        # token 0: domain 0 dominates
        (0, 0, 0, 1, 0, 0.90, 0),
        (0, 0, 0, 2, 2, 0.10, 0),
        # token 1: same critical expert 0
        (0, 0, 1, 1, 0, 0.80, 0),
        (0, 0, 1, 2, 3, 0.20, 0),
        # token 2: balanced
        (1, 0, 0, 1, 1, 0.50, 1),
        (1, 0, 0, 2, 2, 0.50, 1),
    ]
    table = pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "layer",
            "token_position",
            "rank",
            "expert_id",
            "gate_share",
            "home_rank",
        ],
    )
    table["token_index"] = [0, 0, 1, 1, 2, 2]
    return table


def test_risk_matrix_and_selection() -> None:
    routes = toy_routes()
    mapping = np.array([0, 0, 1, 1], dtype=np.int16)
    matrix = build_risk_matrix(
        routes,
        mapping,
        num_experts=4,
        gpus_per_failure_domain=1,
    )
    check(matrix.matrix_csr.shape == (6, 4), "three tokens times two domains")
    check(
        np.allclose(matrix.initial_residual, [0.9, 0.1, 0.8, 0.2, 0.5, 0.5]),
        "failure-domain missing mass is hand-counted",
    )
    check(np.isclose(cvar(matrix.initial_residual, 0.5), (0.9 + 0.8 + 0.5 + 0.5) / 4), "CVaR includes ties")

    gate_order = static_order(matrix, "gate_mass")
    check(gate_order[0] == 0, "gate-mass baseline selects the dominant expert")
    greedy = cvar_greedy_order(matrix, max_candidates=2, alpha=0.5)
    check(greedy[0] == 0, "CVaR greedy selects expert 0 first")
    residual = residual_after(matrix, greedy[:1])
    check(
        np.allclose(residual, [0.0, 0.1, 0.0, 0.2, 0.5, 0.5]),
        "selected shadow removes only its failed-domain contribution",
    )
    summary = summarize_residual(matrix, residual, alpha=0.95)
    check(np.isclose(summary["mean_uncovered_mass"], 1.3 / 6), "summary mean closes")
    check(summary["fraction_fully_covered"] == 2 / 6, "coverage fraction closes")


def test_c1_cross_domain_fraction() -> None:
    routes = toy_routes()
    mapping = np.array([0, 0, 1, 1], dtype=np.int16)
    fraction = c1_cross_domain_fraction(
        routes,
        mapping,
        gpus_per_failure_domain=1,
    )
    # Each token visits both owner ranks.  One record per token is local and
    # the other is cross-domain.
    check(np.isclose(fraction, 3 / 6), "C1 cross-domain accounting is hand-counted")


if __name__ == "__main__":
    test_risk_matrix_and_selection()
    test_c1_cross_domain_fraction()
    print(f"PASS: {ASSERTIONS} assertions")

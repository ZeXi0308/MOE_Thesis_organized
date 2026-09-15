"""Dependency-light toy tests for the frozen RouteFidelity P0-B runner."""

from __future__ import annotations

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

import numpy as np

from route_fidelity_p0b_core import C0_EXPANDED, C1_UNIQUE_OWNER, RouteIR
from run_route_fidelity_p0b import (
    bootstrap_seed_median_regret,
    c0_from_degree,
    coactivation_hill_climb,
    frequency_lpt,
    holm_adjust,
    logical_cost_matrix,
    packed_size_accounting,
    random_balanced,
    request_expert_degree,
    validate_balanced,
)


def make_route() -> RouteIR:
    return RouteIR(
        experts=np.asarray(
            [
                [0, 1],
                [0, 1],
                [2, 3],
                [2, 3],
                [2, 3],
                [0, 1],
            ],
            dtype=np.int32,
        ),
        request_id=np.asarray([10, 10, 10, 20, 20, 20]),
        layer_id=np.zeros(6, dtype=np.int32),
        token_position=np.asarray([0, 1, 2, 0, 1, 2]),
        home_rank=np.asarray([0, 0, 0, 4, 4, 4]),
        num_experts=4,
        top_k=2,
        home_rank_source="toy-observed",
    )


def main() -> None:
    route = make_route()
    mappings = [np.asarray([0, 0, 4, 4]), np.asarray([0, 4, 0, 4])]
    requests, c0 = logical_cost_matrix(route, mappings, contract=C0_EXPANDED)
    _, c1 = logical_cost_matrix(route, mappings, contract=C1_UNIQUE_OWNER)
    assert requests.tolist() == [10, 20]
    assert c0.tolist() == [[2, 3], [2, 3]]
    assert c1.tolist() == [[1, 3], [1, 3]]

    degree = request_expert_degree(route, requests)
    from_degree = c0_from_degree(degree, mappings, np.asarray([0, 4]))
    assert np.array_equal(from_degree, c0)

    random_a = random_balanced(8, 4, 2026072000)
    random_b = random_balanced(8, 4, 2026072000)
    assert np.array_equal(random_a, random_b)
    validate_balanced(random_a, 8, 4)
    lpt = frequency_lpt(route, 2)
    validate_balanced(lpt, 4, 2)
    climbed, metadata = coactivation_hill_climb(
        route, lpt, iterations=100, seed=7
    )
    validate_balanced(climbed, 4, 2)
    assert metadata["accepted_swaps"] >= 0

    exact = np.asarray([[1.0, 2.0], [1.0, 2.0]])
    synthetic = np.asarray(
        [
            [[2.0, 1.0], [2.0, 1.0]],
            [[2.0, 1.0], [2.0, 1.0]],
        ]
    )
    first = bootstrap_seed_median_regret(
        exact, synthetic, np.asarray([1.0, 1.0]), resamples=100, seed=9
    )
    second = bootstrap_seed_median_regret(
        exact, synthetic, np.asarray([1.0, 1.0]), resamples=100, seed=9
    )
    assert np.array_equal(first, second)
    assert np.allclose(first, 1.0)
    assert holm_adjust({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}

    size = packed_size_accounting(route, 2)
    assert size["expert_id_bits"] == 2
    assert size["s3_raw_packed_bytes"] > 0
    assert size["s4_raw_packed_bytes"] > 0
    print("run_route_fidelity_p0b: all toy assertions passed")


if __name__ == "__main__":
    main()

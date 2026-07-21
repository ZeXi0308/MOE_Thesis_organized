"""Manual toy assertions for :mod:`route_fidelity_p0b_core`.

Run directly; pytest is intentionally not required:

    ./.venv/bin/python experiments/idea_a_mac/test_route_fidelity_p0b_core.py
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

import csv
import math
from pathlib import Path
import tempfile

import numpy as np

from route_fidelity_p0b_core import (
    C0_EXPANDED,
    C1_UNIQUE_OWNER,
    RouteIR,
    balanced_placements,
    canonical_s2_size_bytes,
    canonical_s3_size_bytes,
    canonical_s4_size_bytes,
    cluster_bootstrap_ranking,
    deterministic_home_ranks,
    kendall_tau_b,
    load_route_csv,
    lower_cross_domain_cost,
    placement_hash,
    placement_manifest_hash,
    s1r_exact_degree,
    selection_regret,
)


ASSERTIONS = 0


def check(condition: bool, message: str) -> None:
    global ASSERTIONS
    assert condition, message
    ASSERTIONS += 1


def assert_raises(error_type: type[BaseException], function, *args, **kwargs) -> None:
    global ASSERTIONS
    try:
        function(*args, **kwargs)
    except error_type:
        ASSERTIONS += 1
        return
    raise AssertionError(f"expected {error_type.__name__}")


def make_cost_route() -> RouteIR:
    return RouteIR(
        experts=np.array([[0, 1], [2, 3], [0, 2], [1, 3]]),
        request_id=np.array([10, 10, 20, 20]),
        layer_id=np.array([0, 0, 0, 0]),
        token_position=np.array([0, 1, 0, 1]),
        home_rank=np.array([0, 1, 0, 1]),
        num_experts=4,
        top_k=2,
        home_rank_source="toy-observed",
    )


def make_shuffle_route() -> RouteIR:
    base = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 0],
            [0, 2],
            [3, 5],
        ],
        dtype=np.int32,
    )
    experts = np.concatenate([base, np.roll(base, 1, axis=0), np.roll(base, 2, axis=0)])
    return RouteIR(
        experts=experts,
        request_id=np.repeat([100, 200, 200], len(base)),
        layer_id=np.repeat([0, 0, 1], len(base)),
        token_position=np.tile(np.arange(len(base)), 3),
        home_rank=np.tile(np.arange(len(base)) % 4, 3),
        num_experts=6,
        top_k=2,
        home_rank_source="toy-observed",
    )


def test_route_ir_and_loader() -> None:
    route = make_cost_route()
    check(route.token_count == 4, "RouteIR token count")
    check(route.experts.dtype == np.int32, "RouteIR normalizes expert dtype")
    check(not route.experts.flags.writeable, "RouteIR arrays must be immutable")
    assert_raises(
        ValueError,
        RouteIR,
        experts=np.array([[0, 0]]),
        request_id=np.array([0]),
        layer_id=np.array([0]),
        token_position=np.array([0]),
        home_rank=np.array([0]),
        num_experts=2,
        top_k=2,
    )

    with tempfile.TemporaryDirectory() as directory:
        observed_path = Path(directory) / "observed.csv"
        with observed_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["sample_id", "layer", "token_position", "rank", "expert_id", "home_rank"]
            )
            # Intentionally shuffled input order: loader must canonicalize tokens/ranks.
            writer.writerow([8, 0, 1, 2, 3, 1])
            writer.writerow([7, 0, 0, 2, 2, 0])
            writer.writerow([8, 0, 1, 1, 1, 1])
            writer.writerow([7, 0, 0, 1, 0, 0])
        loaded = load_route_csv(observed_path, num_experts=4, top_k=2)
        check(loaded.request_id.tolist() == [7, 8], "loader sorts token keys")
        check(loaded.experts.tolist() == [[0, 2], [1, 3]], "loader preserves rank order")
        check(loaded.home_rank.tolist() == [0, 1], "loader retains observed home ranks")
        check(loaded.home_rank_source == "csv:home_rank", "loader labels observed metadata")
        assert_raises(
            ValueError,
            load_route_csv,
            observed_path,
            num_experts=3,
            top_k=2,
        )
        assert_raises(
            ValueError,
            load_route_csv,
            observed_path,
            num_experts=4,
            top_k=2,
            synthetic_home_ep_size=2,
        )

        synthetic_path = Path(directory) / "synthetic.csv"
        with synthetic_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "layer", "token_position", "rank", "expert_id"])
            writer.writerow([2, 1, 3, 1, 0])
            writer.writerow([2, 1, 3, 2, 1])
        assert_raises(
            ValueError,
            load_route_csv,
            synthetic_path,
            num_experts=4,
            top_k=2,
        )
        synthetic = load_route_csv(
            synthetic_path,
            num_experts=4,
            top_k=2,
            synthetic_home_ep_size=4,
        )
        check(synthetic.home_rank.tolist() == [2], "synthetic home policy is frozen modulo")
        check(
            synthetic.home_rank_source == "synthetic_mod:ep_size=4",
            "synthetic home metadata is visibly labeled",
        )

    homes = deterministic_home_ranks(
        np.array([0, 1]), np.array([1, 1]), np.array([2, 3]), ep_size=4
    )
    check(homes.tolist() == [3, 1], "deterministic home-rank formula")


def test_c0_c1_cross_domain_costs() -> None:
    route = make_cost_route()
    placement = np.array([0, 0, 1, 2])
    rank_to_domain = np.array([0, 0, 1])
    c0 = lower_cross_domain_cost(
        route, placement, rank_to_domain, contract=C0_EXPANDED
    )
    c1 = lower_cross_domain_cost(
        route, placement, rank_to_domain, contract=C1_UNIQUE_OWNER
    )
    check(c0.request_id.tolist() == [10, 20], "C0 is reported per request")
    check(c0.logical_records.tolist() == [4, 4], "C0 emits one record per routed pair")
    check(c0.cross_domain_records.tolist() == [1, 1], "C0 hand-counted domains")
    check(c0.local_records.tolist() == [3, 3], "C0 local records close the accounting")
    check(c1.logical_records.tolist() == [3, 4], "C1 deduplicates owners per token")
    check(c1.cross_domain_records.tolist() == [1, 1], "C1 hand-counted domains")
    check(c1.total_logical_records < c0.total_logical_records, "C1 cannot expand this toy")
    assert_raises(
        ValueError,
        lower_cross_domain_cost,
        route,
        placement,
        rank_to_domain,
        contract="C2_NOT_IMPLEMENTED",
    )


def test_request_layer_s1r_certificate() -> None:
    route = make_shuffle_route()
    result = s1r_exact_degree(route, seed=17, swap_multiplier=16)
    certificate = result.certificate
    check(certificate.group_count == 3, "S1-R groups by request and layer")
    check(certificate.accepted_swaps > 0, "toy graph admits rewiring")
    check(certificate.duplicate_rows == 0, "S1-R retains simple token routes")
    check(certificate.degree_tv_max == 0.0, "S1-R exact degree certificate")
    check(0.0 <= certificate.token_jaccard_mean < 1.0, "S1-R reports route overlap")
    check(0.0 <= certificate.pair_distance_mean <= 1.0, "S1-R reports pair TV")
    check(
        np.array_equal(result.route.request_id, route.request_id)
        and np.array_equal(result.route.home_rank, route.home_rank),
        "S1-R preserves exogenous metadata",
    )
    for request, layer in {(int(r), int(l)) for r, l in zip(route.request_id, route.layer_id)}:
        mask = (route.request_id == request) & (route.layer_id == layer)
        before = np.bincount(route.experts[mask].reshape(-1), minlength=route.num_experts)
        after = np.bincount(result.route.experts[mask].reshape(-1), minlength=route.num_experts)
        check(np.array_equal(before, after), f"degree invariant for group {(request, layer)}")


def test_canonical_sizes_and_placements() -> None:
    route = make_shuffle_route()
    windows = route.token_position // 3
    s2_first = canonical_s2_size_bytes(route)
    s2_second = canonical_s2_size_bytes(route)
    s3 = canonical_s3_size_bytes(route, windows)
    s4_first = canonical_s4_size_bytes(route)
    s4_second = canonical_s4_size_bytes(route)
    check(s2_first == s2_second and s2_first > 0, "S2 canonical size is deterministic")
    check(s3 > s2_first, "request-window S3 carries more keys than request-layer S2")
    check(s4_first == s4_second and s4_first > 0, "S4 full-route size is deterministic")
    assert_raises(ValueError, canonical_s3_size_bytes, route, windows[:-1])

    placements = balanced_placements(8, 4, random_count=3, seed=9)
    check(len(placements) == 5, "placement pool includes two controls and random placements")
    for name, mapping in placements.items():
        counts = np.bincount(mapping, minlength=4)
        check(counts.tolist() == [2, 2, 2, 2], f"balanced placement {name}")
    check(
        placement_hash(placements["contiguous"])
        == placement_hash(placements["contiguous"].astype(np.int64)),
        "placement hash is dtype-independent",
    )
    check(
        placement_hash(placements["contiguous"])
        != placement_hash(placements["round_robin"]),
        "different placements have different hashes in toy",
    )
    check(
        placement_manifest_hash(placements)
        == placement_manifest_hash(dict(reversed(list(placements.items())))),
        "manifest hash is independent of mapping insertion order",
    )
    assert_raises(ValueError, balanced_placements, 10, 4)


def test_tau_regret_and_cluster_bootstrap() -> None:
    check(kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0, "tau-b perfect ordering")
    check(kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0, "tau-b reverse ordering")
    expected_tied_tau = 2.0 / math.sqrt(6.0)
    check(
        math.isclose(kendall_tau_b([1, 1, 2], [1, 2, 3]), expected_tied_tau),
        "tau-b tie correction",
    )
    regret = selection_regret(
        [10.0, 12.0, 11.0],
        [3.0, 1.0, 2.0],
        config_ids=["a", "b", "c"],
    )
    check(regret.selected_config == "b", "surrogate selects expected config")
    check(regret.exact_best_config == "a", "exact objective identifies expected best")
    check(math.isclose(regret.relative_regret, 0.2), "relative regret is hand-computed")

    exact = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 5.0],
            [1.5, 2.5, 4.5],
            [3.0, 4.0, 6.0],
        ]
    )
    summary = cluster_bootstrap_ranking(
        exact,
        np.stack([exact, exact]),
        cluster_ids=["request-a", "request-b", "request-c", "request-d"],
        config_ids=["p0", "p1", "p2"],
        bootstrap_count=200,
        seed=5,
    )
    check(summary.cluster_count == 4, "bootstrap unit is request cluster")
    check(summary.bootstrap_count == 200, "requested bootstrap count is honored")
    check(summary.tau_b_point == 1.0, "identical surrogate has point tau one")
    check(summary.tau_b_ci_low == 1.0 and summary.tau_b_ci_high == 1.0, "tau CI toy")
    check(summary.relative_regret_point == 0.0, "identical surrogate has zero regret")
    check(
        summary.relative_regret_ci_low == 0.0
        and summary.relative_regret_ci_high == 0.0,
        "regret CI toy",
    )


def run_tests() -> None:
    test_route_ir_and_loader()
    test_c0_c1_cross_domain_costs()
    test_request_layer_s1r_certificate()
    test_canonical_sizes_and_placements()
    test_tau_regret_and_cluster_bootstrap()
    print(f"route_fidelity_p0b_core: all {ASSERTIONS} toy assertions passed")


if __name__ == "__main__":
    run_tests()

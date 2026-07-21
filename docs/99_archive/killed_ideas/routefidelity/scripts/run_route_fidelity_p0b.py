"""Frozen campaign runner for the RouteFidelity-EP P0-B experiment.

The runner has two state-changing commands:

``build-placements`` reads calibration captures only and freezes the exact
132-placement pools plus all analysis source hashes.  ``evaluate`` refuses to
run when any frozen source/artifact drifts.  A sealed evaluation also refuses
an existing output directory.

Evidence boundary: every cost is a teacher-forced logical C0/C1 record count.
This file makes no backend-frame, NIC-byte, latency, TTFT, TPOT, or P99 claim.
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

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

from route_fidelity_p0b_core import (
    C0_EXPANDED,
    C1_UNIQUE_OWNER,
    RouteIR,
    kendall_tau_b,
    load_route_csv,
    lower_cross_domain_cost,
    placement_hash,
    placement_manifest_hash,
    s1r_exact_degree,
    selection_regret,
)


ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
CORE = SELF.with_name("route_fidelity_p0b_core.py")
CORE_TEST = SELF.with_name("test_route_fidelity_p0b_core.py")
RUNNER_TEST = SELF.with_name("test_run_route_fidelity_p0b.py")
CAPTURE = SELF.with_name("capture_route_fidelity_p0b.py")
PREPARE = SELF.with_name("prepare_route_fidelity_p0b_configs.py")
HUMAN_PROTOCOL = ROOT / "RouteFidelity_EP_Sealed_P0B_Protocol_2026-07-18.md"
SCHEMA = "routefidelity_ep.p0b.campaign.v1"
RANK_TO_DOMAIN = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def hash_lines(values: Iterable[str]) -> str:
    return sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_new(path: Path, payload: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def parse_bindings(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{label} must be SLUG=PATH, got {raw!r}")
        slug, raw_path = raw.split("=", 1)
        if slug in result or slug not in {"olmoe", "llmjp"}:
            raise ValueError(f"invalid or duplicate {label} slug: {slug!r}")
        result[slug] = Path(raw_path).expanduser().resolve()
    if set(result) != {"olmoe", "llmjp"}:
        raise ValueError(f"{label} requires exactly olmoe and llmjp")
    return result


def model_slug(model: str) -> str:
    if model == "allenai/OLMoE-1B-7B-0924":
        return "olmoe"
    if model == "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M":
        return "llmjp"
    raise ValueError(f"unregistered primary model: {model}")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if protocol.get("schema") != "routefidelity_ep.p0b.machine_protocol.v1":
        raise ValueError("unexpected machine protocol schema")
    expected = protocol.get("machine_protocol_payload_sha256")
    unsigned = dict(protocol)
    unsigned.pop("machine_protocol_payload_sha256", None)
    observed = sha256_bytes(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )
    if observed != expected:
        raise RuntimeError(f"machine protocol canonical hash drift: {observed} != {expected}")
    for key in ("human_protocol", "document_builder", "freeze_tool"):
        source = protocol["source_integrity"][key]
        source_path = ROOT / source["path"]
        expected_source_hash = source.get("expected_sha256", source.get("sha256"))
        if sha256_file(source_path) != expected_source_hash:
            raise RuntimeError(f"pinned protocol source drift: {source_path}")
    return protocol


def cell_table(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for cell in protocol["primary_cells"]:
        slug = model_slug(cell["model"])
        cells[slug] = dict(cell)
    if set(cells) != {"olmoe", "llmjp"}:
        raise RuntimeError("primary cells are incomplete")
    return cells


def validate_capture(
    directory: Path,
    *,
    phase: str,
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[RouteIR, dict[str, Any], dict[str, Any]]:
    required = ["routes.csv", "request_manifest.json", "source_manifest.json", "config.json"]
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"capture artifact missing: {directory / name}")
    completed = read_json(directory / "config.json")
    request_manifest = read_json(directory / "request_manifest.json")
    if completed.get("status") != "COMPLETE" or completed.get("phase") != phase:
        raise RuntimeError(f"capture status/phase mismatch: {directory}")
    if completed["model"]["name"] != cell["model"]:
        raise RuntimeError(f"capture model mismatch: {directory}")
    if int(completed["ep_size"]) != int(cell["ep_size"]):
        raise RuntimeError("capture EP size mismatch")
    if not completed["exact_full_path"]["torch_equal"]:
        raise RuntimeError("instrumented full path was not bit exact")
    for name, expected in completed["outputs"].items():
        observed = sha256_file(directory / name)
        if observed != expected:
            raise RuntimeError(f"capture artifact hash drift: {directory / name}")

    partition = protocol["data"]["partitions"][phase]
    expected_ids = list(
        range(int(partition["offset"]), int(partition["offset"]) + int(partition["requests"]))
    )
    requests = request_manifest["requests"]
    by_id = {int(row["sample_id"]): row for row in requests}
    if sorted(by_id) != expected_ids or len(by_id) != len(requests):
        raise RuntimeError("request IDs do not match the frozen phase")
    expected_hashes = list(partition["document_sha256"])
    dataset_order_hashes = [by_id[sample_id]["sha256"] for sample_id in expected_ids]
    if dataset_order_hashes != expected_hashes:
        raise RuntimeError("capture document identity/order drift")
    if hash_lines(dataset_order_hashes) != partition["expected_hash_of_document_hashes"]:
        raise RuntimeError("capture partition hash drift")
    for sha_order, row in enumerate(sorted(requests, key=lambda item: item["sha256"])):
        if int(row["sha_order"]) != sha_order or int(row["home_rank"]) != sha_order % 8:
            raise RuntimeError("home-rank assignment drift")
    home_counts = np.bincount(
        np.asarray([int(row["home_rank"]) for row in requests]), minlength=8
    )
    expected_per_rank = int(partition["requests"]) // 8
    if home_counts.tolist() != [expected_per_rank] * 8:
        raise RuntimeError("request home ranks are not exactly balanced")

    route = load_route_csv(
        directory / "routes.csv",
        num_experts=int(cell["experts"]),
        top_k=int(cell["top_k"]),
    )
    if sorted(np.unique(route.request_id).tolist()) != expected_ids:
        raise RuntimeError("route request IDs do not close against manifest")
    for sample_id in expected_ids:
        mask = route.request_id == sample_id
        homes = np.unique(route.home_rank[mask])
        if homes.tolist() != [int(by_id[sample_id]["home_rank"])]:
            raise RuntimeError(f"route home rank disagrees for request {sample_id}")
    capture = completed["capture"]
    if int(capture["rows"]) != route.token_count * route.top_k:
        raise RuntimeError("route row count does not close")
    if int(capture["top_k"]) != route.top_k:
        raise RuntimeError("route top-k drift")
    return route, completed, request_manifest


def frequency_lpt(route: RouteIR, ep_size: int) -> np.ndarray:
    counts = np.bincount(route.experts.reshape(-1), minlength=route.num_experts)
    capacity = route.num_experts // ep_size
    load = np.zeros(ep_size, dtype=np.int64)
    slots = np.zeros(ep_size, dtype=np.int32)
    mapping = np.full(route.num_experts, -1, dtype=np.int32)
    order = sorted(range(route.num_experts), key=lambda expert: (-int(counts[expert]), expert))
    for expert in order:
        candidates = [rank for rank in range(ep_size) if slots[rank] < capacity]
        rank = min(candidates, key=lambda value: (int(load[value]), value))
        mapping[expert] = rank
        slots[rank] += 1
        load[rank] += int(counts[expert])
    return mapping


def coactivation_matrix(route: RouteIR) -> np.ndarray:
    weights = np.zeros((route.num_experts, route.num_experts), dtype=np.int64)
    for left in range(route.top_k - 1):
        left_values = route.experts[:, left]
        for right in range(left + 1, route.top_k):
            right_values = route.experts[:, right]
            np.add.at(weights, (left_values, right_values), 1)
            np.add.at(weights, (right_values, left_values), 1)
    return weights


def coactivation_hill_climb(
    route: RouteIR, start: np.ndarray, *, iterations: int, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    weights = coactivation_matrix(route)
    mapping = np.array(start, dtype=np.int32, copy=True)
    rng = np.random.default_rng(seed)
    accepted = 0
    for _ in range(iterations):
        left, right = rng.choice(route.num_experts, size=2, replace=False)
        rank_left = int(mapping[left])
        rank_right = int(mapping[right])
        if rank_left == rank_right:
            continue
        experts = np.arange(route.num_experts)
        mask = (experts != left) & (experts != right)
        delta_left = np.sum(
            weights[left, mask]
            * ((mapping[mask] == rank_right).astype(np.int64) - (mapping[mask] == rank_left))
        )
        delta_right = np.sum(
            weights[right, mask]
            * ((mapping[mask] == rank_left).astype(np.int64) - (mapping[mask] == rank_right))
        )
        if int(delta_left + delta_right) > 0:
            mapping[left], mapping[right] = rank_right, rank_left
            accepted += 1
    score = int(sum(weights[i, j] for i in range(route.num_experts) for j in range(i + 1, route.num_experts) if mapping[i] == mapping[j]))
    return mapping, {"accepted_swaps": accepted, "final_pair_colocation_score": score}


def random_balanced(num_experts: int, ep_size: int, seed: int) -> np.ndarray:
    capacity = num_experts // ep_size
    slots = np.repeat(np.arange(ep_size, dtype=np.int32), capacity)
    permutation = np.random.default_rng(seed).permutation(num_experts)
    mapping = np.empty(num_experts, dtype=np.int32)
    mapping[permutation] = slots
    return mapping


def validate_balanced(mapping: np.ndarray, num_experts: int, ep_size: int) -> None:
    if mapping.shape != (num_experts,):
        raise RuntimeError("placement shape drift")
    counts = np.bincount(mapping, minlength=ep_size)
    if counts.tolist() != [num_experts // ep_size] * ep_size:
        raise RuntimeError(f"placement is not exactly balanced: {counts.tolist()}")


def source_hashes(machine_protocol: Path) -> dict[str, str]:
    paths = {
        "machine_protocol": machine_protocol,
        "human_protocol": HUMAN_PROTOCOL,
        "runner": SELF,
        "core": CORE,
        "core_test": CORE_TEST,
        "runner_test": RUNNER_TEST,
        "capture": CAPTURE,
        "prepare_configs": PREPARE,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def build_placements(args: argparse.Namespace) -> None:
    protocol_path = Path(args.machine_protocol).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    cells = cell_table(protocol)
    captures = parse_bindings(args.capture, "--capture")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite placement lock: {output}")
    output.mkdir(parents=True)

    placement_hashes: dict[str, str] = {}
    document_sets: dict[str, list[str]] = {}
    capture_artifacts: dict[str, dict[str, str]] = {}
    for slug in ("olmoe", "llmjp"):
        route, completed, request_manifest = validate_capture(
            captures[slug], phase="calibration", cell=cells[slug], protocol=protocol
        )
        document_sets[slug] = [
            row["sha256"]
            for row in sorted(request_manifest["requests"], key=lambda row: row["sample_id"])
        ]
        cell = cells[slug]
        num_experts = int(cell["experts"])
        ep_size = int(cell["ep_size"])
        fixed: list[tuple[str, str, int | None, np.ndarray, dict[str, Any]]] = []
        capacity = num_experts // ep_size
        contiguous = np.arange(num_experts, dtype=np.int32) // capacity
        round_robin = np.arange(num_experts, dtype=np.int32) % ep_size
        lpt = frequency_lpt(route, ep_size)
        coactive, coactive_meta = coactivation_hill_climb(
            route,
            lpt,
            iterations=int(protocol["placement_pool"]["coactivation_hill_climbing_balanced_swaps"]),
            seed=2026071999,
        )
        fixed.extend(
            [
                ("contiguous", "fixed_control", None, contiguous, {}),
                ("round_robin", "fixed_control", None, round_robin, {}),
                ("calibration_frequency_lpt", "calibration_frequency", None, lpt, {}),
                (
                    "calibration_coactivation_balanced",
                    "calibration_coactivation",
                    2026071999,
                    coactive,
                    coactive_meta,
                ),
            ]
        )
        entries: list[dict[str, Any]] = []
        for name, kind, seed, mapping, metadata in fixed:
            validate_balanced(mapping, num_experts, ep_size)
            entries.append(
                {
                    "name": name,
                    "kind": kind,
                    "seed": seed,
                    "expert_to_rank": mapping.astype(int).tolist(),
                    "mapping_sha256": placement_hash(mapping),
                    "metadata": metadata,
                }
            )
        random_seeds = list(protocol["placement_pool"]["balanced_random_seeds"])
        for index, seed in enumerate(random_seeds):
            mapping = random_balanced(num_experts, ep_size, int(seed))
            validate_balanced(mapping, num_experts, ep_size)
            entries.append(
                {
                    "name": f"random_{index:03d}",
                    "kind": "balanced_random",
                    "seed": int(seed),
                    "expert_to_rank": mapping.astype(int).tolist(),
                    "mapping_sha256": placement_hash(mapping),
                    "metadata": {},
                }
            )
        if len(entries) != int(protocol["placement_pool"]["placements_per_model"]):
            raise RuntimeError("placement count drift")
        entries.sort(key=lambda row: (row["mapping_sha256"], row["name"]))
        mapping_dict = {
            row["name"]: np.asarray(row["expert_to_rank"], dtype=np.int32) for row in entries
        }
        registry = {
            "schema": "routefidelity_ep.p0b.placement_registry.v1",
            "cell": cell["cell"],
            "model": cell["model"],
            "num_experts": num_experts,
            "top_k": int(cell["top_k"]),
            "ep_size": ep_size,
            "tie_order": "mapping_sha256_then_name_lexicographic",
            "calibration_capture": {
                "directory": str(captures[slug]),
                "config_sha256": sha256_file(captures[slug] / "config.json"),
                "routes_sha256": sha256_file(captures[slug] / "routes.csv"),
                "request_manifest_sha256": sha256_file(captures[slug] / "request_manifest.json"),
                "source_manifest_sha256": sha256_file(captures[slug] / "source_manifest.json"),
                "frozen_capture_config_sha256": completed["frozen_config"]["sha256"],
            },
            "generation": {
                "frequency": "balanced LPT descending expert occurrence count",
                "coactivation": "pair-count objective; 20000 cross-rank proposals; strictly positive swaps",
                "coactivation_seed": 2026071999,
                "random": "independent numpy.default_rng(seed) permutation for each frozen seed",
            },
            "placement_manifest_sha256": placement_manifest_hash(mapping_dict),
            "placements": entries,
        }
        registry_path = output / f"placements_{slug}.json"
        placement_hashes[slug] = write_new(registry_path, json_bytes(registry))
        capture_artifacts[slug] = registry["calibration_capture"]

    if document_sets["olmoe"] != document_sets["llmjp"]:
        raise RuntimeError("the two calibration captures do not use identical raw documents")
    lock = {
        "schema": SCHEMA,
        "status": "PLACEMENTS_AND_ANALYSIS_FROZEN_BEFORE_SEALED_CAPTURE",
        "evidence_boundary": protocol["evidence_boundary"],
        "machine_protocol_payload_sha256": protocol["machine_protocol_payload_sha256"],
        "source_sha256": source_hashes(protocol_path),
        "placement_registry_sha256": placement_hashes,
        "calibration_capture": capture_artifacts,
        "calibration_document_hash_of_hashes": hash_lines(document_sets["olmoe"]),
        "sealed_expected_hash_of_hashes": protocol["data"]["partitions"]["sealed"][
            "expected_hash_of_document_hashes"
        ],
        "sealed_output_policy": "new directory only; evaluate once",
    }
    lock_hash = write_new(output / "campaign_lock.json", json_bytes(lock))
    print(
        json.dumps(
            {
                "status": lock["status"],
                "campaign_lock_sha256": lock_hash,
                "placement_registry_sha256": placement_hashes,
            },
            indent=2,
        )
    )


def load_placement_registry(
    path: Path, *, cell: Mapping[str, Any], expected_hash: str
) -> tuple[list[str], list[np.ndarray], dict[str, Any]]:
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"placement registry hash drift: {path}")
    registry = read_json(path)
    if registry.get("model") != cell["model"]:
        raise RuntimeError("placement registry model mismatch")
    entries = registry["placements"]
    ids: list[str] = []
    mappings: list[np.ndarray] = []
    observed_order: list[tuple[str, str]] = []
    for entry in entries:
        mapping = np.asarray(entry["expert_to_rank"], dtype=np.int32)
        validate_balanced(mapping, int(cell["experts"]), int(cell["ep_size"]))
        observed_hash = placement_hash(mapping)
        if observed_hash != entry["mapping_sha256"]:
            raise RuntimeError("placement mapping hash drift")
        ids.append(f"{observed_hash}:{entry['name']}")
        mappings.append(mapping)
        observed_order.append((observed_hash, entry["name"]))
    if observed_order != sorted(observed_order):
        raise RuntimeError("placement registry tie order drift")
    if len(ids) != 132 or len(set(ids)) != 132:
        raise RuntimeError("placement registry must have 132 named configurations")
    return ids, mappings, registry


def verify_campaign_lock(
    placement_dir: Path, protocol_path: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    lock_path = placement_dir / "campaign_lock.json"
    lock = read_json(lock_path)
    if lock.get("status") != "PLACEMENTS_AND_ANALYSIS_FROZEN_BEFORE_SEALED_CAPTURE":
        raise RuntimeError("campaign lock status drift")
    if lock["machine_protocol_payload_sha256"] != protocol["machine_protocol_payload_sha256"]:
        raise RuntimeError("campaign lock points to another protocol")
    observed_sources = source_hashes(protocol_path)
    if observed_sources != lock["source_sha256"]:
        changed = sorted(
            name
            for name in set(observed_sources) | set(lock["source_sha256"])
            if observed_sources.get(name) != lock["source_sha256"].get(name)
        )
        raise RuntimeError(f"analysis source drift after placement freeze: {changed}")
    return lock


def logical_cost_matrix(
    route: RouteIR,
    mappings: list[np.ndarray],
    *,
    contract: str,
    chunk_configs: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-request logical cross-domain counts for all placements."""

    requests, request_slot = np.unique(route.request_id, return_inverse=True)
    mapping_array = np.stack(mappings).astype(np.int16, copy=False)
    result = np.empty((len(requests), len(mappings)), dtype=np.int64)
    home_domain = RANK_TO_DOMAIN[route.home_rank][None, :, None]
    for start in range(0, len(mappings), chunk_configs):
        stop = min(start + chunk_configs, len(mappings))
        owners = mapping_array[start:stop, route.experts]
        if contract == C0_EXPANDED:
            row_cross = np.count_nonzero(RANK_TO_DOMAIN[owners] != home_domain, axis=2)
        elif contract == C1_UNIQUE_OWNER:
            sorted_owners = np.sort(owners, axis=2)
            unique = np.ones(sorted_owners.shape, dtype=bool)
            unique[:, :, 1:] = sorted_owners[:, :, 1:] != sorted_owners[:, :, :-1]
            row_cross = np.count_nonzero(
                unique & (RANK_TO_DOMAIN[sorted_owners] != home_domain), axis=2
            )
        else:
            raise ValueError(f"unsupported logical contract: {contract}")
        for local in range(stop - start):
            result[:, start + local] = np.bincount(
                request_slot, weights=row_cross[local], minlength=len(requests)
            ).astype(np.int64)
    return requests, result


def input_token_vector(
    requests: np.ndarray, request_manifest: Mapping[str, Any]
) -> np.ndarray:
    by_id = {
        int(row["sample_id"]): int(row["tokens_used"])
        for row in request_manifest["requests"]
    }
    values = np.asarray([by_id[int(request)] for request in requests], dtype=np.float64)
    if np.any(values <= 0):
        raise RuntimeError("request token count must be positive")
    return values


def aggregate_scores(counts: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    return np.sum(counts, axis=-2) / float(np.sum(tokens))


def point_seed_statistics(
    exact_counts: np.ndarray,
    surrogate_counts: np.ndarray,
    tokens: np.ndarray,
    config_ids: list[str],
) -> list[dict[str, Any]]:
    exact = aggregate_scores(exact_counts, tokens)
    rows: list[dict[str, Any]] = []
    for seed_index, seed_counts in enumerate(surrogate_counts):
        surrogate = aggregate_scores(seed_counts, tokens)
        regret = selection_regret(
            exact, surrogate, config_ids=config_ids, minimize=True
        )
        rows.append(
            {
                "seed_index": seed_index,
                "selected_config": regret.selected_config,
                "exact_best_config": regret.exact_best_config,
                "selected_exact_score": regret.selected_exact_score,
                "exact_best_score": regret.exact_best_score,
                "relative_regret": regret.relative_regret,
                "kendall_tau_b": kendall_tau_b(exact, surrogate),
            }
        )
    return rows


def bootstrap_seed_median_regret(
    exact_counts: np.ndarray,
    surrogate_counts: np.ndarray,
    tokens: np.ndarray,
    *,
    resamples: int,
    seed: int,
    batch_size: int = 250,
) -> np.ndarray:
    """Paired request bootstrap; synthesis seeds remain a nested uncertainty."""

    request_count, config_count = exact_counts.shape
    seed_count = surrogate_counts.shape[0]
    rng = np.random.default_rng(seed)
    output = np.empty(resamples, dtype=np.float64)
    cursor = 0
    exact_float = exact_counts.astype(np.float64)
    surrogate_float = surrogate_counts.astype(np.float64)
    while cursor < resamples:
        size = min(batch_size, resamples - cursor)
        draws = rng.integers(0, request_count, size=(size, request_count))
        weights = np.zeros((size, request_count), dtype=np.float64)
        np.add.at(
            weights,
            (np.repeat(np.arange(size), request_count), draws.reshape(-1)),
            1.0,
        )
        # NumPy 2.0 linked against Accelerate can emit spurious floating-point
        # warnings from small GEMV/GEMM calls even when the result is finite.
        # The explicit contraction is deterministic and avoids that backend.
        denominator = np.einsum("br,r->b", weights, tokens, optimize=False)
        exact_scores = np.einsum(
            "br,rc->bc", weights, exact_float, optimize=False
        ) / denominator[:, None]
        exact_best = np.min(exact_scores, axis=1)
        seed_regrets = np.empty((seed_count, size), dtype=np.float64)
        row_index = np.arange(size)
        for synthesis_seed in range(seed_count):
            surrogate_scores = np.einsum(
                "br,rc->bc",
                weights,
                surrogate_float[synthesis_seed],
                optimize=False,
            ) / denominator[:, None]
            selected = np.argmin(surrogate_scores, axis=1)
            selected_exact = exact_scores[row_index, selected]
            seed_regrets[synthesis_seed] = (selected_exact - exact_best) / np.maximum(
                exact_best, 1e-12
            )
        output[cursor : cursor + size] = np.median(seed_regrets, axis=0)
        cursor += size
    return output


def holm_adjust(raw_p: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p, key=lambda key: (raw_p[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * raw_p[key]))
        adjusted[key] = running
    return adjusted


def holm_alpha(raw_p: Mapping[str, float], family_alpha: float = 0.05) -> dict[str, float]:
    ordered = sorted(raw_p, key=lambda key: (raw_p[key], key))
    count = len(ordered)
    return {key: family_alpha / (count - index) for index, key in enumerate(ordered)}


def bits_for_values(count: int) -> int:
    return max(1, int(math.ceil(math.log2(max(2, int(count))))))


def packed_size_accounting(route: RouteIR, window_tokens: int) -> dict[str, Any]:
    """Frozen mathematical bit packing for the P0-B S3/S4 size gate.

    Order itself carries token position in S4.  Both formats share a compact
    request header (request ordinal, home rank, token count) and layer
    boundaries.  S3 replaces ordered token hyperedges with per-window unique
    hyperedges and exact counts.  No Python struct alignment is counted.
    """

    requests = np.unique(route.request_id)
    layers = np.unique(route.layer_id)
    expert_bits = bits_for_values(route.num_experts)
    request_bits = bits_for_values(len(requests))
    layer_bits = bits_for_values(len(layers))
    home_bits = bits_for_values(len(RANK_TO_DOMAIN))
    maximum_position = int(np.max(route.token_position)) + 1
    token_count_bits = bits_for_values(maximum_position + 1)
    chunk_bits = bits_for_values(int(math.ceil(maximum_position / window_tokens)))
    header_bits = 32 * 5  # version, E, k, request count, window/ordered marker
    request_header_bits = request_bits + home_bits + token_count_bits
    s4_bits = header_bits
    s3_bits = header_bits
    group_count = 0
    dictionary_entries = 0
    for request_index, request in enumerate(requests):
        del request_index
        request_mask = route.request_id == request
        s4_bits += request_header_bits
        s3_bits += request_header_bits
        for layer in layers:
            mask = request_mask & (route.layer_id == layer)
            indices = np.flatnonzero(mask)
            if len(indices) == 0:
                continue
            order = indices[np.argsort(route.token_position[indices], kind="stable")]
            s4_bits += layer_bits + len(order) * route.top_k * expert_bits
            windows = route.token_position[order] // int(window_tokens)
            for window in np.unique(windows):
                rows = np.sort(route.experts[order[windows == window]], axis=1)
                unique_rows, counts = np.unique(rows, axis=0, return_counts=True)
                group_size = int(np.sum(counts))
                entry_count_bits = bits_for_values(group_size + 1)
                count_bits = bits_for_values(group_size + 1)
                s3_bits += layer_bits + chunk_bits + entry_count_bits
                s3_bits += len(unique_rows) * (route.top_k * expert_bits + count_bits)
                group_count += 1
                dictionary_entries += len(unique_rows)
    s4_bytes = int(math.ceil(s4_bits / 8.0))
    s3_bytes = int(math.ceil(s3_bits / 8.0))
    return {
        "schema": "routefidelity_ep.p0b.bitpacked_size.v1",
        "expert_id_bits": expert_bits,
        "request_id_bits": request_bits,
        "layer_id_bits": layer_bits,
        "home_rank_bits": home_bits,
        "token_count_bits": token_count_bits,
        "chunk_id_bits": chunk_bits,
        "window_tokens": int(window_tokens),
        "s3_raw_packed_bytes": s3_bytes,
        "s4_raw_packed_bytes": s4_bytes,
        "s3_over_s4": s3_bytes / s4_bytes,
        "s3_group_count": group_count,
        "s3_dictionary_entries": dictionary_entries,
        "boundary": "mathematical canonical bits, not backend serialization or wire bytes",
    }


def certificate_dict(value: Any) -> dict[str, Any]:
    return {
        "group_count": value.group_count,
        "attempted_swaps": value.attempted_swaps,
        "accepted_swaps": value.accepted_swaps,
        "acceptance_rate": value.acceptance_rate,
        "duplicate_rows": value.duplicate_rows,
        "degree_tv_mean": value.degree_tv_mean,
        "degree_tv_max": value.degree_tv_max,
        "token_jaccard_mean": value.token_jaccard_mean,
        "pair_distance_mean": value.pair_distance_mean,
        "pair_distance_max": value.pair_distance_max,
    }


def request_expert_degree(route: RouteIR, requests: np.ndarray) -> np.ndarray:
    slots = {int(request): index for index, request in enumerate(requests)}
    degree = np.zeros((len(requests), route.num_experts), dtype=np.int64)
    repeated_slots = np.asarray([slots[int(request)] for request in route.request_id])
    for rank in range(route.top_k):
        np.add.at(degree, (repeated_slots, route.experts[:, rank]), 1)
    return degree


def c0_from_degree(
    degree: np.ndarray,
    mappings: list[np.ndarray],
    home_ranks: np.ndarray,
) -> np.ndarray:
    home_domains = RANK_TO_DOMAIN[home_ranks]
    result = np.empty((degree.shape[0], len(mappings)), dtype=np.int64)
    for index, mapping in enumerate(mappings):
        remote = RANK_TO_DOMAIN[mapping][None, :] != home_domains[:, None]
        result[:, index] = np.sum(degree * remote, axis=1)
    return result


def evaluate_model(
    slug: str,
    route: RouteIR,
    request_manifest: Mapping[str, Any],
    config_ids: list[str],
    mappings: list[np.ndarray],
    protocol: Mapping[str, Any],
    bootstrap_seed_offset: int,
) -> tuple[dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    requests, exact_c1 = logical_cost_matrix(
        route, mappings, contract=C1_UNIQUE_OWNER
    )
    tokens = input_token_vector(requests, request_manifest)
    home_by_id = {
        int(row["sample_id"]): int(row["home_rank"])
        for row in request_manifest["requests"]
    }
    homes = np.asarray([home_by_id[int(request)] for request in requests], dtype=np.int32)
    exact_degree = request_expert_degree(route, requests)
    exact_c0 = c0_from_degree(exact_degree, mappings, homes)
    # One direct lowerer comparison catches an independent C0 formula mistake;
    # exact degree equality below then certifies every placement and every seed.
    direct_c0 = lower_cross_domain_cost(
        route, mappings[0], RANK_TO_DOMAIN, contract=C0_EXPANDED
    )
    if not np.array_equal(direct_c0.request_id, requests) or not np.array_equal(
        direct_c0.cross_domain_records, exact_c0[:, 0]
    ):
        raise RuntimeError("independent C0 lowerers disagree")

    synthesis_seeds = list(protocol["representations"]["S1_R"]["seeds"])
    synthetic_c1 = np.empty(
        (len(synthesis_seeds), len(requests), len(mappings)), dtype=np.int64
    )
    certificates: list[dict[str, Any]] = []
    for index, seed in enumerate(synthesis_seeds):
        synthesized = s1r_exact_degree(route, seed=int(seed), swap_multiplier=8)
        cert = synthesized.certificate
        if cert.degree_tv_max != 0.0 or cert.duplicate_rows != 0:
            raise RuntimeError(f"S1-R invariant failed for seed {seed}")
        synthetic_degree = request_expert_degree(synthesized.route, requests)
        if not np.array_equal(synthetic_degree, exact_degree):
            raise RuntimeError(f"S1-R request-degree drift for seed {seed}")
        synthetic_c0 = c0_from_degree(synthetic_degree, mappings, homes)
        if not np.array_equal(synthetic_c0, exact_c0):
            raise RuntimeError(f"C0 negative control failed for seed {seed}")
        synthetic_requests, synthetic_c1[index] = logical_cost_matrix(
            synthesized.route, mappings, contract=C1_UNIQUE_OWNER
        )
        if not np.array_equal(synthetic_requests, requests):
            raise RuntimeError("S1-R request order drift")
        certificates.append({"seed": int(seed), **certificate_dict(cert)})
        print(
            f"{slug}: S1-R {index + 1}/{len(synthesis_seeds)} "
            f"accepted={cert.accepted_swaps} pair_distance={cert.pair_distance_mean:.4f}",
            flush=True,
        )

    point_rows = point_seed_statistics(
        exact_c1, synthetic_c1, tokens, config_ids
    )
    point_regrets = np.asarray([row["relative_regret"] for row in point_rows])
    bootstrap_spec = protocol["statistics"]["bootstrap"]
    bootstrap = bootstrap_seed_median_regret(
        exact_c1,
        synthetic_c1,
        tokens,
        resamples=int(bootstrap_spec["resamples"]),
        seed=int(bootstrap_spec["seed"]) + int(bootstrap_seed_offset),
    )
    threshold = float(
        protocol["primary_thresholds"]["problem_gate"]["S1_R_seed_point_regret_min"]
    )
    size = packed_size_accounting(
        route, int(protocol["representations"]["S3_W"]["primary_window_tokens"])
    )
    exact_scores = aggregate_scores(exact_c1, tokens)
    best_index = int(np.argmin(exact_scores))
    result = {
        "slug": slug,
        "request_count": len(requests),
        "route_token_layer_rows": route.token_count,
        "input_tokens": int(np.sum(tokens)),
        "placement_count": len(mappings),
        "exact_best_config": config_ids[best_index],
        "exact_best_cross_domain_records_per_input_token": float(exact_scores[best_index]),
        "S1_R": {
            "point_by_seed": point_rows,
            "seeds_at_or_above_5pct": int(np.count_nonzero(point_regrets >= threshold)),
            "seed_median_regret": float(np.median(point_regrets)),
            "seed_min_regret": float(np.min(point_regrets)),
            "seed_max_regret": float(np.max(point_regrets)),
            "raw_one_sided_p_at_zero": float(
                (1 + np.count_nonzero(bootstrap <= 0.0)) / (len(bootstrap) + 1)
            ),
            "bootstrap_median_regret_point": float(np.median(point_regrets)),
            "bootstrap_unadjusted_one_sided_95pct_lower": float(
                np.quantile(bootstrap, 0.05, method="linear")
            ),
            "bootstrap_median": float(np.median(bootstrap)),
            "bootstrap_95pct_interval": [
                float(np.quantile(bootstrap, 0.025, method="linear")),
                float(np.quantile(bootstrap, 0.975, method="linear")),
            ],
            "certificates": certificates,
        },
        "C0_negative_control": {
            "all_20_seeds_all_132_placements_exact": True,
            "proof": "request-level expert-degree equality plus independent direct lowerer check",
        },
        "S3_W": {
            "cost_equivalence_to_S4": "CONSTRUCTIONAL_EXACT_FOR_ADDITIVE_C1",
            "point_regret": 0.0,
            "kendall_tau_b": 1.0,
            "size": size,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    return result, bootstrap


def render_report(
    phase: str, verdict: str, models: Mapping[str, Mapping[str, Any]], valid: bool
) -> str:
    lines = [
        "# RouteFidelity-EP P0-B 严格验证结果",
        "",
        f"> Phase: `{phase}`  ",
        f"> Verdict: **{verdict}**  ",
        "> Boundary: teacher-forced logical C0/C1 records only; not backend frames, wire bytes, latency, TTFT, TPOT, or P99.",
        "",
        "## Primary problem gate",
        "",
        "| model | seeds regret>=5% | seed median regret | Holm lower bound | Holm p | H-P |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for slug in ("olmoe", "llmjp"):
        model = models[slug]
        s1 = model["S1_R"]
        lines.append(
            f"| {slug} | {s1['seeds_at_or_above_5pct']}/20 | "
            f"{s1['seed_median_regret']:.2%} | {s1['holm_one_sided_lower']:.2%} | "
            f"{s1['holm_adjusted_p']:.4g} | {'PASS' if model['problem_gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "H-P 要求两个模型均达到 16/20 seeds regret>=5%、seed median>=5%，且 Holm-adjusted one-sided lower bound >0。任何一项失败即杀死 CCF-C 主线。",
            "",
            "## Controls and method boundary",
            "",
        ]
    )
    for slug in ("olmoe", "llmjp"):
        model = models[slug]
        size = model["S3_W"]["size"]
        lines.append(
            f"- `{slug}`: C0 all seeds/placements exact; S3 C1 regret=0 is constructional; "
            f"bit-packed size ratio={size['s3_over_s4']:.2%}; method gate="
            f"`{model['method_gate_status']}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 若 verdict 为 `KILL_CCFC_MAINLINE`，证据表示 request-conditioned exact expert degrees 已足以在此 C1 配置池中保持 placement 决策；不能靠 architecture-only 或 P99 maximum 重新包装该主线。",
            "- 若为 `EXACT_REPLAY_ONLY / KILL_METHOD_NOVELTY`，说明问题存在，但紧凑 S3 表示没有达到预注册的 size/decision gate。",
            "- 只有 `PROMOTE_TO_GPU_P1` 才允许实现真实 backend adapter；即便晋级，本实验本身仍不证明系统加速。",
            "",
            f"Run validity: `{valid}`.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> None:
    phase = args.phase
    protocol_path = Path(args.machine_protocol).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    cells = cell_table(protocol)
    captures = parse_bindings(args.capture, "--capture")
    placement_dir = Path(args.placement_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    lock = verify_campaign_lock(placement_dir, protocol_path, protocol)

    model_results: dict[str, dict[str, Any]] = {}
    bootstrap_arrays: dict[str, np.ndarray] = {}
    document_sets: dict[str, list[str]] = {}
    for offset, slug in enumerate(("olmoe", "llmjp")):
        route, _completed, request_manifest = validate_capture(
            captures[slug], phase=phase, cell=cells[slug], protocol=protocol
        )
        document_sets[slug] = [
            row["sha256"]
            for row in sorted(request_manifest["requests"], key=lambda row: row["sample_id"])
        ]
        registry_path = placement_dir / f"placements_{slug}.json"
        config_ids, mappings, _registry = load_placement_registry(
            registry_path,
            cell=cells[slug],
            expected_hash=lock["placement_registry_sha256"][slug],
        )
        result, bootstrap = evaluate_model(
            slug,
            route,
            request_manifest,
            config_ids,
            mappings,
            protocol,
            bootstrap_seed_offset=offset,
        )
        model_results[slug] = result
        bootstrap_arrays[slug] = bootstrap
    if document_sets["olmoe"] != document_sets["llmjp"]:
        raise RuntimeError("the two primary cells do not share identical documents")

    raw_p = {
        slug: float(model_results[slug]["S1_R"]["raw_one_sided_p_at_zero"])
        for slug in model_results
    }
    adjusted_p = holm_adjust(raw_p)
    alphas = holm_alpha(raw_p)
    problem = protocol["primary_thresholds"]["problem_gate"]
    for slug in ("olmoe", "llmjp"):
        model = model_results[slug]
        s1 = model["S1_R"]
        lower = float(np.quantile(bootstrap_arrays[slug], alphas[slug], method="linear"))
        s1["holm_stepdown_alpha"] = alphas[slug]
        s1["holm_one_sided_lower"] = lower
        s1["holm_adjusted_p"] = adjusted_p[slug]
        model["problem_gate_pass"] = bool(
            s1["seeds_at_or_above_5pct"]
            >= int(problem["S1_R_seeds_meeting_point_regret_min"])
            and s1["seed_median_regret"]
            >= float(problem["S1_R_seed_median_regret_min"])
            and lower > 0.0
            and adjusted_p[slug] < 0.05
        )

    all_problem = all(model_results[slug]["problem_gate_pass"] for slug in model_results)
    method = protocol["primary_thresholds"]["method_gate"]
    if all_problem:
        for slug in ("olmoe", "llmjp"):
            model = model_results[slug]
            s1 = model["S1_R"]
            s3 = model["S3_W"]
            passed = bool(
                s3["point_regret"] <= float(method["S3_W_regret_max"])
                and s3["kendall_tau_b"]
                >= float(method["kendall_tau_b_one_sided_95pct_lower_bound_min"])
                and s1["seed_median_regret"] - s3["point_regret"]
                >= float(method["S1_R_minus_S3_W_regret_min"])
                and s3["size"]["s3_over_s4"]
                <= float(method["S3_W_raw_canonical_size_over_S4_max"])
            )
            model["method_gate_status"] = "PASS" if passed else "FAIL"
        all_method = all(
            model_results[slug]["method_gate_status"] == "PASS" for slug in model_results
        )
    else:
        for model in model_results.values():
            model["method_gate_status"] = "NOT_RUN_BY_PROTOCOL"
        all_method = False

    if phase == "calibration":
        verdict = "CALIBRATION_ENGINEERING_ONLY"
    elif not all_problem:
        verdict = "KILL_CCFC_MAINLINE"
    elif not all_method:
        verdict = "EXACT_REPLAY_ONLY / KILL_METHOD_NOVELTY"
    else:
        verdict = "PROMOTE_TO_GPU_P1"

    summary = {
        "schema": "routefidelity_ep.p0b.evaluation.v1",
        "phase": phase,
        "verdict": verdict,
        "valid": True,
        "evidence_boundary": protocol["evidence_boundary"],
        "machine_protocol_payload_sha256": protocol["machine_protocol_payload_sha256"],
        "campaign_lock_sha256": sha256_file(placement_dir / "campaign_lock.json"),
        "documents_hash_of_hashes": hash_lines(document_sets["olmoe"]),
        "models": model_results,
    }
    output.mkdir(parents=True)
    write_new(output / "summary.json", json_bytes(summary))
    bootstrap_path = output / "bootstrap_regret_replicates.npz"
    if bootstrap_path.exists():
        raise FileExistsError(bootstrap_path)
    np.savez_compressed(bootstrap_path, **bootstrap_arrays)
    write_new(output / "report.md", render_report(phase, verdict, model_results, True).encode("utf-8"))
    artifact_manifest = {
        "summary.json": sha256_file(output / "summary.json"),
        "report.md": sha256_file(output / "report.md"),
        "bootstrap_regret_replicates.npz": sha256_file(bootstrap_path),
    }
    write_new(output / "artifact_manifest.json", json_bytes(artifact_manifest))
    print(json.dumps({"phase": phase, "verdict": verdict, "models": {slug: {"problem_gate_pass": result["problem_gate_pass"], "seed_median_regret": result["S1_R"]["seed_median_regret"], "seeds_ge_5pct": result["S1_R"]["seeds_at_or_above_5pct"], "holm_lower": result["S1_R"]["holm_one_sided_lower"]} for slug, result in model_results.items()}}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-placements")
    build.add_argument("--machine-protocol", required=True)
    build.add_argument("--capture", action="append", required=True, help="SLUG=calibration capture directory")
    build.add_argument("--output-dir", required=True)

    run = subparsers.add_parser("evaluate")
    run.add_argument("--machine-protocol", required=True)
    run.add_argument("--phase", choices=("calibration", "sealed"), required=True)
    run.add_argument("--capture", action="append", required=True, help="SLUG=capture directory")
    run.add_argument("--placement-dir", required=True)
    run.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-placements":
        build_placements(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

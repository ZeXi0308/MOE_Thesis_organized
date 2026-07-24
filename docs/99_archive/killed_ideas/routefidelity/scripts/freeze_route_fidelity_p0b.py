"""Freeze the RouteFidelity-EP P0-B machine-readable protocol.

This program performs only preregistration and data-leakage checks.  It does
not capture routes, synthesize representations, evaluate placements, or open
sealed outputs.  A successful invocation creates exactly one
``machine_protocol.json`` in a previously nonexistent output directory.
"""

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

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable, Iterable


SCHEMA = "routefidelity_ep.p0b.machine_protocol.v1"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
HUMAN_PROTOCOL_RELATIVE_PATH = Path(
    "RouteFidelity_EP_Sealed_P0B_Protocol_2026-07-18.md"
)
HUMAN_PROTOCOL_SHA256 = (
    "89f1213c3eea19b5847f15b2f1d8c0c18cf09e985ca9c53067a8108fe99c6214"
)
PROMPTS_RELATIVE_PATH = Path("experiments/idea_a_mac/prompts.py")
PROMPTS_SHA256 = (
    "d69c1832f627e03c187e19c7052183bbeabccc932f09533860b71c89c9c6029f"
)

DATASET = {
    "name": "wikitext",
    "configuration": "wikitext-2-raw-v1",
    "split": "train",
    "sampling_unit": "article-level document",
    "minimum_characters": 500,
    "shuffle_seed": 20260717,
}
PARTITIONS = {
    "calibration": {
        "offset": 192,
        "requests": 32,
        "expected_hash_of_document_hashes": (
            "264360550fe1caa24f97411155812a746290ad5742f1fe9842555427c7a4ba2c"
        ),
        "freshness_gate": True,
    },
    "sealed": {
        "offset": 224,
        "requests": 64,
        "expected_hash_of_document_hashes": (
            "255af39aa7bb32c6f972195b0399a3c59b90af0460c11f93f7f780fb07391b04"
        ),
        "freshness_gate": True,
    },
    "reserve": {
        "offset": 288,
        "requests": 128,
        "expected_hash_of_document_hashes": (
            "5eda4c4314fbb20c64ad56493fb508d4008a7f26c5990c481bbd94c5fa7a4f57"
        ),
        "freshness_gate": False,
    },
}

PRIMARY_CELLS = [
    {
        "cell": "P1",
        "model": "allenai/OLMoE-1B-7B-0924",
        "experts": 64,
        "top_k": 8,
        "ep_size": 8,
        "contract": "C1_rank_major_unique_owner",
        "primary_objective": "cross_domain_unique_owner_records_per_token",
    },
    {
        "cell": "P2",
        "model": "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M",
        "experts": 32,
        "top_k": 16,
        "ep_size": 8,
        "contract": "C1_rank_major_unique_owner",
        "primary_objective": "cross_domain_unique_owner_records_per_token",
    },
]
S1_SEEDS = [2026071800 + index for index in range(20)]
PLACEMENT_SEEDS = [2026072000 + index for index in range(128)]
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=WORKSPACE_ROOT,
        help="workspace containing the human protocol and experiments directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory in which machine_protocol.json will be created",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def hash_lines(values: Iterable[str]) -> str:
    """Hash ordered UTF-8 lines, including one final newline."""
    return sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path is outside workspace: {path}") from exc


def verify_pinned_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: observed {observed}, expected {expected}"
        )
    return observed


def walk_document_sha256(value: object) -> list[str]:
    """Extract exact ``sha256`` fields from a JSON data manifest."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "sha256" and isinstance(item, str):
                if not SHA256_PATTERN.fullmatch(item):
                    raise RuntimeError(f"malformed sha256 value in JSON manifest: {item!r}")
                found.append(item.lower())
            found.extend(walk_document_sha256(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(walk_document_sha256(item))
    return found


def manifest_document_hashes(path: Path, raw: bytes) -> list[str]:
    if path.name == "data_manifest.json":
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot parse JSON data manifest: {path}") from exc
        hashes = walk_document_sha256(payload)
    elif path.name == "data_manifest.csv":
        hashes = []
        try:
            with io.StringIO(raw.decode("utf-8"), newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "sha256" not in reader.fieldnames:
                    raise RuntimeError(f"CSV data manifest lacks sha256 column: {path}")
                for line_number, row in enumerate(reader, start=2):
                    value = row.get("sha256", "")
                    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
                        raise RuntimeError(
                            f"malformed sha256 in {path}:{line_number}: {value!r}"
                        )
                    hashes.append(value.lower())
        except (OSError, UnicodeError, csv.Error) as exc:
            raise RuntimeError(f"cannot parse CSV data manifest: {path}") from exc
    else:
        raise ValueError(f"unsupported data manifest name: {path}")

    if not hashes:
        raise RuntimeError(f"data manifest contains no document SHA256 values: {path}")
    return hashes


def scan_historical_registry(workspace_root: Path) -> dict[str, object]:
    outputs_root = workspace_root / "experiments/idea_a_mac/outputs"
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"outputs root does not exist: {outputs_root}")
    paths = sorted(
        {
            *outputs_root.rglob("data_manifest.json"),
            *outputs_root.rglob("data_manifest.csv"),
        },
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise RuntimeError(f"no data_manifest.json/csv files found below {outputs_root}")

    files: list[dict[str, object]] = []
    occurrences: list[str] = []
    for path in paths:
        raw = path.read_bytes()
        hashes = manifest_document_hashes(path, raw)
        occurrences.extend(hashes)
        files.append(
            {
                "path": relative_posix(path, workspace_root),
                "file_sha256": sha256_bytes(raw),
                "document_hash_occurrences": len(hashes),
                "unique_document_hashes": len(set(hashes)),
            }
        )

    unique = sorted(set(occurrences))
    registry = {
        "scan_globs": ["**/data_manifest.json", "**/data_manifest.csv"],
        "outputs_root": relative_posix(outputs_root, workspace_root),
        "manifest_file_count": len(files),
        "document_hash_occurrences": len(occurrences),
        "unique_document_hash_count": len(unique),
        "unique_document_hashes_sha256": hash_lines(unique),
        "unique_document_hashes": unique,
        "manifest_files": files,
    }
    registry["manifest_registry_payload_sha256"] = canonical_sha256(registry)
    return registry


def load_document_builder(path: Path) -> Callable[..., list[str]]:
    """Load the already checksum-verified builder from its exact file path."""
    spec = importlib.util.spec_from_file_location(
        "routefidelity_frozen_prompts", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct import spec for document builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "get_wikitext2_documents", None)
    if not callable(builder):
        raise RuntimeError(f"document builder callable is missing from {path}")
    return builder


def rebuild_partitions(
    get_documents: Callable[..., list[str]],
) -> dict[str, dict[str, object]]:
    end = max(
        int(spec["offset"]) + int(spec["requests"])
        for spec in PARTITIONS.values()
    )
    documents = get_documents(
        end,
        min_chars=int(DATASET["minimum_characters"]),
        offset=0,
        split=str(DATASET["split"]),
        seed=int(DATASET["shuffle_seed"]),
    )
    document_hashes = [sha256_bytes(text.encode("utf-8")) for text in documents]

    rebuilt: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for name, spec in PARTITIONS.items():
        offset = int(spec["offset"])
        requests = int(spec["requests"])
        selected = document_hashes[offset : offset + requests]
        if len(selected) != requests:
            raise RuntimeError(
                f"rebuilt {name} has {len(selected)} documents, expected {requests}"
            )
        if len(set(selected)) != requests:
            raise RuntimeError(f"rebuilt {name} contains duplicate documents")
        overlap = seen.intersection(selected)
        if overlap:
            raise RuntimeError(
                f"frozen partitions are not disjoint; {name} overlaps by {len(overlap)}"
            )
        seen.update(selected)
        observed = hash_lines(selected)
        expected = str(spec["expected_hash_of_document_hashes"])
        if observed != expected:
            raise RuntimeError(
                f"{name} hash-of-document-hashes mismatch: "
                f"observed {observed}, expected {expected}"
            )
        rebuilt[name] = {
            "offset": offset,
            "requests": requests,
            "expected_hash_of_document_hashes": expected,
            "observed_hash_of_document_hashes": observed,
            "document_sha256": selected,
            "freshness_gate": bool(spec["freshness_gate"]),
        }
    return rebuilt


def enforce_freshness(
    partitions: dict[str, dict[str, object]], registry: dict[str, object]
) -> dict[str, object]:
    historical = set(registry["unique_document_hashes"])
    results: dict[str, object] = {}
    for name, partition in partitions.items():
        hashes = set(partition["document_sha256"])
        overlap = sorted(hashes.intersection(historical))
        gated = bool(partition["freshness_gate"])
        results[name] = {
            "is_fresh_against_historical_registry": not overlap,
            "historical_overlap_count": len(overlap),
            "historical_overlap_sha256": overlap,
            "enforced": gated,
        }
        if gated and overlap:
            raise RuntimeError(
                f"{name} is not fresh: {len(overlap)} document hashes occur in "
                "historical data manifests"
            )
    return results


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def build_machine_protocol(
    workspace_root: Path,
    human_protocol_sha256: str,
    prompts_sha256: str,
    registry: dict[str, object],
    partitions: dict[str, dict[str, object]],
    freshness: dict[str, object],
) -> dict[str, object]:
    protocol: dict[str, object] = {
        "schema": SCHEMA,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "FROZEN_BEFORE_FRESH_ROUTE_CAPTURE",
        "evidence_boundary": (
            "teacher-forced request-local logical EP record-cost protocol only; "
            "no serving arrival, continuous batching, backend latency, NIC bytes, "
            "TTFT, TPOT, or P99 claim"
        ),
        "source_integrity": {
            "human_protocol": {
                "path": HUMAN_PROTOCOL_RELATIVE_PATH.as_posix(),
                "expected_sha256": HUMAN_PROTOCOL_SHA256,
                "observed_sha256": human_protocol_sha256,
            },
            "document_builder": {
                "path": PROMPTS_RELATIVE_PATH.as_posix(),
                "callable": "get_wikitext2_documents",
                "expected_sha256": PROMPTS_SHA256,
                "observed_sha256": prompts_sha256,
            },
            "freeze_tool": {
                "path": relative_posix(Path(__file__), workspace_root),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "data": {
            **DATASET,
            "tokenization": {
                "per_model_tokenizer": True,
                "maximum_non_padding_tokens_per_request": 256,
                "same_raw_documents_across_models": True,
            },
            "document_sha256_algorithm": "sha256(UTF-8 normalized article text)",
            "partition_hash_algorithm": (
                "sha256(ordered lowercase document SHA256 values joined by LF, "
                "including one final LF)"
            ),
            "partitions": partitions,
            "partitions_pairwise_disjoint": True,
            "historical_freshness": freshness,
        },
        "historical_exclusion_registry": registry,
        "primary_cells": PRIMARY_CELLS,
        "topology": {
            "ep_size": 8,
            "domains": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "balanced_experts_per_rank_required": True,
            "request_home_assignment": (
                "sort requests by frozen document SHA256, then round-robin ranks 0..7"
            ),
            "all_tokens_of_request_share_home_rank": True,
        },
        "contracts": {
            "C0": "expanded_expert_major_negative_control_exact",
            "C1": "rank_major_unique_owner_primary",
            "C2": "domain_partial_mathematical_stress_secondary_only",
        },
        "representations": {
            "S0": "architecture_only_uniform",
            "S1_R": {
                "name": "request_conditioned_exact_degree",
                "seeds": S1_SEEDS,
                "seed_count": len(S1_SEEDS),
                "required_degree_tv": 0,
                "required_duplicate_expert_count": 0,
            },
            "S2": "request_layer_hyperedge_multiset",
            "S3_W": {
                "name": "windowed_hyperedge_dictionary",
                "primary_window_tokens": 32,
                "secondary_window_tokens": [16, 64, 128],
            },
            "S4": "full_ordered_route_oracle",
        },
        "placement_pool": {
            "placements_per_model": 132,
            "fixed": [
                "contiguous",
                "round_robin",
                "calibration_only_frequency_balanced_LPT",
                "calibration_only_coactivation_aware_balanced",
            ],
            "balanced_random_seeds": PLACEMENT_SEEDS,
            "balanced_random_seed_count": len(PLACEMENT_SEEDS),
            "coactivation_hill_climbing_balanced_swaps": 20000,
            "tie_break": "placement_mapping_sha256_lexicographic",
            "sealed_driven_generation_or_filtering_forbidden": True,
        },
        "statistics": {
            "bootstrap": {
                "method": "paired_request_cluster_bootstrap",
                "resamples": 10000,
                "seed": 2026071899,
                "cluster_unit": "request/article carrying all layers and tokens",
            },
            "multiple_comparison_correction": "Holm over P1/P2 primary comparisons",
            "synthesis_seeds_are_not_independent_samples": True,
        },
        "primary_thresholds": {
            "problem_gate": {
                "required_cells": ["P1", "P2"],
                "S1_R_seed_point_regret_min": 0.05,
                "S1_R_seeds_meeting_point_regret_min": 16,
                "S1_R_total_seeds": 20,
                "S1_R_seed_median_regret_min": 0.05,
                "Holm_adjusted_one_sided_95pct_lower_bound_strictly_gt": 0.0,
            },
            "method_gate": {
                "required_cells": ["P1", "P2"],
                "S3_W_regret_max": 0.02,
                "Holm_adjusted_one_sided_95pct_upper_bound_max": 0.02,
                "kendall_tau_b_one_sided_95pct_lower_bound_min": 0.95,
                "S1_R_minus_S3_W_regret_min": 0.03,
                "S3_W_raw_canonical_size_over_S4_max": 0.70,
            },
            "temporal_secondary_gate": {
                "window_tokens": 32,
                "regret_min": 0.05,
                "confidence_interval_must_exclude": 0.0,
                "required_cells": ["P1", "P2"],
            },
            "GPU_P1_gate": {
                "minimum_gpus": 2,
                "first_backend_operator_latency_regret_min": 0.05,
                "confidence_interval_must_exclude": 0.0,
            },
        },
        "canonical_size_accounting": {
            "expert_id_bits": "ceil(log2(E))",
            "include": [
                "request_boundaries",
                "layer_boundaries",
                "chunk_boundaries",
                "counts",
                "dictionary_or_prototype_ids",
                "seed",
            ],
            "primary": "raw_canonical_packed_bytes",
            "secondary": "zstd_level_3_bytes",
        },
        "run_policy": {
            "calibration_before_sealed": True,
            "sealed_once_only": True,
            "existing_sealed_output_refuses_overwrite": True,
            "promotion_requires_both_primary_cells_to_pass_all_primary_gates": True,
        },
    }
    protocol["machine_protocol_payload_sha256"] = canonical_sha256(protocol)
    return protocol


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()

    # lexists also rejects a broken symlink, which Path.exists() would miss.
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"refusing to overwrite or reuse existing output directory: {output_dir}"
        )

    human_protocol_sha256 = verify_pinned_file(
        workspace_root / HUMAN_PROTOCOL_RELATIVE_PATH,
        HUMAN_PROTOCOL_SHA256,
        "human protocol",
    )
    prompts_sha256 = verify_pinned_file(
        workspace_root / PROMPTS_RELATIVE_PATH,
        PROMPTS_SHA256,
        "document builder",
    )
    get_documents = load_document_builder(workspace_root / PROMPTS_RELATIVE_PATH)
    registry = scan_historical_registry(workspace_root)
    partitions = rebuild_partitions(get_documents)
    freshness = enforce_freshness(partitions, registry)
    registry_after_validation = scan_historical_registry(workspace_root)
    if (
        registry_after_validation["manifest_registry_payload_sha256"]
        != registry["manifest_registry_payload_sha256"]
    ):
        raise RuntimeError(
            "historical data manifests changed while the registry was being frozen"
        )
    protocol = build_machine_protocol(
        workspace_root,
        human_protocol_sha256,
        prompts_sha256,
        registry,
        partitions,
        freshness,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "machine_protocol.json"
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "machine_protocol": str(output_path),
                "machine_protocol_payload_sha256": protocol[
                    "machine_protocol_payload_sha256"
                ],
                "historical_manifest_files": registry["manifest_file_count"],
                "historical_unique_document_hashes": registry[
                    "unique_document_hash_count"
                ],
                "calibration_requests": partitions["calibration"]["requests"],
                "sealed_requests": partitions["sealed"]["requests"],
                "reserve_requests": partitions["reserve"]["requests"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FREEZE_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

#!/usr/bin/env python3
"""Run the frozen SFV2-O1 fresh online semantic-observability gate.

The runner has three deliberately separated phases:

* ``prepare-documents`` selects a document-disjoint 6/2/4 split from the
  pinned WikiText revision after the full historical exclusion union.
* ``dry-run`` verifies that frozen input without importing Torch or producing
  any expert outcome.
* ``run`` captures only dispatch-time state, freezes ``PRE_OUTCOME_LOCK.json``,
  then executes train, validation, freezes the global threshold and test
  admission plan, and only then executes test semantic outcomes once.

This is a single-RTX-5090 expert-stage mechanism gate.  It is not a serving,
EP, NCCL, RDMA, multi-tenant, or paper result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[3]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import prepare_eval_manifest as prepare_source  # noqa: E402

gpu: Any = None
pilot: Any = None
shadow: Any = None


SCHEMA = "semanticfence-online-observability-v1"
DOCUMENT_INPUT_SCHEMA = "semanticfence-online-observability-document-input-v1"
LOCK_SCHEMA = "semanticfence-online-observability-pre-outcome-lock-v1"
COMPLETE_SCHEMA = "semanticfence-online-observability-complete-v1"
ABI = "olmoe_bf16_raw_expert_contribution_m2_v1"
EXPECTED_SELECTED_HASHES = (
    "81360b2d9b7b4f9620408450262fec4321175717e0be382b7f6ffb4e7f9dc825",
    "c15a10b2f6ef1303482384961fecf0c047e8c1417b0873b6bf0dc905784e8b2e",
    "a3de334144fc409f72635d1417c25405e098fc3cf19305d69d1c48504fd5bc35",
    "40be4222da440a8e4b3b5d7aa743488c815be7c9e97a11c2941a6ef1f7130458",
    "ea4dcb6e85548f02010028df8a7e777a9fc0575fb2fd796c13f910522d5189b4",
    "bd3a22bad1b5109337102df223cc266dc6732e05977605e97601ebe50889f50f",
    "5f0472761109683b9ca09fd098e65eb15fdb3f9ccd58d77cfbc6f311411f5e21",
    "629afa7f1f314e98dc0b2251615af7496d4edb1b1d107f338bbb8bf07c95b86b",
    "1f350d8abe3ddc783bc8775c9f95a35f22d8c360ff07282e48c4764375a44bea",
    "bc592f4a9d161b54011e951f944bfbd44cf81cec2d868111e2ac3b00bf345c93",
    "1a680cc66839b645a4fcef541bfa4d6dce2b20fbd077dbb55dd24c35a61dbeb9",
    "54b42b8049a183fc69819a7012d85e0102c77c5ad27af67ebcbfa91e11563866",
)
EXPECTED_ARTICLE_INDICES = (
    1738,
    16175,
    12254,
    23818,
    289,
    10781,
    1484,
    13422,
    15132,
    25816,
    1737,
    28398,
)
SPLIT_COUNTS = {"train": 6, "validation": 2, "test": 4}


class GateError(RuntimeError):
    """The frozen gate cannot be interpreted or executed safely."""


def load_runtime_modules() -> tuple[Any, Any, Any]:
    """Import GPU/executor modules only for the formal run.

    This keeps ``prepare-documents`` usable in the workspace's dataset-only
    Python environment without importing Torch or executor dataclasses.
    """

    global gpu, pilot, shadow
    if gpu is None:
        import gpu_execution as gpu_module
        import run_pilot_5090 as pilot_module
        import run_semantic_oracle_shadow_replay_5090 as shadow_module

        gpu = gpu_module
        pilot = pilot_module
        shadow = shadow_module
    return gpu, pilot, shadow


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise GateError(f"value is not strict canonical JSON: {exc}") from exc
    return (text + suffix).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value, newline=False)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_set_sha256(values: Iterable[str]) -> str:
    unique = sorted(set(map(str, values)))
    return hashlib.sha256("".join(unique).encode("ascii")).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot load JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise GateError(f"{path}:{line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot load JSONL {path}: {exc}") from exc
    return rows


def write_json_exclusive(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise GateError(f"refusing to overwrite artifact: {path}") from exc


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(dict(row)))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise GateError(f"refusing to overwrite artifact: {path}") from exc


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(config)
    required = {
        "schema_version",
        "status",
        "gate",
        "paper_result",
        "base_pilot_config",
        "dataset",
        "document_split",
        "capture",
        "candidate_schedule",
        "feature",
        "certificate",
        "semantic_label",
        "verdict",
    }
    if set(value) != required:
        raise GateError(f"config keys differ: {sorted(set(value) ^ required)}")
    if value["gate"] != "SFV2-O1" or value["paper_result"] is not False:
        raise GateError("config gate/paper boundary differs")
    if value["status"] != "FROZEN":
        raise GateError("config is not frozen")
    dataset = value["dataset"]
    if (
        dataset["revision"] != prepare_source.DATASET_REVISION
        or dataset["selection_salt"] != prepare_source.FIXED_SELECTION_SALT
    ):
        raise GateError("dataset revision or frozen selection salt differs")
    if value["document_split"]["preferred"] != SPLIT_COUNTS:
        raise GateError("preferred split is not frozen 6/2/4")
    candidate = value["candidate_schedule"]
    if (
        int(candidate["rolling_horizon_compatible_arrivals"]) != 8
        or int(candidate["maximum_edges_per_document"]) != 32
        or int(candidate["minimum_test_edges"]) != 64
        or candidate["stream_boundary"] != "document_and_capture_window"
    ):
        raise GateError("candidate schedule contract differs")
    feature = value["feature"]
    if int(feature["projection_dimension"]) != 64 or int(feature["projection_seed"]) != 20260810:
        raise GateError("projection contract differs")
    if value["semantic_label"]["pair_safe"] != "both_endpoints_safe":
        raise GateError("semantic pair label differs")
    return value


def validate_frozen_stack_except_gpu_uuid(
    observed: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the frozen stack exactly, except for the physical GPU UUID.

    A fresh SeetaCloud allocation can expose the same accepted RTX 5090
    software/hardware stack under a different device UUID.  UUID is an
    instance identity, not a stack property.  This helper removes exactly that
    one leaf from defensive copies and compares every remaining semantic key
    and value without normalization or subset matching.  ``stack_digest`` is
    also removed from that comparison only after each side has independently
    proved that the digest exactly binds its own complete stack payload.
    """

    if not isinstance(observed, Mapping) or not isinstance(frozen, Mapping):
        raise GateError("observed/frozen stack must both be mappings")
    if set(observed) != set(frozen):
        raise GateError("live stack top-level keys differ from frozen acceptance")
    if "stack_digest" not in observed:
        raise GateError("live/frozen stack lacks stack_digest")
    observed_gpu = observed.get("gpu")
    frozen_gpu = frozen.get("gpu")
    if not isinstance(observed_gpu, Mapping) or not isinstance(frozen_gpu, Mapping):
        raise GateError("live/frozen stack lacks a GPU mapping")
    if set(observed_gpu) != set(frozen_gpu) or "uuid" not in observed_gpu:
        raise GateError("live GPU identity keys differ from frozen acceptance")
    observed_uuid = observed_gpu.get("uuid")
    frozen_uuid = frozen_gpu.get("uuid")
    if not isinstance(observed_uuid, str) or not observed_uuid:
        raise GateError("live GPU UUID is absent or malformed")
    if not isinstance(frozen_uuid, str) or not frozen_uuid:
        raise GateError("frozen GPU UUID is absent or malformed")

    hex_digits = set("0123456789abcdef")
    observed_digest = observed.get("stack_digest")
    frozen_digest = frozen.get("stack_digest")
    for label, digest in (
        ("live", observed_digest),
        ("frozen", frozen_digest),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or set(digest) - hex_digits
        ):
            raise GateError(f"{label} stack_digest is not a lowercase SHA-256")
        bound_payload = dict(observed if label == "live" else frozen)
        bound_payload.pop("stack_digest")
        if canonical_sha256(bound_payload) != digest:
            raise GateError(f"{label} stack_digest does not bind its own stack")

    observed_without_uuid = dict(observed)
    frozen_without_uuid = dict(frozen)
    observed_without_uuid.pop("stack_digest")
    frozen_without_uuid.pop("stack_digest")
    observed_without_uuid["gpu"] = {
        key: value for key, value in observed_gpu.items() if key != "uuid"
    }
    frozen_without_uuid["gpu"] = {
        key: value for key, value in frozen_gpu.items() if key != "uuid"
    }
    if observed_without_uuid != frozen_without_uuid:
        raise GateError(
            "live GPU/software stack differs from frozen run03 acceptance "
            "outside gpu.uuid"
        )
    return {
        "all_non_uuid_stack_fields_exact": True,
        "ignored_identity_leaf": "gpu.uuid",
        "derived_fields_recomputed_not_directly_compared": ["stack_digest"],
        "frozen_gpu_uuid": frozen_uuid,
        "observed_gpu_uuid": observed_uuid,
        "frozen_stack_digest": frozen_digest,
        "observed_stack_digest": observed_digest,
    }


def _document_hashes(path: Path) -> set[str]:
    values, _ = prepare_source._hashes_from_document_manifest(Path(path))
    return values


def _d10_prompt_hashes(path: Path) -> set[str]:
    value = load_json(path)
    requests = value.get("requests", []) if isinstance(value, dict) else []
    hashes: set[str] = set()
    for index, row in enumerate(requests):
        prompt = str(row.get("prompt", ""))
        declared = str(row.get("prompt_sha256", ""))
        observed = prepare_source.text_sha256(prompt)
        if declared != observed:
            raise GateError(f"D10 prompt {index} hash mismatch")
        hashes.add(observed)
    if len(hashes) != 128:
        raise GateError(f"expected 128 D10 prompt hashes, got {len(hashes)}")
    return hashes


def build_full_exclusion(repo_root: Path, config: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    base, base_report = prepare_source.load_exclusion_hashes(
        prepare_source.default_exclusion_sources(repo_root)
    )
    extra_paths = config["dataset"]["additional_exclusion_sources"]
    semantic_path = repo_root / extra_paths["semanticfence_fresh32"]
    stable_path = repo_root / extra_paths["stablebatch_fresh16"]
    d10_path = repo_root / extra_paths["d10_test_prompts"]
    semantic = _document_hashes(semantic_path)
    stable_values = _document_hashes(stable_path)
    d10 = _d10_prompt_hashes(d10_path)
    train_union = set(base) | semantic | stable_values
    global_union = train_union | d10
    if len(base) != 1137 or len(train_union) != 1185 or len(global_union) != 1313:
        raise GateError(
            "historical exclusion cardinality drift: "
            f"base={len(base)} train={len(train_union)} global={len(global_union)}"
        )
    report = {
        "base_sources": base_report,
        "additional_sources": {
            "semanticfence_fresh32": {
                "path": str(semantic_path.relative_to(repo_root)),
                "file_sha256": sha256_file(semantic_path),
                "unique_hashes": len(semantic),
            },
            "stablebatch_fresh16": {
                "path": str(stable_path.relative_to(repo_root)),
                "file_sha256": sha256_file(stable_path),
                "unique_hashes": len(stable_values),
            },
            "d10_test_prompts": {
                "path": str(d10_path.relative_to(repo_root)),
                "file_sha256": sha256_file(d10_path),
                "unique_hashes": len(d10),
                "unit": "wikitext_test_prompt_not_train_article",
            },
        },
        "base_unique_hashes": len(base),
        "train_article_union_unique_hashes": len(train_union),
        "global_defensive_union_unique_hashes": len(global_union),
        "train_article_union_sha256": ordered_set_sha256(train_union),
        "d10_prompt_set_sha256": ordered_set_sha256(d10),
        "global_defensive_union_sha256": ordered_set_sha256(global_union),
    }
    return global_union, report


def select_fresh_documents(
    candidate_documents: Sequence[str],
    token_lengths: Any,
    *,
    excluded_hashes: set[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: dict[str, tuple[int, str]] = {}
    for article_index, raw_text in enumerate(candidate_documents):
        text = prepare_source.canonical_text(raw_text)
        digest = prepare_source.text_sha256(text)
        candidates.setdefault(digest, (article_index, text))
    eligible = [
        (digest, article_index, text)
        for digest, (article_index, text) in candidates.items()
        if digest not in excluded_hashes
    ]
    qualifying: dict[str, tuple[int, str, int]] = {}
    for start in range(0, len(eligible), prepare_source.TOKENIZER_BATCH_SIZE):
        batch = eligible[start : start + prepare_source.TOKENIZER_BATCH_SIZE]
        lengths = list(token_lengths([row[2] for row in batch]))
        if len(lengths) != len(batch):
            raise GateError("token-length batch cardinality mismatch")
        for (digest, article_index, text), raw_length in zip(batch, lengths):
            length = int(raw_length)
            if length >= int(config["dataset"]["minimum_tokens"]):
                qualifying[digest] = (article_index, text, length)
    ordered = sorted(
        qualifying,
        key=lambda digest: (prepare_source.selection_sha256(digest), digest),
    )
    selected = ordered[:12]
    if len(selected) < 8:
        raise GateError("NOT_EXECUTABLE_INSUFFICIENT_FRESH_DOCUMENTS")
    if tuple(selected) != EXPECTED_SELECTED_HASHES:
        raise GateError("deterministic fresh-12 selection differs from pre-run inventory")
    rows: list[dict[str, Any]] = []
    for index, digest in enumerate(selected):
        split = "train" if index < 6 else "validation" if index < 8 else "test"
        article_index, text, length = qualifying[digest]
        rows.append(
            {
                "document_index": index,
                "logical_split": split,
                "source_article_index": article_index,
                "text_sha256": digest,
                "selection_sha256": prepare_source.selection_sha256(digest),
                "token_length_at_least": length,
                "text": text,
            }
        )
    return rows


def materialize_inventory_selected_documents(
    candidate_documents: Sequence[str],
    token_lengths: Any,
    *,
    excluded_hashes: set[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Materialize the exact fresh-12 already established by the full inventory.

    The exhaustive 19,049-eligible-document scan is bound by its source/set
    hashes and exact expected identities.  This function avoids repeating the
    memory-heavy all-document tokenizer pass; it revalidates full text, article
    index, exclusion membership, selection key, and token support for all 12.
    """

    if tuple(config["dataset"]["expected_selected_text_sha256"]) != EXPECTED_SELECTED_HASHES:
        raise GateError("config expected fresh-12 identities differ")
    selected_texts: list[str] = []
    for article_index, expected_hash in zip(
        EXPECTED_ARTICLE_INDICES, EXPECTED_SELECTED_HASHES
    ):
        if article_index >= len(candidate_documents):
            raise GateError("inventory article index is outside pinned dataset")
        text = prepare_source.canonical_text(candidate_documents[article_index])
        observed = prepare_source.text_sha256(text)
        if observed != expected_hash or observed in excluded_hashes:
            raise GateError("inventory fresh document hash/exclusion differs")
        selected_texts.append(text)
    lengths = list(token_lengths(selected_texts))
    if len(lengths) != 12 or any(
        int(value) < int(config["dataset"]["minimum_tokens"]) for value in lengths
    ):
        raise GateError("inventory fresh document token support differs")
    rows: list[dict[str, Any]] = []
    for index, (article_index, digest, text, length) in enumerate(
        zip(EXPECTED_ARTICLE_INDICES, EXPECTED_SELECTED_HASHES, selected_texts, lengths)
    ):
        split = "train" if index < 6 else "validation" if index < 8 else "test"
        rows.append(
            {
                "document_index": index,
                "logical_split": split,
                "source_article_index": article_index,
                "text_sha256": digest,
                "selection_sha256": prepare_source.selection_sha256(digest),
                "token_length_at_least": int(length),
                "text": text,
            }
        )
    return rows


def prepare_documents(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    config = validate_config(load_json(config_path))
    excluded, exclusion_report = build_full_exclusion(repo_root, config)
    documents, token_lengths, provenance = prepare_source.load_pinned_candidate_stream(
        cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None
    )
    rows = materialize_inventory_selected_documents(
        documents, token_lengths, excluded_hashes=excluded, config=config
    )
    counts = {name: sum(row["logical_split"] == name for row in rows) for name in SPLIT_COUNTS}
    if counts != SPLIT_COUNTS:
        raise GateError(f"fresh split count differs: {counts}")
    value = {
        "schema_version": DOCUMENT_INPUT_SCHEMA,
        "status": "PREPARED_NO_SEMANTIC_OUTCOME",
        "selection_rule": (
            "full pre-run inventory: dedupe canonical full text; exclude global defensive "
            "union; require >=2081 OLMoE tokens; sort by frozen salted full-text hash; "
            "take first12 as 6/2/4; this artifact revalidates the exact inventoried rows"
        ),
        "selection_inventory": {
            "eligible_fresh_train_articles": 19001,
            "expected_article_indices": list(EXPECTED_ARTICLE_INDICES),
            "expected_text_sha256": list(EXPECTED_SELECTED_HASHES),
            "exhaustive_scan_completed_before_semantic_outcomes": True,
        },
        "config_sha256": sha256_file(config_path),
        "dataset": {
            "repo_id": prepare_source.DATASET_REPO_ID,
            "config": prepare_source.DATASET_CONFIG,
            "revision": prepare_source.DATASET_REVISION,
            "split": prepare_source.DATASET_SPLIT,
            **provenance,
        },
        "tokenizer": {
            "repo_id": prepare_source.MODEL_REPO_ID,
            "revision": prepare_source.MODEL_REVISION,
        },
        "exclusions": exclusion_report,
        "split_counts": counts,
        "selected_ordered_hash_sha256": hashlib.sha256(
            ("\n".join(row["text_sha256"] for row in rows) + "\n").encode("ascii")
        ).hexdigest(),
        "documents": rows,
    }
    write_json_exclusive(Path(args.output).resolve(), value)
    print(canonical_json_bytes({
        "output": str(Path(args.output).resolve()),
        "documents": len(rows),
        "split_counts": counts,
        "selected_hashes": [row["text_sha256"] for row in rows],
    }).decode("utf-8"), end="")
    return 0


def validate_document_input(
    value: Mapping[str, Any], *, config_sha256: str | None = None
) -> list[dict[str, Any]]:
    if value.get("schema_version") != DOCUMENT_INPUT_SCHEMA:
        raise GateError("document input schema differs")
    if value.get("status") != "PREPARED_NO_SEMANTIC_OUTCOME":
        raise GateError("document input status differs")
    if config_sha256 is not None and value.get("config_sha256") != config_sha256:
        raise GateError("document input config hash differs")
    rows = [dict(row) for row in value.get("documents", [])]
    if len(rows) < 8:
        raise GateError("NOT_EXECUTABLE_INSUFFICIENT_FRESH_DOCUMENTS")
    if len(rows) != 12:
        raise GateError("preferred fresh-12 split was not materialized")
    observed = tuple(str(row.get("text_sha256")) for row in rows)
    if observed != EXPECTED_SELECTED_HASHES or len(set(observed)) != len(observed):
        raise GateError("document identities differ or repeat")
    counts = {name: sum(row.get("logical_split") == name for row in rows) for name in SPLIT_COUNTS}
    if counts != SPLIT_COUNTS:
        raise GateError(f"document split differs: {counts}")
    for index, row in enumerate(rows):
        if int(row.get("document_index", -1)) != index:
            raise GateError("document indices are not frozen order")
        if prepare_source.text_sha256(str(row.get("text", ""))) != row["text_sha256"]:
            raise GateError("document full-text hash mismatch")
    return rows


def dry_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = validate_config(load_json(config_path))
    input_path = Path(args.document_input).resolve()
    value = load_json(input_path)
    rows = validate_document_input(value, config_sha256=sha256_file(config_path))
    report = {
        "schema_version": "semanticfence-online-observability-dry-run-v1",
        "status": "PASS_NO_TORCH_NO_GPU_NO_SEMANTIC_OUTCOME",
        "config_sha256": sha256_file(config_path),
        "document_input_sha256": sha256_file(input_path),
        "selected_hashes": [row["text_sha256"] for row in rows],
        "split_counts": {name: sum(row["logical_split"] == name for row in rows) for name in SPLIT_COUNTS},
        "rolling_horizon": int(config["candidate_schedule"]["rolling_horizon_compatible_arrivals"]),
        "stream_boundary": config["candidate_schedule"]["stream_boundary"],
        "test_minimum_edges": int(config["candidate_schedule"]["minimum_test_edges"]),
    }
    write_json_exclusive(Path(args.output).resolve(), report)
    print(canonical_json_bytes(report).decode("utf-8"), end="")
    return 0


def _row_arrival_key(row: Any) -> tuple[int, int, int, int, int, str]:
    record = row.record
    return (
        int(record.document_index),
        int(record.offset),
        int(record.layer),
        int(record.token_position),
        int(record.route_rank),
        str(row.row_id),
    )


def _edge_sample_key(
    document_sha256: str,
    left: Any,
    right: Any,
    *,
    salt: str,
) -> str:
    payload = {
        "salt": salt,
        "document_sha256": str(document_sha256),
        "stream": {
            "window_id": str(left.context.window_id),
            "layer": int(left.record.layer),
            "expert_id": int(left.record.expert_id),
            "abi": ABI,
        },
        "row_ids": sorted([str(left.row_id), str(right.row_id)]),
    }
    return canonical_sha256(payload)


def build_candidate_schedule(
    rows: Sequence[Any],
    document_splits: Mapping[str, str],
    *,
    rolling_horizon: int = 8,
    maximum_edges_per_document: int = 32,
    sample_salt: str = "semanticfence-sfv2-o1-natural-edge-sample-v1",
) -> list[dict[str, Any]]:
    """Build the frozen natural graph without crossing capture windows.

    ``rolling_horizon`` is measured in prior compatible arrivals inside one
    ``(document, window, layer, expert, ABI)`` stream.  The two captured
    offsets are independent requests/windows; the queue is reset between them.
    """

    if rolling_horizon != 8 or maximum_edges_per_document != 32:
        raise GateError("candidate W/cap differs from frozen contract")
    ordered = sorted(rows, key=_row_arrival_key)
    global_arrival = {str(row.row_id): index for index, row in enumerate(ordered)}
    streams: dict[tuple[str, str, int, int, str], list[Any]] = defaultdict(list)
    for row in ordered:
        record = row.record
        split = document_splits.get(str(record.document_sha256))
        if split not in SPLIT_COUNTS:
            raise GateError("routed row belongs to an unknown document")
        key = (
            str(record.document_sha256),
            str(row.context.window_id),
            int(record.layer),
            int(record.expert_id),
            ABI,
        )
        streams[key].append(row)

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    undirected: set[frozenset[str]] = set()
    for stream_key in sorted(streams):
        stream_rows = streams[stream_key]
        for closing_index, right in enumerate(stream_rows):
            opening_start = max(0, closing_index - rolling_horizon)
            for opening_index in range(opening_start, closing_index):
                left = stream_rows[opening_index]
                endpoints = frozenset((str(left.row_id), str(right.row_id)))
                if len(endpoints) != 2 or endpoints in undirected:
                    raise GateError("candidate graph has a loop or duplicate edge")
                undirected.add(endpoints)
                document_sha256, window_id, layer, expert_id, abi = stream_key
                edge_payload = {
                    "document_sha256": document_sha256,
                    "window_id": window_id,
                    "layer": layer,
                    "expert_id": expert_id,
                    "abi": abi,
                    "row_ids": [str(left.row_id), str(right.row_id)],
                    "compatible_arrival_indices": [opening_index, closing_index],
                }
                by_document[document_sha256].append(
                    {
                        **edge_payload,
                        "edge_id": canonical_sha256(edge_payload),
                        "sample_key": _edge_sample_key(
                            document_sha256, left, right, salt=sample_salt
                        ),
                        "logical_split": document_splits[document_sha256],
                        "global_arrival_indices": [
                            global_arrival[str(left.row_id)],
                            global_arrival[str(right.row_id)],
                        ],
                        "row_records": [
                            left.record.identity_payload(),
                            right.record.identity_payload(),
                        ],
                        "endpoint_context": [
                            {
                                "window_id": str(left.context.window_id),
                                "absolute_token_position": int(
                                    left.context.absolute_token_position
                                ),
                                "routing_weight": float(left.context.routing_weight),
                            },
                            {
                                "window_id": str(right.context.window_id),
                                "absolute_token_position": int(
                                    right.context.absolute_token_position
                                ),
                                "routing_weight": float(right.context.routing_weight),
                            },
                        ],
                    }
                )

    selected: list[dict[str, Any]] = []
    for document_sha256 in sorted(by_document):
        candidates = sorted(
            by_document[document_sha256],
            key=lambda edge: (str(edge["sample_key"]), str(edge["edge_id"])),
        )[:maximum_edges_per_document]
        selected.extend(candidates)
    selected.sort(
        key=lambda edge: (
            int(edge["row_records"][0]["document_index"]),
            int(edge["global_arrival_indices"][1]),
            int(edge["global_arrival_indices"][0]),
            str(edge["edge_id"]),
        )
    )
    for index, edge in enumerate(selected):
        edge["schedule_index"] = index
        edge["dispatch_order"] = index
        edge["deadline_policy"] = "close_after_8_compatible_arrivals_within_window"
    ids = [str(edge["edge_id"]) for edge in selected]
    if len(ids) != len(set(ids)):
        raise GateError("sampled candidate edge IDs repeat")
    return selected


def validate_candidate_schedule(schedule: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, int] = defaultdict(int)
    by_document: dict[str, int] = defaultdict(int)
    for index, edge in enumerate(schedule):
        if int(edge["schedule_index"]) != index:
            raise GateError("candidate schedule order/index differs")
        if edge["logical_split"] not in SPLIT_COUNTS:
            raise GateError("candidate split differs")
        if len(set(map(str, edge["row_ids"]))) != 2:
            raise GateError("candidate edge has a self-loop")
        if len(set(str(context["window_id"]) for context in edge["endpoint_context"])) != 1:
            raise GateError("candidate edge crosses a capture window")
        opening, closing = map(int, edge["compatible_arrival_indices"])
        if not (1 <= closing - opening <= 8):
            raise GateError("candidate edge violates W=8")
        by_split[str(edge["logical_split"])] += 1
        by_document[str(edge["document_sha256"])] += 1
    if any(value > 32 for value in by_document.values()):
        raise GateError("candidate document cap exceeds 32")
    return {
        "edges": len(schedule),
        "edges_by_split": dict(sorted(by_split.items())),
        "edges_by_document": dict(sorted(by_document.items())),
        "unique_vertices": len(
            {str(row_id) for edge in schedule for row_id in edge["row_ids"]}
        ),
    }


def _capture_by_window(captures: Sequence[Any]) -> dict[str, Any]:
    result = {str(capture.window_id): capture for capture in captures}
    if len(result) != len(captures):
        raise GateError("capture window IDs repeat")
    return result


def make_projection(hidden_size: int, dimension: int, seed: int) -> Any:
    import torch

    if hidden_size != 2048 or dimension != 64 or seed != 20260810:
        raise GateError("projection shape/seed differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return (
        torch.randn(hidden_size, dimension, dtype=torch.float32, generator=generator)
        / math.sqrt(float(dimension))
    ).contiguous()


SCALAR_FIELDS = (
    "rms",
    "l2",
    "mean_abs",
    "max_abs",
    "std",
    "router_top1_top2_probability_margin",
    "selected_expert_gate_weight",
    "layer",
    "expert_id",
    "abi_numeric",
    "absolute_token_position",
    "request_phase",
)


def raw_online_feature(
    row: Any, capture: Any, projection: Any, *, hidden_override: Any | None = None
) -> list[float]:
    import torch

    source_hidden = row.tensor if hidden_override is None else hidden_override
    hidden = source_hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if hidden.ndim != 1 or int(hidden.numel()) != int(projection.shape[0]):
        raise GateError("hidden/projection shape differs")
    projected = torch.matmul(hidden, projection)
    record = row.record
    weights = capture.routing_weights[
        int(record.layer), int(record.token_position)
    ].to(dtype=torch.float32)
    if int(weights.numel()) < 2:
        raise GateError("router margin requires top2")
    scalars = [
        float(torch.sqrt(torch.mean(hidden * hidden)).item()),
        float(torch.linalg.vector_norm(hidden).item()),
        float(torch.mean(torch.abs(hidden)).item()),
        float(torch.max(torch.abs(hidden)).item()),
        float(torch.std(hidden, correction=0).item()),
        float((weights[0] - weights[1]).item()),
        float(row.context.routing_weight),
        float(record.layer),
        float(record.expert_id),
        1.0,
        float(row.context.absolute_token_position),
        0.0,
    ]
    result = [float(value) for value in projected.tolist()] + scalars
    if len(result) != 64 + len(SCALAR_FIELDS) or not all(math.isfinite(v) for v in result):
        raise GateError("online feature is nonfinite or wrong width")
    return result


def normalize_features(
    raw_by_row: Mapping[str, Sequence[float]], train_row_ids: Iterable[str]
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    import torch

    train_ids = sorted(set(map(str, train_row_ids)))
    if not train_ids or any(row_id not in raw_by_row for row_id in train_ids):
        raise GateError("train normalization rows are absent")
    train = torch.tensor([raw_by_row[row_id] for row_id in train_ids], dtype=torch.float64)
    mean = train.mean(dim=0)
    std = train.std(dim=0, correction=0)
    floor = torch.full_like(std, 1e-12)
    scale = torch.maximum(std, floor)
    result: dict[str, list[float]] = {}
    for row_id, raw in raw_by_row.items():
        vector = (torch.tensor(raw, dtype=torch.float64) - mean) / scale
        values = [float(value) for value in vector.tolist()]
        if not all(math.isfinite(value) for value in values):
            raise GateError("normalized feature is nonfinite")
        result[str(row_id)] = values
    manifest = {
        "fit_scope": "unique_train_candidate_endpoints_only",
        "train_row_count": len(train_ids),
        "dimension": int(train.shape[1]),
        "population_mean": [float(value) for value in mean.tolist()],
        "population_std": [float(value) for value in std.tolist()],
        "applied_scale_max_std_floor_1e-12": [float(value) for value in scale.tolist()],
        "train_row_ids_sha256": hashlib.sha256(
            ("\n".join(train_ids) + "\n").encode("ascii")
        ).hexdigest(),
    }
    return result, manifest


def build_pre_outcome_features(
    schedule: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Any],
    captures_by_window: Mapping[str, Any],
    projection: Any,
) -> tuple[list[dict[str, Any]], dict[str, list[float]], dict[str, Any]]:
    identity: dict[str, dict[str, Any]] = {}
    for edge in schedule:
        for row_id, record, context in zip(
            edge["row_ids"], edge["row_records"], edge["endpoint_context"]
        ):
            value = {
                "logical_split": edge["logical_split"],
                "document_sha256": edge["document_sha256"],
                "row_record": record,
                "window_id": context["window_id"],
                "cell": [int(edge["layer"]), int(edge["expert_id"]), ABI],
            }
            existing = identity.setdefault(str(row_id), value)
            if existing != value:
                raise GateError("candidate endpoint identity changes across edges")
    raw_by_row: dict[str, list[float]] = {}
    for row_id in sorted(identity):
        row = rows_by_id[row_id]
        capture = captures_by_window[str(identity[row_id]["window_id"])]
        raw_by_row[row_id] = raw_online_feature(row, capture, projection)
    train_ids = [
        row_id for row_id, value in identity.items() if value["logical_split"] == "train"
    ]
    normalized, normalization = normalize_features(raw_by_row, train_ids)
    records = [
        {
            "row_id": row_id,
            **identity[row_id],
            "feature_vector": normalized[row_id],
            "feature_vector_sha256": canonical_sha256(normalized[row_id]),
        }
        for row_id in sorted(identity)
    ]
    return records, normalized, normalization


def conservative_row_labels(
    edge_results: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    observations: dict[str, list[bool]] = defaultdict(list)
    for edge in edge_results:
        for endpoint in edge["endpoints"]:
            observations[str(endpoint["row_id"])].append(bool(endpoint["semantic_safe"]))
    result = {row_id: all(flags) for row_id, flags in observations.items()}
    if any(not flags for flags in observations.values()):
        raise GateError("endpoint label aggregation is empty")
    return result


def pair_semantic_safe(endpoints: Sequence[Mapping[str, Any]]) -> bool:
    if len(endpoints) != 2:
        raise GateError("M2 pair must have exactly two endpoint labels")
    return all(bool(endpoint["semantic_safe"]) for endpoint in endpoints)


def build_witness_bank(
    train_results: Sequence[Mapping[str, Any]],
    features: Mapping[str, Sequence[float]],
    row_cells: Mapping[str, tuple[int, int, str]],
) -> tuple[dict[tuple[int, int, str], dict[str, list[str]]], dict[str, Any]]:
    labels = conservative_row_labels(train_results)
    banks: dict[tuple[int, int, str], dict[str, list[str]]] = defaultdict(
        lambda: {"safe": [], "unsafe": []}
    )
    for row_id in sorted(labels):
        if row_id not in features or row_id not in row_cells:
            raise GateError("train label lacks frozen feature/cell")
        bucket = "safe" if labels[row_id] else "unsafe"
        banks[row_cells[row_id]][bucket].append(row_id)
    cells = []
    for cell in sorted(banks):
        safe_ids = sorted(banks[cell]["safe"])
        unsafe_ids = sorted(banks[cell]["unsafe"])
        if set(safe_ids) & set(unsafe_ids):
            raise GateError("one train row entered both witness banks")
        cells.append(
            {
                "cell": list(cell),
                "safe_row_ids": safe_ids,
                "unsafe_row_ids": unsafe_ids,
                "safe_count": len(safe_ids),
                "unsafe_count": len(unsafe_ids),
                "safe_feature_digest": canonical_sha256(
                    [features[row_id] for row_id in safe_ids]
                ),
                "unsafe_feature_digest": canonical_sha256(
                    [features[row_id] for row_id in unsafe_ids]
                ),
            }
        )
    manifest = {
        "schema_version": "semanticfence-online-witness-bank-v1",
        "label_aggregation": "unique_row_safe_iff_all_incident_train_endpoint_observations_safe",
        "cell_key": ["layer", "expert_id", "abi"],
        "safe_rows": sum(labels.values()),
        "unsafe_rows": len(labels) - sum(labels.values()),
        "unique_rows": len(labels),
        "cells": cells,
    }
    return dict(banks), manifest


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise GateError("witness feature dimensions differ")
    value = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
    if not math.isfinite(value):
        raise GateError("witness distance is nonfinite")
    return value


def witness_score(
    row_id: str,
    *,
    features: Mapping[str, Sequence[float]],
    row_cells: Mapping[str, tuple[int, int, str]],
    banks: Mapping[tuple[int, int, str], Mapping[str, Sequence[str]]],
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    cell = row_cells[str(row_id)]
    bank = banks.get(cell)
    if bank is None or not bank.get("safe") or not bank.get("unsafe"):
        return {
            "row_id": str(row_id),
            "cell": list(cell),
            "eligible": False,
            "score": None,
            "d_safe": None,
            "d_unsafe": None,
            "abstention_reason": "MISSING_SAFE_OR_UNSAFE_CELL_BANK",
        }
    vector = features[str(row_id)]
    d_safe = min(_euclidean(vector, features[str(other)]) for other in bank["safe"])
    d_unsafe = min(_euclidean(vector, features[str(other)]) for other in bank["unsafe"])
    score = math.log((d_unsafe + epsilon) / (d_safe + epsilon))
    if not math.isfinite(score):
        raise GateError("witness score is nonfinite")
    return {
        "row_id": str(row_id),
        "cell": list(cell),
        "eligible": True,
        "score": score,
        "d_safe": d_safe,
        "d_unsafe": d_unsafe,
        "abstention_reason": None,
    }


def rolling_greedy_matching(
    schedule: Sequence[Mapping[str, Any]], admitted_rows: set[str]
) -> list[dict[str, Any]]:
    matched: set[str] = set()
    selected: list[dict[str, Any]] = []
    ordered = sorted(
        schedule,
        key=lambda edge: (
            int(edge["row_records"][0]["document_index"]),
            int(edge["global_arrival_indices"][1]),
            int(edge["global_arrival_indices"][0]),
            str(edge["edge_id"]),
        ),
    )
    for edge in ordered:
        left, right = map(str, edge["row_ids"])
        if left not in admitted_rows or right not in admitted_rows:
            continue
        if left in matched or right in matched:
            continue
        matched.update((left, right))
        selected.append(
            {"edge_id": str(edge["edge_id"]), "row_ids": [left, right]}
        )
    return selected


def general_maximum_matching(
    schedule: Sequence[Mapping[str, Any]],
    edge_safe: Mapping[str, bool],
) -> dict[str, Any]:
    try:
        import networkx as nx
    except ImportError as exc:
        raise GateError("networkx is required for general-graph matching") from exc

    vertices = sorted({str(row_id) for edge in schedule for row_id in edge["row_ids"]})
    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    edge_by_pair: dict[frozenset[str], str] = {}
    safe_edges = 0
    for edge in sorted(schedule, key=lambda row: str(row["edge_id"])):
        edge_id = str(edge["edge_id"])
        left, right = map(str, edge["row_ids"])
        key = frozenset((left, right))
        if len(key) != 2 or key in edge_by_pair:
            raise GateError("matching graph has loop/duplicate")
        edge_by_pair[key] = edge_id
        if bool(edge_safe.get(edge_id, False)):
            graph.add_edge(left, right)
            safe_edges += 1
    raw = nx.max_weight_matching(graph, maxcardinality=True)
    pairs = sorted(tuple(sorted(map(str, pair))) for pair in raw)
    matching = [
        {"row_ids": list(pair), "edge_id": edge_by_pair[frozenset(pair)]}
        for pair in pairs
    ]
    return {
        "algorithm": "networkx_general_graph_blossom_max_weight_matching_maxcardinality",
        "networkx_version": nx.__version__,
        "candidate_edges": len(schedule),
        "safe_edges": safe_edges,
        "unique_vertices": len(vertices),
        "matching_edges": len(matching),
        "covered_vertices": 2 * len(matching),
        "row_coverage": 2 * len(matching) / len(vertices) if vertices else 0.0,
        "pair_slot_coverage": (
            len(matching) / (len(vertices) // 2) if len(vertices) >= 2 else 0.0
        ),
        "matching": matching,
    }


def projected_cost(
    *,
    vertices: int,
    pairs: int,
    c1_ms: float,
    c2_ms: float,
    online_overhead_ms: float = 0.0,
) -> dict[str, float | int]:
    if vertices < 2 * pairs or min(c1_ms, c2_ms) <= 0 or online_overhead_ms < 0:
        raise GateError("cost projection inputs differ")
    baseline = vertices * c1_ms
    gross = pairs * c2_ms + (vertices - 2 * pairs) * c1_ms
    net = gross + online_overhead_ms
    return {
        "vertices": vertices,
        "pairs": pairs,
        "c1_ms": c1_ms,
        "c2_ms": c2_ms,
        "all_m1_ms": baseline,
        "gross_runtime_ms": gross,
        "online_overhead_ms": online_overhead_ms,
        "net_runtime_ms": net,
        "gross_saved_fraction": (baseline - gross) / baseline if baseline else 0.0,
        "net_saved_fraction": (baseline - net) / baseline if baseline else 0.0,
    }


def _threshold_admitted(
    scores: Mapping[str, Mapping[str, Any]], threshold: float | None
) -> set[str]:
    if threshold is None:
        return set()
    return {
        row_id
        for row_id, value in scores.items()
        if bool(value["eligible"]) and float(value["score"]) >= float(threshold)
    }


def select_validation_threshold(
    validation_schedule: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    conservative_labels: Mapping[str, bool],
    *,
    c1_ms: float,
    c2_ms: float,
    fixed_online_overhead_ms: float,
) -> dict[str, Any]:
    """Select one global validation-only fail-closed threshold."""

    vertices = sorted(
        {str(row_id) for edge in validation_schedule for row_id in edge["row_ids"]}
    )
    if set(vertices) != set(conservative_labels):
        raise GateError("validation label vertex set differs")
    finite = sorted(
        {
            float(value["score"])
            for value in scores.values()
            if bool(value["eligible"])
        },
        reverse=True,
    )
    candidates: list[float | None] = [None, *finite]
    evaluations: list[dict[str, Any]] = []
    for threshold in candidates:
        admitted = _threshold_admitted(scores, threshold)
        unsafe = sum(not bool(conservative_labels[row_id]) for row_id in admitted)
        matching = rolling_greedy_matching(validation_schedule, admitted)
        cost = projected_cost(
            vertices=len(vertices),
            pairs=len(matching),
            c1_ms=c1_ms,
            c2_ms=c2_ms,
            online_overhead_ms=fixed_online_overhead_ms,
        )
        evaluations.append(
            {
                "threshold_mode": (
                    "ABSTAIN_ALL_POSITIVE_INFINITY" if threshold is None else "FINITE"
                ),
                "threshold_value": threshold,
                "admitted_endpoints": len(admitted),
                "unsafe_admitted_endpoints": unsafe,
                "greedy_matched_pairs": len(matching),
                "projected_saving": float(cost["net_saved_fraction"]),
            }
        )
    feasible = [row for row in evaluations if row["unsafe_admitted_endpoints"] == 0]
    if not feasible:
        raise GateError("validation has no zero-unsafe threshold")

    def tie_threshold(row: Mapping[str, Any]) -> float:
        return math.inf if row["threshold_value"] is None else float(row["threshold_value"])

    chosen = max(
        feasible,
        key=lambda row: (float(row["projected_saving"]), tie_threshold(row)),
    )
    value = chosen["threshold_value"]
    admitted = _threshold_admitted(scores, None if value is None else float(value))
    matching = rolling_greedy_matching(validation_schedule, admitted)
    return {
        "schema_version": "semanticfence-online-validation-threshold-v1",
        "selection_scope": "validation_only",
        "constraint": "zero_unsafe_admitted_unique_endpoints",
        "objective": "maximize_net_projected_saving",
        "tie_break": "higher_threshold",
        "threshold": {
            "mode": chosen["threshold_mode"],
            "value": value,
            "comparison": "score_greater_than_or_equal" if value is not None else None,
        },
        "cost_inputs": {
            "c1_ms": c1_ms,
            "c2_ms": c2_ms,
            "fixed_online_overhead_ms": fixed_online_overhead_ms,
            "overhead_is_constant_across_thresholds": True,
        },
        "chosen_metrics": dict(chosen),
        "chosen_matching": matching,
        "candidate_evaluations": evaluations,
        "validation_scores": [scores[row_id] for row_id in sorted(scores)],
    }


def threshold_value(threshold_artifact: Mapping[str, Any]) -> float | None:
    value = threshold_artifact["threshold"]
    if value["mode"] == "ABSTAIN_ALL_POSITIVE_INFINITY":
        if value["value"] is not None:
            raise GateError("abstain-all threshold must encode null")
        return None
    observed = float(value["value"])
    if not math.isfinite(observed):
        raise GateError("finite threshold is nonfinite")
    return observed


def risk_coverage_diagnostics(
    scores: Mapping[str, Mapping[str, Any]], labels: Mapping[str, bool]
) -> dict[str, Any]:
    total = len(labels)
    groups: dict[float, list[str]] = defaultdict(list)
    for row_id, value in scores.items():
        if bool(value["eligible"]):
            groups[float(value["score"])].append(row_id)
    admitted: list[str] = []
    curve = [
        {
            "threshold": None,
            "coverage": 0.0,
            "risk": 0.0,
            "admitted": 0,
            "unsafe": 0,
        }
    ]
    for score in sorted(groups, reverse=True):
        admitted.extend(sorted(groups[score]))
        unsafe = sum(not bool(labels[row_id]) for row_id in admitted)
        curve.append(
            {
                "threshold": score,
                "coverage": len(admitted) / total if total else 0.0,
                "risk": unsafe / len(admitted),
                "admitted": len(admitted),
                "unsafe": unsafe,
            }
        )

    coverage_at: dict[str, float] = {}
    for name, budget in (("0pct", 0.0), ("1pct", 0.01), ("5pct", 0.05)):
        qualifying = [
            row
            for row in curve
            if (row["unsafe"] == 0 if budget == 0.0 else row["risk"] <= budget)
        ]
        coverage_at[name] = max(float(row["coverage"]) for row in qualifying)

    scorable = sorted(
        (float(value["score"]), bool(labels[row_id]), row_id)
        for row_id, value in scores.items()
        if bool(value["eligible"])
    )
    positives = sum(label for _, label, _ in scorable)
    negatives = len(scorable) - positives
    auroc: float | None
    auroc_reason: str | None = None
    if positives == 0 or negatives == 0:
        auroc = None
        auroc_reason = "SCORABLE_CLASS_DEGENERATE"
    else:
        favorable = 0.0
        for positive_score, positive, _ in scorable:
            if not positive:
                continue
            for negative_score, negative_label, _ in scorable:
                if negative_label:
                    continue
                favorable += (
                    1.0
                    if positive_score > negative_score
                    else 0.5 if positive_score == negative_score else 0.0
                )
        auroc = favorable / (positives * negatives)

    average_precision: float | None
    ap_reason: str | None = None
    if positives == 0:
        average_precision = None
        ap_reason = "NO_SCORABLE_SAFE_POSITIVE"
    else:
        descending = sorted(scorable, key=lambda row: (-row[0], row[2]))
        seen = 0
        true_positive = 0
        precision_sum = 0.0
        index = 0
        while index < len(descending):
            score = descending[index][0]
            group: list[tuple[float, bool, str]] = []
            while index < len(descending) and descending[index][0] == score:
                group.append(descending[index])
                index += 1
            seen += len(group)
            group_positive = sum(label for _, label, _ in group)
            true_positive += group_positive
            precision_sum += group_positive * (true_positive / seen)
        average_precision = precision_sum / positives

    return {
        "unit": "unique_test_row_conservative_incident_label",
        "scorable_rows": len(scorable),
        "total_rows": total,
        "scorable_coverage": len(scorable) / total if total else 0.0,
        "risk_coverage_curve": curve,
        "coverage_at_risk": coverage_at,
        "auroc": auroc,
        "auroc_reason": auroc_reason,
        "auprc_average_precision": average_precision,
        "auprc_reason": ap_reason,
    }


def decide_verdict(
    oracle: Mapping[str, Any], certificate: Mapping[str, Any]
) -> str:
    natural_saving = float(oracle["cost_projection"]["gross_saved_fraction"])
    natural_coverage = float(oracle["matching"]["row_coverage"])
    if (
        natural_saving < 0.05
        or natural_coverage < 0.05
        or int(oracle["matching"]["matching_edges"]) == 0
    ):
        return "NO_GO_NATURAL_SEMANTIC_HEADLINE"
    go = (
        int(certificate["unsafe_admissible_candidate_edges"]) == 0
        and int(certificate["unsafe_greedy_executed_pairs"]) == 0
        and int(certificate["greedy_executed_pairs"]) >= 16
        and float(certificate["admitted_row_coverage"]) >= 0.05
        and int(certificate["positive_action_documents"]) >= 2
        and float(certificate["cost_projection"]["net_saved_fraction"]) > 0.0
    )
    return "GO_SEMANTIC_WITNESS_GATE" if go else "PIVOT_TO_SHADOW_VERIFY"


def _stable_expert_output(expert: Any, batch: Any, repeats: int) -> tuple[list[Any], list[str]]:
    import torch

    outputs: list[Any] = []
    hashes: list[list[str]] = []
    with torch.inference_mode():
        for _ in range(repeats):
            value = expert(batch).detach().cpu().contiguous().clone()
            if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all().item()):
                raise GateError("expert side output is invalid")
            outputs.append(value)
            hashes.append([shadow.stable.tensor_sha256(row) for row in value])
    if len({tuple(row) for row in hashes}) != 1:
        raise GateError("expert side output is not repeat-stable")
    return [row.detach().clone() for row in outputs[0]], hashes[0]


def _semantic_endpoint_call(
    edge: Mapping[str, Any],
    endpoint_index: int,
    capture: Any,
) -> dict[str, Any]:
    other = 1 - endpoint_index
    record = edge["row_records"][endpoint_index]
    return {
        "call_index": int(edge["schedule_index"]),
        "pair_call_index": int(edge["schedule_index"]),
        "endpoint_index": endpoint_index,
        "layer": int(edge["layer"]),
        "expert_id": int(edge["expert_id"]),
        "focal_row_id": str(edge["row_ids"][endpoint_index]),
        "focal_baseline_label": "UNLABELED_PRE_OUTCOME",
        "companion_row_id": str(edge["row_ids"][other]),
        "resolved_companion_baseline_label": "UNLABELED_PRE_OUTCOME",
        "intervention_kind": "fresh_natural_candidate_m2",
        "target_row_record": record,
        "target_topk_rank_zero_based": int(record["route_rank"]) - 1,
        "window_token_ids": list(map(int, capture.window_token_ids)),
        "window_id": str(capture.window_id),
        "_capture": capture,
    }


def execute_semantic_split(
    model: Any,
    schedule: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Any],
    captures_by_window: Mapping[str, Any],
    *,
    pre_outcome_lock: Path,
    threshold_artifact: Path | None = None,
) -> list[dict[str, Any]]:
    """Execute one split; test additionally requires the frozen threshold file."""

    import torch

    if not Path(pre_outcome_lock).is_file():
        raise GateError("M1/M2 outcome attempted before PRE_OUTCOME_LOCK")
    split_names = {str(edge["logical_split"]) for edge in schedule}
    if len(split_names) != 1:
        raise GateError("semantic split execution mixes logical splits")
    split = next(iter(split_names))
    if split == "test" and (
        threshold_artifact is None or not Path(threshold_artifact).is_file()
    ):
        raise GateError("test outcome attempted before frozen threshold/admission plan")
    if split == "test":
        plan_path = Path(threshold_artifact)
        plan = load_json(plan_path)
        threshold_path = plan_path.parent / "VALIDATION_THRESHOLD.json"
        if (
            plan.get("status") != "FROZEN_BEFORE_FIRST_TEST_SEMANTIC_OUTCOME"
            or not threshold_path.is_file()
            or plan.get("validation_threshold_sha256") != sha256_file(threshold_path)
        ):
            raise GateError("test admission plan/threshold binding differs")
        planned_edges = {
            str(row["edge_id"]) for row in schedule
        }
        if not set(map(str, plan.get("candidate_admissible_edge_ids", []))).issubset(
            planned_edges
        ):
            raise GateError("test admission plan contains an unknown edge")

    side_repeats = int(shadow.SIDE_REPEATS)
    unique_rows = sorted({str(row_id) for edge in schedule for row_id in edge["row_ids"]})
    m1_by_row: dict[str, Any] = {}
    m1_hashes: dict[str, str] = {}
    for row_id in unique_rows:
        row = rows_by_id[row_id]
        record = row.record
        expert = model.model.layers[int(record.layer)].mlp.experts[int(record.expert_id)]
        batch = row.tensor.to(device="cuda", dtype=torch.bfloat16).reshape(1, -1)
        outputs, hashes = _stable_expert_output(expert, batch, side_repeats)
        m1_by_row[row_id] = outputs[0]
        m1_hashes[row_id] = hashes[0]

    m2_by_edge: dict[str, list[Any]] = {}
    m2_hashes: dict[str, list[str]] = {}
    for edge in schedule:
        edge_id = str(edge["edge_id"])
        materialized = [rows_by_id[str(row_id)] for row_id in edge["row_ids"]]
        batch = torch.stack(
            [row.tensor.to(device="cuda", dtype=torch.bfloat16) for row in materialized]
        )
        expert = model.model.layers[int(edge["layer"])].mlp.experts[int(edge["expert_id"])]
        outputs, hashes = _stable_expert_output(expert, batch, side_repeats)
        if len(outputs) != 2:
            raise GateError("M2 output cardinality differs")
        m2_by_edge[edge_id] = outputs
        m2_hashes[edge_id] = hashes

    baseline_by_row: dict[str, Mapping[str, Any]] = {}
    public_baseline_by_row: dict[str, Mapping[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for edge in schedule:
        edge_id = str(edge["edge_id"])
        endpoint_results: list[dict[str, Any]] = []
        for endpoint_index, row_id_raw in enumerate(edge["row_ids"]):
            row_id = str(row_id_raw)
            context = edge["endpoint_context"][endpoint_index]
            capture = captures_by_window[str(context["window_id"])]
            call = _semantic_endpoint_call(edge, endpoint_index, capture)
            if row_id not in baseline_by_row:
                baseline, public_baseline = shadow._native_and_noop(model, call)
                baseline_by_row[row_id] = baseline
                public_baseline_by_row[row_id] = public_baseline
            intervention = shadow._run_intervention(
                model,
                call,
                m1_by_row[row_id].to(device="cuda", dtype=torch.bfloat16),
                m2_by_edge[edge_id][endpoint_index].to(
                    device="cuda", dtype=torch.bfloat16
                ),
                baseline_by_row[row_id],
            )
            safe = bool(intervention["route_topk_semantic_safe"])
            endpoint_results.append(
                {
                    "row_id": row_id,
                    "row_record": edge["row_records"][endpoint_index],
                    "window_id": str(context["window_id"]),
                    "independent_m1_output_sha256": m1_hashes[row_id],
                    "paired_m2_output_sha256": m2_hashes[edge_id][endpoint_index],
                    "native_noop": public_baseline_by_row[row_id],
                    "semantic_safe": safe,
                    "route_topk_changed": bool(
                        intervention["route_delta"]["any_ordered_topk_change"]
                    ),
                    "route_delta": intervention["route_delta"],
                    "greedy_changed_diagnostic": bool(intervention["greedy_changed"]),
                    "final_logits_m2_vs_m1_diagnostic": intervention[
                        "final_logits_m2_vs_m1"
                    ],
                    "m1_injected_full_forward_stable_2_of_2": bool(
                        intervention["m1_injected_full_forward_stable_2_of_2"]
                    ),
                    "m2_injected_full_forward_stable_2_of_2": bool(
                        intervention["m2_injected_full_forward_stable_2_of_2"]
                    ),
                    "m1_injected_baseline": intervention["m1_injected_baseline"],
                    "m2_injected_treatment": intervention["m2_injected_treatment"],
                }
            )
        pair_safe = pair_semantic_safe(endpoint_results)
        results.append(
            {
                "schema_version": "semanticfence-online-edge-result-v1",
                "schedule_index": int(edge["schedule_index"]),
                "edge_id": edge_id,
                "logical_split": split,
                "document_sha256": str(edge["document_sha256"]),
                "window_id": str(edge["window_id"]),
                "layer": int(edge["layer"]),
                "expert_id": int(edge["expert_id"]),
                "abi": str(edge["abi"]),
                "row_ids": list(map(str, edge["row_ids"])),
                "endpoints": endpoint_results,
                "pair_safe": pair_safe,
            }
        )
    if [row["edge_id"] for row in results] != [str(edge["edge_id"]) for edge in schedule]:
        raise GateError("semantic result/schedule identity differs")
    return results


def measure_expert_microcost(
    model: Any,
    schedule: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    if not schedule:
        raise GateError("cannot measure empty microcost schedule")
    raw = shadow._measure_microcost(model, schedule, rows_by_id)
    edges = len(schedule)
    c1_ms = float(raw["m1_median_ms"]) / (2 * edges)
    c2_ms = float(raw["m2_median_ms"]) / edges
    return {
        "scope": "same_split_candidate_edges_expert_stage_only",
        "candidate_edges": edges,
        "raw_aggregate": raw,
        "estimated_single_m1_ms": c1_ms,
        "estimated_pair_m2_ms": c2_ms,
    }


def _apply_normalization(raw: Sequence[float], normalization: Mapping[str, Any]) -> list[float]:
    mean = list(map(float, normalization["population_mean"]))
    scale = list(map(float, normalization["applied_scale_max_std_floor_1e-12"]))
    if len(raw) != len(mean) or len(raw) != len(scale):
        raise GateError("normalization width differs")
    values = [(float(value) - center) / divisor for value, center, divisor in zip(raw, mean, scale)]
    if not all(math.isfinite(value) for value in values):
        raise GateError("online normalized vector is nonfinite")
    return values


def measure_online_overhead(
    schedule: Sequence[Mapping[str, Any]],
    *,
    rows_by_id: Mapping[str, Any],
    captures_by_window: Mapping[str, Any],
    projection: Any,
    normalization: Mapping[str, Any],
    train_features: Mapping[str, Sequence[float]],
    row_cells: Mapping[str, tuple[int, int, str]],
    banks: Mapping[tuple[int, int, str], Mapping[str, Sequence[str]]],
    threshold: float | None,
    repeats: int = 5,
) -> dict[str, Any]:
    import torch

    query_ids = sorted({str(row_id) for edge in schedule for row_id in edge["row_ids"]})
    # In a real dispatch path the pre-expert hidden rows reside on CUDA.  Seed
    # one GPU-resident batch outside the timed region, then include its actual
    # D2H transfer on every measured decision.  This is deliberately
    # conservative for the CPU witness-v1 prototype.
    query_gpu = torch.stack(
        [rows_by_id[row_id].tensor.to(dtype=torch.bfloat16) for row_id in query_ids]
    ).to(device="cuda")
    torch.cuda.synchronize()
    timings: list[float] = []
    digest: str | None = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        hidden_cpu = query_gpu.to(device="cpu", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        combined = dict(train_features)
        for index, row_id in enumerate(query_ids):
            row = rows_by_id[row_id]
            capture = captures_by_window[str(row.context.window_id)]
            combined[row_id] = _apply_normalization(
                raw_online_feature(
                    row,
                    capture,
                    projection,
                    hidden_override=hidden_cpu[index],
                ),
                normalization,
            )
        scores = {
            row_id: witness_score(
                row_id,
                features=combined,
                row_cells=row_cells,
                banks=banks,
            )
            for row_id in query_ids
        }
        admitted = _threshold_admitted(scores, threshold)
        matching = rolling_greedy_matching(schedule, admitted)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        timings.append(elapsed_ms)
        observed = canonical_sha256(
            {
                "scores": [scores[row_id] for row_id in query_ids],
                "matching": matching,
            }
        )
        if digest is None:
            digest = observed
        elif observed != digest:
            raise GateError("online feature/lookup decision is not repeat-stable")
    return {
        "scope": (
            "cuda_hidden_to_cpu_transfer_plus_python_cpu_projection_normalization_"
            "cell_nn_and_greedy"
        ),
        "unique_endpoints": len(query_ids),
        "candidate_edges": len(schedule),
        "repeats": repeats,
        "elapsed_ms": timings,
        "median_total_ms": statistics.median(timings),
        "decision_digest": digest,
        "boundary": (
            "measured conservative CPU witness-v1 prototype dispatch overhead including "
            "actual D2H transfer; not optimized serving overhead"
        ),
    }


def _capture_audit_rows(captures: Sequence[Any], split_by_hash: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = pilot.capture_audit_rows(captures)
    for row in rows:
        row["logical_split"] = split_by_hash[str(row["document_sha256"])]
        row["capture_mode"] = "all_16_positions_pre_outcome_dispatch_state"
    return rows


def _artifact_hashes(output_dir: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    skipped = set(exclude or set())
    result: dict[str, str] = {}
    for path in sorted(Path(output_dir).iterdir(), key=lambda value: value.name):
        if path.is_file() and path.name not in skipped:
            result[path.name] = sha256_file(path)
    return result


def _save_torch_exclusive(path: Path, value: Any) -> None:
    import torch

    if Path(path).exists():
        raise GateError(f"refusing to overwrite tensor artifact: {path}")
    torch.save(value, path)


def _source_bindings(config_path: Path, document_input_path: Path) -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "test": EXPERIMENT_DIR / "test_run_semantic_online_observability.py",
        "config": Path(config_path).resolve(),
        "document_input": Path(document_input_path).resolve(),
        "gpu_execution": EXPERIMENT_DIR / "gpu_execution.py",
        "executor_contract": EXPERIMENT_DIR / "executor_contract.py",
        "pilot_runner": EXPERIMENT_DIR / "run_pilot_5090.py",
        "base_pilot_config": EXPERIMENT_DIR / "configs" / "pilot_5090_v1.json",
        "semantic_injection_runner": EXPERIMENT_DIR
        / "run_semantic_oracle_shadow_replay_5090.py",
        "cross_companion_runtime": EXPERIMENT_DIR
        / "run_cross_companion_metric_replay_5090.py",
        "partner_permutation_runtime": EXPERIMENT_DIR
        / "run_partner_permutation_5090.py",
        "document_preparer": EXPERIMENT_DIR / "prepare_eval_manifest.py",
        "single_contribution_runner": EXPERIMENT_DIR.parents[1]
        / "stablebatch"
        / "experiments"
        / "run_single_contribution_pilot.py",
        "observable_selector_runtime": EXPERIMENT_DIR.parents[1]
        / "stablebatch"
        / "experiments"
        / "run_observable_selector_pilot.py",
        "pilot_legacy_stack_runtime": EXPERIMENT_DIR.parents[1]
        / "spectatorroute"
        / "experiments"
        / "run_phase0a_5090.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise GateError(f"source binding files missing: {missing}")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _split_schedule(
    schedule: Sequence[Mapping[str, Any]], split: str
) -> list[dict[str, Any]]:
    result = [dict(edge) for edge in schedule if edge["logical_split"] == split]
    if not result:
        raise GateError(f"candidate schedule has no {split} edges")
    return result


def _score_split(
    schedule: Sequence[Mapping[str, Any]],
    *,
    features: Mapping[str, Sequence[float]],
    row_cells: Mapping[str, tuple[int, int, str]],
    banks: Mapping[tuple[int, int, str], Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, Any]]:
    row_ids = sorted({str(row_id) for edge in schedule for row_id in edge["row_ids"]})
    return {
        row_id: witness_score(
            row_id,
            features=features,
            row_cells=row_cells,
            banks=banks,
        )
        for row_id in row_ids
    }


def _edge_safe_map(results: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    value = {str(row["edge_id"]): bool(row["pair_safe"]) for row in results}
    if len(value) != len(results):
        raise GateError("edge result IDs repeat")
    return value


def _positive_documents(
    matching: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]
) -> int:
    document_by_edge = {
        str(edge["edge_id"]): str(edge["document_sha256"]) for edge in schedule
    }
    return len({document_by_edge[str(row["edge_id"])] for row in matching})


def _certificate_result(
    schedule: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    threshold: float | None,
    *,
    c1_ms: float,
    c2_ms: float,
    online_overhead_ms: float,
) -> dict[str, Any]:
    labels = conservative_row_labels(results)
    admitted_rows = _threshold_admitted(scores, threshold)
    result_by_edge = {str(row["edge_id"]): row for row in results}
    admissible_edges = [
        edge
        for edge in schedule
        if set(map(str, edge["row_ids"])).issubset(admitted_rows)
    ]
    unsafe_admissible = [
        str(edge["edge_id"])
        for edge in admissible_edges
        if not bool(result_by_edge[str(edge["edge_id"])]["pair_safe"])
    ]
    greedy = rolling_greedy_matching(schedule, admitted_rows)
    unsafe_greedy = [
        str(edge["edge_id"])
        for edge in greedy
        if not bool(result_by_edge[str(edge["edge_id"])]["pair_safe"])
    ]
    unsafe_endpoints = sorted(
        row_id for row_id in admitted_rows if not bool(labels[row_id])
    )
    vertices = len(labels)
    documents = {str(edge["document_sha256"]) for edge in schedule}
    positive_documents = _positive_documents(greedy, schedule)
    return {
        "schema_version": "semanticfence-online-certificate-result-v1",
        "threshold": {
            "mode": "FINITE" if threshold is not None else "ABSTAIN_ALL_POSITIVE_INFINITY",
            "value": threshold,
        },
        "total_unique_endpoints": vertices,
        "eligible_endpoints": sum(bool(value["eligible"]) for value in scores.values()),
        "admitted_endpoints": len(admitted_rows),
        "unsafe_admitted_endpoints": len(unsafe_endpoints),
        "unsafe_admitted_endpoint_ids": unsafe_endpoints,
        "admissible_candidate_edges": len(admissible_edges),
        "unsafe_admissible_candidate_edges": len(unsafe_admissible),
        "unsafe_admissible_edge_ids": unsafe_admissible,
        "greedy_executed_pairs": len(greedy),
        "unsafe_greedy_executed_pairs": len(unsafe_greedy),
        "unsafe_greedy_edge_ids": unsafe_greedy,
        "admitted_row_coverage": 2 * len(greedy) / vertices if vertices else 0.0,
        "admitted_pair_slot_coverage": (
            len(greedy) / (vertices // 2) if vertices >= 2 else 0.0
        ),
        "positive_action_documents": positive_documents,
        "document_coverage": (
            positive_documents / len(documents) if documents else 0.0
        ),
        "greedy_matching": greedy,
        "cost_projection": projected_cost(
            vertices=vertices,
            pairs=len(greedy),
            c1_ms=c1_ms,
            c2_ms=c2_ms,
            online_overhead_ms=online_overhead_ms,
        ),
        "scores": [scores[row_id] for row_id in sorted(scores)],
    }


def _oracle_result(
    schedule: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    c1_ms: float,
    c2_ms: float,
) -> dict[str, Any]:
    edge_safe = _edge_safe_map(results)
    matching = general_maximum_matching(schedule, edge_safe)
    safe_edges = int(matching["safe_edges"])
    positive_documents = _positive_documents(matching["matching"], schedule)
    total_documents = len({str(edge["document_sha256"]) for edge in schedule})
    result = {
        "schema_version": "semanticfence-online-natural-oracle-v1",
        "semantic_safe_definition": (
            "both independent endpoints have no downstream target-token ordered-top-k change "
            "under paired-M2 versus fresh-M1 injection"
        ),
        "safe_edge_density": safe_edges / len(schedule) if schedule else 0.0,
        "matching": matching,
        "positive_action_documents": positive_documents,
        "document_coverage": (
            positive_documents / total_documents if total_documents else 0.0
        ),
        "cost_projection": projected_cost(
            vertices=int(matching["unique_vertices"]),
            pairs=int(matching["matching_edges"]),
            c1_ms=c1_ms,
            c2_ms=c2_ms,
        ),
        "exact_m2_baseline_projected_saving": 0.034034,
    }
    return result


def run_gate(args: argparse.Namespace) -> int:
    import torch

    load_runtime_modules()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    input_path = Path(args.document_input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise GateError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    config = validate_config(load_json(config_path))
    config_sha = sha256_file(config_path)
    document_input = load_json(input_path)
    documents = validate_document_input(document_input, config_sha256=config_sha)
    bindings = _source_bindings(config_path, input_path)

    base_config_path = (repo_root / str(config["base_pilot_config"])).resolve()
    base_config = pilot.validate_config(pilot.load_json(base_config_path))
    model_path = pilot.resolve_model_path(base_config, args.model_path)
    model, tokenizer = pilot.load_model(base_config, model_path)
    acceptance_path = (
        repo_root
        / "docs/ideas/semanticfence/experiments/outputs/"
        "semanticfence_pilot_20260810_run03/frozen_inputs/ACCEPTANCE.json"
    ).resolve()
    # Reuse the frozen producer's validator so ACCEPTANCE_COMPLETE.json, the
    # acceptance file hash, schema/status, and its original stack-digest
    # closure are all checked before applying the narrow physical-UUID rule.
    acceptance = pilot.load_acceptance(acceptance_path)
    observed_stack = pilot.observe_stack(model)
    stack_comparison = validate_frozen_stack_except_gpu_uuid(
        observed_stack, acceptance.get("stack")
    )
    expected_uuid = str(observed_stack["gpu"]["uuid"])
    pilot.assert_clean_gpu(expected_uuid, allowed_pids={os.getpid()})

    split_by_hash = {
        str(row["text_sha256"]): str(row["logical_split"]) for row in documents
    }
    captures = gpu.capture_olmoe_split(
        model=model,
        tokenizer=tokenizer,
        documents=documents,
        split="calibration",
        token_offsets=tuple(map(int, config["capture"]["token_offsets"])),
        window_tokens=int(config["capture"]["window_tokens"]),
        add_special_tokens=bool(config["capture"]["add_special_tokens"]),
        evaluation_position=15,
    )
    if len(captures) != 24:
        raise GateError(f"expected 24 fresh capture windows, got {len(captures)}")
    _save_torch_exclusive(output_dir / "PRE_OUTCOME_CAPTURES.pt", captures)
    write_jsonl_exclusive(
        output_dir / "CAPTURE_MANIFEST.jsonl",
        _capture_audit_rows(captures, split_by_hash),
    )
    routed_rows = gpu.materialize_routed_rows(captures)
    rows_by_id = {str(row.row_id): row for row in routed_rows}
    if len(rows_by_id) != len(routed_rows):
        raise GateError("materialized row IDs repeat")
    captures_by_window = _capture_by_window(captures)

    schedule = build_candidate_schedule(
        routed_rows,
        split_by_hash,
        rolling_horizon=int(
            config["candidate_schedule"]["rolling_horizon_compatible_arrivals"]
        ),
        maximum_edges_per_document=int(
            config["candidate_schedule"]["maximum_edges_per_document"]
        ),
        sample_salt=str(config["candidate_schedule"]["sample_salt"]),
    )
    schedule_stats = validate_candidate_schedule(schedule)
    test_edges = int(schedule_stats["edges_by_split"].get("test", 0))
    if test_edges < int(config["candidate_schedule"]["minimum_test_edges"]):
        not_executable = {
            "schema_version": SCHEMA,
            "status": "NOT_EXECUTABLE_INSUFFICIENT_TEST_SUPPORT",
            "test_candidate_edges": test_edges,
            "required_test_candidate_edges": int(
                config["candidate_schedule"]["minimum_test_edges"]
            ),
            "no_semantic_outcome_generated": True,
        }
        write_json_exclusive(output_dir / "SUMMARY.json", not_executable)
        write_json_exclusive(
            output_dir / "COMPLETE.json",
            {
                "schema_version": COMPLETE_SCHEMA,
                "status": "NOT_EXECUTABLE_INSUFFICIENT_TEST_SUPPORT",
                "completion_last": True,
                "artifact_sha256": _artifact_hashes(output_dir),
            },
        )
        return 0
    write_jsonl_exclusive(output_dir / "CANDIDATE_SCHEDULE.jsonl", schedule)

    projection = make_projection(
        int(base_config["model"]["hidden_size"]),
        int(config["feature"]["projection_dimension"]),
        int(config["feature"]["projection_seed"]),
    )
    _save_torch_exclusive(output_dir / "PROJECTION.pt", projection)
    projection_manifest = {
        "schema_version": "semanticfence-online-projection-manifest-v1",
        "seed": int(config["feature"]["projection_seed"]),
        "distribution": str(config["feature"]["projection_distribution"]),
        "shape": list(map(int, projection.shape)),
        "dtype": str(projection.dtype),
        "tensor_storage_sha256": gpu.tensor_storage_sha256(projection),
        "file_sha256": sha256_file(output_dir / "PROJECTION.pt"),
        "created_before_any_m1_m2_outcome": True,
    }
    write_json_exclusive(output_dir / "PROJECTION_MANIFEST.json", projection_manifest)

    feature_records, normalized_features, normalization = build_pre_outcome_features(
        schedule, rows_by_id, captures_by_window, projection
    )
    write_jsonl_exclusive(output_dir / "PRE_OUTCOME_FEATURES.jsonl", feature_records)
    feature_schema = {
        "schema_version": "semanticfence-online-feature-schema-v1",
        "online_only": True,
        "projection_dimensions": 64,
        "projection_manifest_sha256": sha256_file(
            output_dir / "PROJECTION_MANIFEST.json"
        ),
        "feature_vector_order": [
            *[f"random_projection_{index:02d}" for index in range(64)],
            *SCALAR_FIELDS,
        ],
        "scalar_semantics": {
            "request_phase": "0.0_is_prefill",
            "abi_numeric": "1.0_is_frozen_bf16_raw_expert_abi",
            "router_top1_top2_probability_margin": "current_layer_pre_expert_top1_minus_top2",
        },
        "forbidden_feature_fields": list(config["feature"]["forbidden"]),
        "join_metadata_not_in_feature_vector": [
            "row_id",
            "document_sha256",
            "window_id",
            "row_record",
        ],
        "normalization": normalization,
        "normalization_frozen_before_any_m1_m2_outcome": True,
    }
    write_json_exclusive(output_dir / "FEATURE_SCHEMA.json", feature_schema)

    document_split = {
        "schema_version": "semanticfence-online-document-split-v1",
        "dataset": document_input["dataset"],
        "tokenizer": document_input["tokenizer"],
        "exclusions": document_input["exclusions"],
        "split_counts": document_input["split_counts"],
        "documents": [
            {key: value for key, value in row.items() if key != "text"}
            for row in documents
        ],
        "full_text_bound_by_frozen_document_input_sha256": sha256_file(input_path),
        "document_disjoint": True,
    }
    write_json_exclusive(output_dir / "DOCUMENT_SPLIT.json", document_split)
    write_json_exclusive(output_dir / "FROZEN_CONFIG.json", config)
    write_json_exclusive(output_dir / "FROZEN_DOCUMENT_INPUT.json", document_input)
    environment = {
        "schema_version": "semanticfence-online-environment-v1",
        "stack": observed_stack,
        "frozen_stack_comparison": stack_comparison,
        "frozen_acceptance_path": str(acceptance_path.relative_to(repo_root)),
        "frozen_acceptance_sha256": sha256_file(acceptance_path),
        "model_path": str(model_path),
        "model_file_sha256": dict(base_config["model"]["file_sha256"]),
    }
    write_json_exclusive(output_dir / "ENVIRONMENT.json", environment)

    pre_lock_names = {
        "PRE_OUTCOME_CAPTURES.pt",
        "CAPTURE_MANIFEST.jsonl",
        "CANDIDATE_SCHEDULE.jsonl",
        "PROJECTION.pt",
        "PROJECTION_MANIFEST.json",
        "PRE_OUTCOME_FEATURES.jsonl",
        "FEATURE_SCHEMA.json",
        "DOCUMENT_SPLIT.json",
        "FROZEN_CONFIG.json",
        "FROZEN_DOCUMENT_INPUT.json",
        "ENVIRONMENT.json",
    }
    semantic_names = {
        "TRAIN_EDGE_RESULTS.jsonl",
        "VALIDATION_EDGE_RESULTS.jsonl",
        "TEST_EDGE_RESULTS.jsonl",
        "VALIDATION_THRESHOLD.json",
        "TEST_ADMISSION_PLAN.json",
    }
    if any((output_dir / name).exists() for name in semantic_names):
        raise GateError("semantic outcome/threshold exists before pre-outcome lock")
    pre_artifacts = {
        name: sha256_file(output_dir / name) for name in sorted(pre_lock_names)
    }
    pre_lock = {
        "schema_version": LOCK_SCHEMA,
        "status": "FROZEN_BEFORE_ANY_M1_M2_SEMANTIC_OUTCOME",
        "gate": "SFV2-O1",
        "source_bindings": bindings,
        "pre_outcome_artifact_sha256": pre_artifacts,
        "document_split_sha256": pre_artifacts["DOCUMENT_SPLIT.json"],
        "candidate_schedule_sha256": pre_artifacts["CANDIDATE_SCHEDULE.jsonl"],
        "feature_schema_sha256": pre_artifacts["FEATURE_SCHEMA.json"],
        "projection_manifest_sha256": pre_artifacts["PROJECTION_MANIFEST.json"],
        "projection_tensor_storage_sha256": projection_manifest[
            "tensor_storage_sha256"
        ],
        "schedule_statistics": schedule_stats,
        "natural_schedule": {
            "arrival_order": config["candidate_schedule"]["arrival_order"],
            "stream_boundary": config["candidate_schedule"]["stream_boundary"],
            "rolling_horizon_compatible_arrivals": 8,
            "deadline": "close_after_8_compatible_arrivals_within_window",
            "maximum_edges_per_document": 32,
            "sample_salt": config["candidate_schedule"]["sample_salt"],
        },
        "semantic_surface": config["semantic_label"],
        "threshold_rule": config["certificate"]["threshold"],
        "test_outcome_count_at_lock": 0,
        "paper_result": False,
    }
    write_json_exclusive(output_dir / "PRE_OUTCOME_LOCK.json", pre_lock)

    train_schedule = _split_schedule(schedule, "train")
    validation_schedule = _split_schedule(schedule, "validation")
    test_schedule = _split_schedule(schedule, "test")
    train_results = execute_semantic_split(
        model,
        train_schedule,
        rows_by_id,
        captures_by_window,
        pre_outcome_lock=output_dir / "PRE_OUTCOME_LOCK.json",
    )
    write_jsonl_exclusive(output_dir / "TRAIN_EDGE_RESULTS.jsonl", train_results)

    row_cells = {
        str(record["row_id"]): tuple(record["cell"]) for record in feature_records
    }
    banks, bank_manifest = build_witness_bank(
        train_results, normalized_features, row_cells
    )
    bank_manifest["pre_outcome_features_sha256"] = sha256_file(
        output_dir / "PRE_OUTCOME_FEATURES.jsonl"
    )
    bank_manifest["train_edge_results_sha256"] = sha256_file(
        output_dir / "TRAIN_EDGE_RESULTS.jsonl"
    )
    write_json_exclusive(output_dir / "TRAIN_WITNESS_BANK.json", bank_manifest)

    validation_results = execute_semantic_split(
        model,
        validation_schedule,
        rows_by_id,
        captures_by_window,
        pre_outcome_lock=output_dir / "PRE_OUTCOME_LOCK.json",
    )
    write_jsonl_exclusive(
        output_dir / "VALIDATION_EDGE_RESULTS.jsonl", validation_results
    )
    validation_microcost = measure_expert_microcost(
        model, validation_schedule, rows_by_id
    )
    validation_scores = _score_split(
        validation_schedule,
        features=normalized_features,
        row_cells=row_cells,
        banks=banks,
    )
    validation_overhead = measure_online_overhead(
        validation_schedule,
        rows_by_id=rows_by_id,
        captures_by_window=captures_by_window,
        projection=projection,
        normalization=normalization,
        train_features=normalized_features,
        row_cells=row_cells,
        banks=banks,
        threshold=None,
    )
    validation_labels = conservative_row_labels(validation_results)
    frozen_threshold = select_validation_threshold(
        validation_schedule,
        validation_scores,
        validation_labels,
        c1_ms=float(validation_microcost["estimated_single_m1_ms"]),
        c2_ms=float(validation_microcost["estimated_pair_m2_ms"]),
        fixed_online_overhead_ms=float(validation_overhead["median_total_ms"]),
    )
    frozen_threshold.update(
        {
            "pre_outcome_lock_sha256": sha256_file(
                output_dir / "PRE_OUTCOME_LOCK.json"
            ),
            "train_witness_bank_sha256": sha256_file(
                output_dir / "TRAIN_WITNESS_BANK.json"
            ),
            "validation_edge_results_sha256": sha256_file(
                output_dir / "VALIDATION_EDGE_RESULTS.jsonl"
            ),
            "validation_microcost": validation_microcost,
            "validation_online_overhead": validation_overhead,
            "test_semantic_outcome_count_at_threshold_freeze": 0,
        }
    )
    write_json_exclusive(
        output_dir / "VALIDATION_THRESHOLD.json", frozen_threshold
    )

    threshold = threshold_value(frozen_threshold)
    test_scores = _score_split(
        test_schedule,
        features=normalized_features,
        row_cells=row_cells,
        banks=banks,
    )
    test_overhead = measure_online_overhead(
        test_schedule,
        rows_by_id=rows_by_id,
        captures_by_window=captures_by_window,
        projection=projection,
        normalization=normalization,
        train_features=normalized_features,
        row_cells=row_cells,
        banks=banks,
        threshold=threshold,
    )
    test_admitted = _threshold_admitted(test_scores, threshold)
    test_admission_plan = {
        "schema_version": "semanticfence-online-test-admission-plan-v1",
        "status": "FROZEN_BEFORE_FIRST_TEST_SEMANTIC_OUTCOME",
        "validation_threshold_sha256": sha256_file(
            output_dir / "VALIDATION_THRESHOLD.json"
        ),
        "pre_outcome_features_sha256": sha256_file(
            output_dir / "PRE_OUTCOME_FEATURES.jsonl"
        ),
        "train_witness_bank_sha256": sha256_file(
            output_dir / "TRAIN_WITNESS_BANK.json"
        ),
        "threshold": frozen_threshold["threshold"],
        "scores": [test_scores[row_id] for row_id in sorted(test_scores)],
        "admitted_row_ids": sorted(test_admitted),
        "candidate_admissible_edge_ids": [
            str(edge["edge_id"])
            for edge in test_schedule
            if set(map(str, edge["row_ids"])).issubset(test_admitted)
        ],
        "rolling_greedy_matching": rolling_greedy_matching(
            test_schedule, test_admitted
        ),
        "online_overhead": test_overhead,
        "test_semantic_outcome_count_at_plan_freeze": 0,
    }
    write_json_exclusive(
        output_dir / "TEST_ADMISSION_PLAN.json", test_admission_plan
    )

    test_results = execute_semantic_split(
        model,
        test_schedule,
        rows_by_id,
        captures_by_window,
        pre_outcome_lock=output_dir / "PRE_OUTCOME_LOCK.json",
        threshold_artifact=output_dir / "TEST_ADMISSION_PLAN.json",
    )
    write_jsonl_exclusive(output_dir / "TEST_EDGE_RESULTS.jsonl", test_results)
    test_microcost = measure_expert_microcost(model, test_schedule, rows_by_id)
    c1_ms = float(test_microcost["estimated_single_m1_ms"])
    c2_ms = float(test_microcost["estimated_pair_m2_ms"])
    oracle = _oracle_result(
        test_schedule, test_results, c1_ms=c1_ms, c2_ms=c2_ms
    )
    write_json_exclusive(output_dir / "ORACLE_MATCHING.json", oracle)
    certificate = _certificate_result(
        test_schedule,
        test_results,
        test_scores,
        threshold,
        c1_ms=c1_ms,
        c2_ms=c2_ms,
        online_overhead_ms=float(test_overhead["median_total_ms"]),
    )
    certificate["risk_coverage_diagnostics"] = risk_coverage_diagnostics(
        test_scores, conservative_row_labels(test_results)
    )
    certificate["greedy_stability_diagnostic"] = {
        "endpoint_observations": sum(len(row["endpoints"]) for row in test_results),
        "greedy_changed_endpoint_observations": sum(
            bool(endpoint["greedy_changed_diagnostic"])
            for row in test_results
            for endpoint in row["endpoints"]
        ),
        "diagnostic_only_not_semantic_label": True,
    }
    write_json_exclusive(output_dir / "CERTIFICATE_RESULTS.json", certificate)
    cost = {
        "schema_version": "semanticfence-online-cost-projection-v1",
        "test_expert_microcost": test_microcost,
        "natural_oracle": oracle["cost_projection"],
        "frozen_certificate": certificate["cost_projection"],
        "feature_lookup_and_greedy_overhead": test_overhead,
        "exact_baseline_projected_saving": 0.034034,
        "boundary": (
            "single-RTX5090 additive expert-stage projection plus measured Python CPU "
            "prototype certificate overhead; excludes packing queue serving EP and network"
        ),
    }
    write_json_exclusive(output_dir / "COST_PROJECTION.json", cost)
    verdict = decide_verdict(oracle, certificate)
    summary = {
        "schema_version": "semanticfence-online-observability-summary-v1",
        "gate": "SFV2-O1",
        "verdict": verdict,
        "fresh_document_count": 12,
        "document_split": SPLIT_COUNTS,
        "test_candidate_edges": len(test_schedule),
        "natural_oracle": {
            "safe_edge_density": oracle["safe_edge_density"],
            "matching_edges": oracle["matching"]["matching_edges"],
            "row_coverage": oracle["matching"]["row_coverage"],
            "projected_saving": oracle["cost_projection"]["gross_saved_fraction"],
            "positive_action_documents": oracle["positive_action_documents"],
        },
        "frozen_certificate": {
            "admitted_endpoints": certificate["admitted_endpoints"],
            "admissible_candidate_edges": certificate[
                "admissible_candidate_edges"
            ],
            "unsafe_admissible_candidate_edges": certificate[
                "unsafe_admissible_candidate_edges"
            ],
            "greedy_executed_pairs": certificate["greedy_executed_pairs"],
            "unsafe_greedy_executed_pairs": certificate[
                "unsafe_greedy_executed_pairs"
            ],
            "admitted_row_coverage": certificate["admitted_row_coverage"],
            "positive_action_documents": certificate["positive_action_documents"],
            "net_projected_saving": certificate["cost_projection"][
                "net_saved_fraction"
            ],
        },
        "paper_result": False,
        "serving_result": False,
        "multi_gpu_result": False,
    }
    write_json_exclusive(output_dir / "SUMMARY.json", summary)
    pilot.assert_clean_gpu(expected_uuid, allowed_pids={os.getpid()})
    complete = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "SUCCESS_COMPLETE",
        "verdict": verdict,
        "completion_last": True,
        "pre_outcome_lock_sha256": sha256_file(
            output_dir / "PRE_OUTCOME_LOCK.json"
        ),
        "validation_threshold_sha256": sha256_file(
            output_dir / "VALIDATION_THRESHOLD.json"
        ),
        "test_admission_plan_sha256": sha256_file(
            output_dir / "TEST_ADMISSION_PLAN.json"
        ),
        "summary_sha256": sha256_file(output_dir / "SUMMARY.json"),
        "artifact_sha256": _artifact_hashes(output_dir, exclude={"COMPLETE.json"}),
        "paper_result": False,
    }
    write_json_exclusive(output_dir / "COMPLETE.json", complete)
    print(canonical_json_bytes(summary).decode("utf-8"), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-documents", help="prepare frozen fresh-12 input without GPU outcomes"
    )
    prepare.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    prepare.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "configs" / "semantic_online_observability_v1.json",
    )
    prepare.add_argument("--cache-dir", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=prepare_documents)

    dry = subparsers.add_parser(
        "dry-run", help="validate frozen input without importing Torch or using GPU"
    )
    dry.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "configs" / "semantic_online_observability_v1.json",
    )
    dry.add_argument("--document-input", type=Path, required=True)
    dry.add_argument("--output", type=Path, required=True)
    dry.set_defaults(func=dry_run)

    run = subparsers.add_parser("run", help="execute the single formal RTX-5090 gate")
    run.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    run.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "configs" / "semantic_online_observability_v1.json",
    )
    run.add_argument("--document-input", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model-path")
    run.set_defaults(func=run_gate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"SFV2_O1_FAILED: {error}", file=sys.stderr)
        raise SystemExit(2)

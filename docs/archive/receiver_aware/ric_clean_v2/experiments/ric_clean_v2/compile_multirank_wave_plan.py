#!/usr/bin/env python3
"""Compile a hash-selected native receiver-wave plan for physical RR census.

This tool is intentionally policy- and timing-blind.  It streams a native
route JSONL, validates every complete top-k receiver wave, and retains the N
waves with the smallest frozen identity hashes.  Sender multiplicity is
reported only after selection and never participates in eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "multirank-native-wave-plan-v1"
SELECTION_SALT = "ric-clean-v2-multirank-rr-census-20260723-v1"
MODEL_SPECS = {
    "olmoe": {"hidden": 2048, "intermediate": 1024, "top_k": 8, "virtual_ep": 8},
    "llmjp": {"hidden": 512, "intermediate": 1024, "top_k": 16, "virtual_ep": 8},
}
REQUIRED_FIELDS = {
    "model_key",
    "model_revision",
    "request_id",
    "forward_id",
    "phase",
    "decode_step",
    "layer_id",
    "token_position",
    "token_id",
    "topk_slot",
    "expert_id",
    "sender_rank",
    "receiver_rank",
    "route_weight",
    "native_route_tuple_sha256",
    "valid",
}


class WavePlanError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_rows(path: Path) -> tuple[Iterator[dict[str, Any]], hashlib._Hash]:
    digest = hashlib.sha256()

    def iterator() -> Iterator[dict[str, Any]]:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                if not raw.strip():
                    raise WavePlanError(f"blank route row at line {line_number}")
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise WavePlanError(f"invalid JSON at line {line_number}") from exc
                if not isinstance(row, dict):
                    raise WavePlanError(f"route row {line_number} is not an object")
                missing = sorted(REQUIRED_FIELDS - row.keys())
                if missing:
                    raise WavePlanError(f"route row {line_number} missing {missing}")
                yield row

    return iterator(), digest


def wave_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["model_key"],
        row["request_id"],
        row["forward_id"],
        row["phase"],
        int(row["decode_step"]),
        int(row["layer_id"]),
        int(row["token_position"]),
        row["token_id"],
        int(row["receiver_rank"]),
    )


def iter_contiguous_waves(rows: Iterable[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    current_key: tuple[Any, ...] | None = None
    current: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = wave_key(row)
        if current_key is None:
            current_key = key
        if key != current_key:
            if current_key in seen:
                raise WavePlanError("receiver wave is non-contiguous or repeated")
            seen.add(current_key)
            yield current
            current = []
            current_key = key
        current.append(row)
    if current:
        if current_key in seen:
            raise WavePlanError("receiver wave is non-contiguous or repeated")
        yield current


def compile_wave(rows: Sequence[Mapping[str, Any]], expected_model: str) -> dict[str, Any]:
    if not rows:
        raise WavePlanError("empty receiver wave")
    spec = MODEL_SPECS[expected_model]
    if any(row["model_key"] != expected_model for row in rows):
        raise WavePlanError("model drift within route source")
    if any(row["valid"] is not True for row in rows):
        raise WavePlanError("invalid route contribution in native wave")
    slots = sorted(int(row["topk_slot"]) for row in rows)
    if slots != list(range(spec["top_k"])):
        raise WavePlanError(
            f"incomplete/duplicate top-k slots: expected {spec['top_k']}, observed {slots}"
        )
    first = rows[0]
    identity = {
        "model_key": expected_model,
        "request_id": first["request_id"],
        "forward_id": first["forward_id"],
        "phase": first["phase"],
        "decode_step": int(first["decode_step"]),
        "layer_id": int(first["layer_id"]),
        "token_position": int(first["token_position"]),
        "token_id": first["token_id"],
        "receiver_rank": int(first["receiver_rank"]),
    }
    if any(wave_key(row) != wave_key(first) for row in rows):
        raise WavePlanError("mixed receiver-wave identity")
    wave_id = _object_sha256({"selection_salt": SELECTION_SALT, "identity": identity})
    contributions = []
    for row in sorted(rows, key=lambda value: int(value["topk_slot"])):
        contribution_identity = {
            "wave_id": wave_id,
            "topk_slot": int(row["topk_slot"]),
            "expert_id": int(row["expert_id"]),
            "sender_rank": int(row["sender_rank"]),
            "receiver_rank": int(row["receiver_rank"]),
            "native_route_tuple_sha256": row["native_route_tuple_sha256"],
        }
        contributions.append(
            {
                **contribution_identity,
                "message_id": _object_sha256(contribution_identity),
                "payload_bytes": spec["hidden"] * 2,
                "route_weight": float(row["route_weight"]),
            }
        )
    return {
        **identity,
        "wave_id": wave_id,
        "selection_hash": wave_id,
        "model_revision": first["model_revision"],
        "hidden": spec["hidden"],
        "intermediate": spec["intermediate"],
        "top_k": spec["top_k"],
        "virtual_ep": spec["virtual_ep"],
        "contributions": contributions,
    }


def compile_plan(route_path: Path, model: str, wave_count: int) -> dict[str, Any]:
    if model not in MODEL_SPECS:
        raise WavePlanError(f"unknown model {model!r}")
    if wave_count <= 0:
        raise WavePlanError("wave_count must be positive")
    rows, source_digest = _file_rows(route_path)
    # Python's heap is a min-heap.  Negated integer hashes keep the N smallest
    # selection hashes without retaining the multi-gigabyte source in memory.
    heap: list[tuple[int, str, dict[str, Any]]] = []
    total_waves = 0
    total_rows = 0
    for group in iter_contiguous_waves(rows):
        total_waves += 1
        total_rows += len(group)
        wave = compile_wave(group, model)
        score = int(wave["selection_hash"], 16)
        item = (-score, wave["wave_id"], wave)
        if len(heap) < wave_count:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
    if len(heap) != wave_count:
        raise WavePlanError(f"requested {wave_count} waves, found {len(heap)}")
    selected = sorted((item[2] for item in heap), key=lambda wave: wave["selection_hash"])
    sender_support = [len({row["sender_rank"] for row in wave["contributions"]}) for wave in selected]
    remote_sender_support = [
        len(
            {
                row["sender_rank"]
                for row in wave["contributions"]
                if row["sender_rank"] != wave["receiver_rank"]
            }
        )
        for wave in selected
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_PHYSICAL_INPUT_NOT_EXECUTION_RESULT",
        "scientific_result": False,
        "selection_contract": {
            "salt": SELECTION_SALT,
            "method": "N_SMALLEST_SHA256_OVER_NATIVE_WAVE_IDENTITY",
            "uses_sender_multiplicity": False,
            "uses_timing_or_policy_outcome": False,
        },
        "model": model,
        "source": {
            "route_path": str(route_path),
            "route_file_sha256": source_digest.hexdigest(),
            "route_row_count": total_rows,
            "native_wave_count": total_waves,
        },
        "selected_wave_count": len(selected),
        "post_selection_diagnostics": {
            "sender_count_min": min(sender_support),
            "sender_count_max": max(sender_support),
            "remote_sender_count_min": min(remote_sender_support),
            "remote_sender_count_max": max(remote_sender_support),
            "waves_with_at_least_3_remote_senders": sum(value >= 3 for value in remote_sender_support),
        },
        "waves": selected,
    }
    payload["artifact_sha256"] = _object_sha256(payload)
    return payload


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise WavePlanError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o444)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--wave-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = compile_plan(args.route, args.model, args.wave_count)
        write_once(args.output, plan)
    except (OSError, ValueError, WavePlanError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": plan["status"],
                "model": plan["model"],
                "selected_wave_count": plan["selected_wave_count"],
                "artifact_sha256": plan["artifact_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


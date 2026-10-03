#!/usr/bin/env python3
"""Single-GPU functional smoke for the future multi-rank timed-trace schema.

The program executes the real BF16 expert matrix shapes and payload sizes from
a frozen native-wave plan, but folds every virtual rank onto one GPU and uses a
local clone instead of a transport.  It validates hooks, identities,
conservation, chronology, unpack, and join closure only.  It is deliberately
incapable of emitting a physical-incast or network result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "multirank-timed-trace-local-gpu-smoke-v1"
EVIDENCE_BOUNDARY = "ONE_GPU_ALL_RANKS_FOLDED_LOCAL_CLONE_NOT_NETWORK_NOT_INCAST"
REQUIRED_TIMESTAMPS = (
    "expert_start_ns",
    "expert_ready_ns",
    "credit_recv_ns",
    "send_start_ns",
    "send_end_ns",
    "recv_visible_ns",
    "unpack_start_ns",
    "unpack_end_ns",
    "join_close_ns",
)


class TraceSmokeError(RuntimeError):
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


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != "multirank-native-wave-plan-v1":
        raise TraceSmokeError("unexpected wave-plan schema")
    payload = dict(plan)
    expected = payload.pop("artifact_sha256", None)
    if expected != _object_sha256(payload):
        raise TraceSmokeError("wave-plan self-hash mismatch")
    if plan.get("status") != "FROZEN_PHYSICAL_INPUT_NOT_EXECUTION_RESULT":
        raise TraceSmokeError("wave plan is not frozen")
    waves = plan.get("waves")
    if not isinstance(waves, list) or not waves:
        raise TraceSmokeError("wave plan has no waves")
    for wave in waves:
        contributions = wave.get("contributions")
        if not isinstance(contributions, list) or len(contributions) != wave.get("top_k"):
            raise TraceSmokeError("wave contribution count does not equal top_k")
        if len({row["message_id"] for row in contributions}) != len(contributions):
            raise TraceSmokeError("duplicate route message identity")


def execution_message_id(
    route_message_id: str, *, credit_b: int, measured_index: int
) -> str:
    return _object_sha256(
        {
            "route_message_id": route_message_id,
            "credit_b": credit_b,
            "measured_index": measured_index,
        }
    )


def validate_records(records: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> None:
    if not records:
        raise TraceSmokeError("no measured records")
    if len({row["message_id"] for row in records}) != len(records):
        raise TraceSmokeError("execution message_id is not unique")
    route_messages = {
        row["message_id"]: row
        for wave in plan["waves"]
        for row in wave["contributions"]
    }
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in records:
        route = route_messages.get(row.get("route_message_id"))
        if route is None:
            raise TraceSmokeError("record does not bind to frozen route contribution")
        if row["payload_bytes"] != route["payload_bytes"]:
            raise TraceSmokeError("payload byte mismatch")
        values = [int(row[name]) for name in REQUIRED_TIMESTAMPS]
        if values != sorted(values):
            raise TraceSmokeError("event chronology violation")
        if row.get("transport") != "LOCAL_CLONE":
            raise TraceSmokeError("single-GPU smoke silently changed transport")
        key = (row["wave_id"], int(row["credit_b"]), int(row["measured_index"]))
        grouped.setdefault(key, []).append(row)
    wave_by_id = {wave["wave_id"]: wave for wave in plan["waves"]}
    for (wave_id, _credit_b, _measured_index), rows in grouped.items():
        wave = wave_by_id[wave_id]
        if len(rows) != wave["top_k"]:
            raise TraceSmokeError("measured join does not contain every sibling")
        close = max(int(row["unpack_end_ns"]) for row in rows)
        if any(int(row["join_close_ns"]) != close for row in rows):
            raise TraceSmokeError("join_close is not the last sibling unpack")


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise TraceSmokeError(f"refusing to overwrite {path}")
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


def run_smoke(
    plan: Mapping[str, Any], *, wave_count: int, warmups: int, measured: int
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise TraceSmokeError("torch is required") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TraceSmokeError("local smoke requires exactly one visible CUDA device")
    if wave_count <= 0 or wave_count > len(plan["waves"]):
        raise TraceSmokeError("invalid wave_count")
    if warmups < 1 or measured < 1:
        raise TraceSmokeError("warmups and measured must be positive")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = plan["model"]
    hidden = int(plan["waves"][0]["hidden"])
    intermediate = int(plan["waves"][0]["intermediate"])
    generator = torch.Generator(device=device).manual_seed(20260723)
    input_row = torch.randn((1, hidden), device=device, dtype=torch.bfloat16, generator=generator)
    weight_in = torch.randn(
        (hidden, intermediate), device=device, dtype=torch.bfloat16, generator=generator
    )
    weight_out = torch.randn(
        (intermediate, hidden), device=device, dtype=torch.bfloat16, generator=generator
    )
    torch.cuda.synchronize()
    records: list[dict[str, Any]] = []
    run_id = _object_sha256(
        {
            "plan": plan["artifact_sha256"],
            "pid": os.getpid(),
            "started_ns": time.time_ns(),
        }
    )
    stream = torch.cuda.current_stream(device)
    selected = plan["waves"][:wave_count]
    for credit_b in (1, 2, 4, 8):
        for trial in range(warmups + measured):
            wave = selected[trial % len(selected)]
            wave_records: list[dict[str, Any]] = []
            for contribution in wave["contributions"]:
                label = contribution["message_id"][:16]
                torch.cuda.nvtx.range_push(f"EXPERT:{model}:{label}")
                expert_start = time.perf_counter_ns()
                payload = torch.matmul(
                    functional.silu(torch.matmul(input_row, weight_in)), weight_out
                )
                torch.cuda.synchronize()
                expert_ready = time.perf_counter_ns()
                torch.cuda.nvtx.range_pop()
                credit_recv = expert_ready
                torch.cuda.nvtx.range_push(f"LOCAL_CLONE:{model}:{label}")
                send_start = time.perf_counter_ns()
                received = payload.clone()
                torch.cuda.synchronize()
                send_end = time.perf_counter_ns()
                recv_visible = send_end
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push(f"UNPACK:{model}:{label}")
                unpack_start = time.perf_counter_ns()
                unpacked = received.mul(float(contribution["route_weight"]))
                _ = unpacked.sum()
                torch.cuda.synchronize()
                unpack_end = time.perf_counter_ns()
                torch.cuda.nvtx.range_pop()
                if trial >= warmups:
                    measured_index = trial - warmups
                    wave_records.append(
                        {
                            "run_id": run_id,
                            "policy": "RR_CREDIT_SCHEMA_SMOKE_ONLY",
                            "wave_id": wave["wave_id"],
                            "request_id": wave["request_id"],
                            "layer_id": wave["layer_id"],
                            "token_position": wave["token_position"],
                            "topk_slot": contribution["topk_slot"],
                            "expert_id": contribution["expert_id"],
                            "sender_rank": contribution["sender_rank"],
                            "receiver_rank": contribution["receiver_rank"],
                            "physical_sender_rank": 0,
                            "physical_receiver_rank": 0,
                            "payload_bytes": contribution["payload_bytes"],
                            "expert_start_ns": expert_start,
                            "expert_ready_ns": expert_ready,
                            "credit_recv_ns": credit_recv,
                            "send_start_ns": send_start,
                            "send_end_ns": send_end,
                            "recv_visible_ns": recv_visible,
                            "unpack_start_ns": unpack_start,
                            "unpack_end_ns": unpack_end,
                            "join_close_ns": 0,
                            "stream_id": int(stream.cuda_stream),
                            "route_message_id": contribution["message_id"],
                            "message_id": execution_message_id(
                                contribution["message_id"],
                                credit_b=credit_b,
                                measured_index=measured_index,
                            ),
                            "credit_b": credit_b,
                            "measured_index": measured_index,
                            "transport": "LOCAL_CLONE",
                        }
                    )
            if wave_records:
                close = max(row["unpack_end_ns"] for row in wave_records)
                for row in wave_records:
                    row["join_close_ns"] = close
                records.extend(wave_records)
    validate_records(records, plan)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_FUNCTIONAL_SMOKE_NOT_PHYSICAL_EVIDENCE",
        "scientific_result": False,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "plan_artifact_sha256": plan["artifact_sha256"],
        "model": model,
        "configuration": {
            "wave_count": wave_count,
            "warmups_per_credit_b": warmups,
            "measured_per_credit_b": measured,
            "credit_b_values": [1, 2, 4, 8],
            "all_virtual_ranks_folded_to_cuda0": True,
        },
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "python_executable": os.path.realpath(os.sys.executable),
            "platform": platform.platform(),
        },
        "record_count": len(records),
        "records": records,
    }
    report["artifact_sha256"] = _object_sha256(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave-count", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measured", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        report = run_smoke(
            plan,
            wave_count=args.wave_count,
            warmups=args.warmups,
            measured=args.measured,
        )
        write_once(args.output, report)
    except (OSError, ValueError, TraceSmokeError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "model": report["model"],
                "record_count": report["record_count"],
                "artifact_sha256": report["artifact_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


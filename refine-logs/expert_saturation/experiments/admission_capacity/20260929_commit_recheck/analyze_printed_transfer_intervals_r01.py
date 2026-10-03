#!/usr/bin/env python3
"""Sum only complete printed KV-transfer intervals inside eight measurement phases."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


SESSIONS = (
    ("moe-a-capacity-victim-session-r01-20261001",
     ("eager_performance_off", "eager_performance_on")),
    ("moe-a-capacity-victim-session-b2-r01-20261001",
     ("eager_performance_on", "eager_performance_off")),
    ("moe-a-capacity-identical-session-r01-20261001",
     ("capacity_reference_first_on", "capacity_reference_second_on")),
    ("moe-a-capacity-warm-identical-session-r02-20261001",
     ("capacity_reference_first_on", "capacity_reference_second_on")),
)
PROTECTION_SESSION = "moe-a-capacity-protection-session-r03-20261001"
PROTECTION_ARMS = ("capacity_protection_q1", "capacity_protection_q10")
FIELDS = ("store_bytes", "store_time", "store_size_count",
          "load_bytes", "load_time", "load_size_count")
TIMES = {"store_time", "load_time"}
STAMP = re.compile(r"^INFO\s+(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def values(line: str) -> dict:
    result = {}
    for field in FIELDS:
        match = re.search(r"\bvllm:kv_offload_" + field + r"=([^,\s]+)", line)
        require(match is not None, f"printed transfer field missing: {field}")
        value = Decimal(match.group(1)) if field in TIMES else int(match.group(1))
        require(value >= 0, f"negative printed transfer field: {field}")
        result[field] = value
    return result


def descriptor(number: int, line: str) -> dict:
    stamp = STAMP.search(line)
    require(stamp is not None, f"KV metrics line {number} lacks timestamp")
    return {"line": number, "timestamp": stamp.group(1)}


def analyze_log(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"missing launch.log: {path}")
    lines = path.read_text(errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines, 1) if line == "PHASE MEASUREMENT_BEGIN"]
    ends = [i for i, line in enumerate(lines, 1) if line == "PHASE MEASUREMENT_END"]
    require(len(starts) == len(ends) == 1 and starts[0] < ends[0],
            f"measurement markers differ: {path}")
    printed = [(i, line) for i, line in enumerate(lines, 1)
               if starts[0] < i < ends[0] and "KV Transfer metrics:" in line]
    require(len(printed) >= 2, f"fewer than two measurement transfer prints: {path}")
    excluded = printed[0]
    included = printed[1:]
    totals = {field: Decimal(0) if field in TIMES else 0 for field in FIELDS}
    for _, line in included:
        parsed = values(line)
        for field in FIELDS:
            totals[field] += parsed[field]
    clean = lambda record: {field: float(value) if field in TIMES else value
                            for field, value in record.items()}
    return {
        "launch_log": str(path), "launch_log_sha256": sha256(path),
        "phase_begin_line": starts[0], "phase_end_line": ends[0],
        "measurement_printed_lines": len(printed),
        "included_complete_intervals": len(included),
        "measurement_first_print": descriptor(*printed[0]),
        "measurement_last_print": descriptor(*printed[-1]),
        "excluded_first_possible_warmup_mixture": {
            **descriptor(*excluded), "printed_values": clean(values(excluded[1]))},
        "included_first_print": descriptor(*included[0]),
        "included_last_print": descriptor(*included[-1]),
        "included_interval_sum": clean(totals),
    }


def scope_notes() -> list[str]:
    return [
            "Logger prints and resets about every 10 seconds. Included values are sums of printed complete intervals, not process-cumulative totals.",
            "The first print after MEASUREMENT_BEGIN is excluded because its interval may contain warmup activity; its values are recorded separately.",
            "No MEASUREMENT_END flush/reset was recorded. The unprinted tail after the last included line is unknown, so sums are lower coverage of completed transfers rather than episode totals.",
            "Worker store/load bytes measure completed copy volume. Printed times use CUDA events with prior stream waits before timing start; they are not request-visible latency.",
            "Only store/load bytes, time and size_count are summed. size_sum and cache-usage gauges are not accumulated.",
        ]


def analyze(root: Path) -> dict:
    cells = []
    for session, arms in SESSIONS:
        for index, arm in enumerate(arms):
            path = root / session / f"cell-{index:02d}-{arm}" / "launch.log"
            cells.append({"session": session, "arm": arm, **analyze_log(path)})
    return {
        "schema_version": 1,
        "status": "PRINTED_TRANSFER_INTERVALS_DESCRIBED",
        "cells": cells,
        "scope": scope_notes(),
    }


def analyze_session(session: Path) -> dict:
    require(session.is_dir(), "session missing")
    receipt = json.loads((session / "receipt.json").read_text())
    require(receipt.get("status") == "CELLS_COMPLETE", "session incomplete")
    cells = []
    for index, cell in enumerate(receipt["cells"]):
        arm = cell["arm"]
        path = session / f"cell-{index:02d}-{arm}" / "launch.log"
        cells.append({"session": session.name, "arm": arm, **analyze_log(path)})
    return {
        "schema_version": 1,
        "status": "PRINTED_TRANSFER_INTERVALS_DESCRIBED",
        "cells": cells,
        "scope": scope_notes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, help="A-only experiment directory")
    parser.add_argument("--session", type=Path, help="completed session with receipt.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    require(args.session is None or args.root is None,
            "use either root or --session")
    result = analyze_session(args.session) if args.session else analyze(args.root or Path("."))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "cells": len(result["cells"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

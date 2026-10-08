#!/usr/bin/env python3
"""Known-cap 1024-token chunk kinematics on recorded PF batch-4096 states.

This is a deterministic diagnostic, not a scheduler, native replay, or
performance counterfactual. No future arrivals or early EOS are assumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

BLOCK = 16
BLOCK_CAPACITY = 4096
BATCH = 1024
ARMS = ("past_future_1", "past_future_2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ceil_blocks(tokens: int) -> int:
    return (tokens + BLOCK - 1) // BLOCK


def state(row: dict) -> tuple[int, int, int]:
    prompt, output, cap = (row[key] for key in
                           ("prompt_tokens", "output_tokens", "max_output_tokens"))
    if not (type(prompt) is type(output) is type(cap) is int
            and prompt > 0 and 0 <= output < cap and cap > 0):
        raise ValueError("invalid recorded P/O/M state")
    return prompt, output, cap


def peaks(decision: dict) -> tuple[int, int, int]:
    head = decision["head_input"]
    hp, ho, hm = state(head)
    if head["state"] != "WAIT_IN_QUEUE" or ho != 0:
        raise ValueError("probe requires a fresh one-head waiting request")
    old = [state(row) for row in decision["running_inputs"]]
    if len(old) >= BATCH:
        raise ValueError("old decode count exhausts the hypothetical batch")
    retire = [cap - output for _, output, cap in old]

    # Each old request receives one output token on step t <= its cap remainder.
    # It is freed at the end of that step. The head gets the remaining tokens.
    def prefetched(t: int) -> int:
        return min(hp, BATCH * t - sum(min(t, rem) for rem in retire))

    k = 1
    while prefetched(k) < hp:
        k += 1
    if k > (hp + BATCH - 1) // (BATCH - len(old)):
        raise AssertionError("prefill step bound failed")

    def old_blocks(t: int) -> int:
        return sum(ceil_blocks(p + o + t) for (p, o, _), rem
                   in zip(old, retire) if t <= rem)

    def full_head(t: int) -> int:
        return ceil_blocks(hp + t) if t <= hm else 0

    def chunk_head(t: int) -> int:
        if t > k + hm - 1:
            return 0
        tokens = prefetched(t) + 1 if t <= k else hp + 1 + t - k
        return ceil_blocks(tokens)

    # Within retirement events each active occupancy is nondecreasing, so
    # maxima occur on a cap-reaching step just before its end-step release.
    full = max(old_blocks(t) + full_head(t) for t in set(retire) | {hm})
    chunk = max(old_blocks(t) + chunk_head(t)
                for t in set(retire) | {k + hm - 1})
    if chunk > full:
        raise AssertionError("delayed head exceeded immediate full-head peak")
    return full, chunk, k


def analyze(root: Path) -> dict:
    arms = {}
    for arm in ARMS:
        path = root / arm / "native_past_future_ae" / "past-future-ae.json"
        raw = json.loads(path.read_text())
        if raw["status"] != "DRAINED" or raw["preemptions"] != 0 or raw["violations"]:
            raise ValueError(f"{path}: not a zero-preemption drained PF trace")
        if raw["batch_tokens"] != 4096 or raw["usable_blocks"] != BLOCK_CAPACITY:
            raise ValueError(f"{path}: incompatible recorded resource regime")
        counts, reasons, opportunity_heads = Counter(), Counter(), set()
        reductions, prefill_steps = [], []
        max_full = max_chunk = 0
        for row in raw["decisions"]:
            full, chunk, k = peaks(row)
            reason = row["reason"]
            reasons[reason] += 1
            counts["decisions"] += 1
            counts["full_peak_over_capacity"] += full > BLOCK_CAPACITY
            counts["chunk_peak_over_capacity"] += chunk > BLOCK_CAPACITY
            counts["both_fit"] += full <= BLOCK_CAPACITY and chunk <= BLOCK_CAPACITY
            counts["both_fail"] += full > BLOCK_CAPACITY and chunk > BLOCK_CAPACITY
            if chunk <= BLOCK_CAPACITY < full:
                counts["chunk_only_fits_calls"] += 1
                reasons[f"chunk_only_fits_{reason}"] += 1
                opportunity_heads.add(row["head"])
            reductions.append(full - chunk)
            prefill_steps.append(k)
            max_full, max_chunk = max(max_full, full), max(max_chunk, chunk)
        arms[arm] = dict(input_file=str(path.relative_to(root)),
            input_sha256=sha256(path), decisions=counts["decisions"],
            unique_heads=len({row["head"] for row in raw["decisions"]}),
            reason_counts=dict(sorted((key, val) for key, val in reasons.items()
                                      if not key.startswith("chunk_only_fits_"))),
            chunk_only_fits_by_reason=dict(sorted((key.removeprefix("chunk_only_fits_"), val)
                                                 for key, val in reasons.items()
                                                 if key.startswith("chunk_only_fits_"))),
            chunk_only_fits_unique_heads=len(opportunity_heads),
            max_peak_reduction_blocks=max(reductions, default=0),
            max_prefill_steps=max(prefill_steps, default=0),
            max_full_peak_blocks=max_full, max_chunk_peak_blocks=max_chunk,
            counts=dict(sorted(counts.items())))
    return dict(schema="c-pf-chunk-kinematic-probe-v1", source_root=str(root),
                hypothetical_batch_tokens=BATCH, usable_blocks=BLOCK_CAPACITY,
                block_tokens=BLOCK, arms=arms,
                scope="Recorded batch-4096 PF states evaluated under a hypothetical "
                      "1024-token kinematic schedule. Deterministic cap bounds only; "
                      "not a native execution, policy, or benefit estimate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

#!/usr/bin/env python3
"""Account full-150 native scheduled tokens and ideal prompt-prefix work; CPU only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROWS = 150
BLOCK = 16
WORKLOAD_SHA256 = "3e34bc8a46abc1329744e22b6c97b18582a540f0abd98de55f012f560aac92fc"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ideal_prefix_work(rows: list[dict]) -> tuple[int, int, int]:
    """Unique reusable prefix nodes plus tails, subject to this cohort's audit."""
    nodes: dict[tuple[int, tuple[int, ...]], int] = {}
    tail_tokens = 0
    for row in rows:
        ids = row["prompt_token_ids"]
        # vLLM APC cannot make the whole prompt a hit: leave at least one token.
        reusable_blocks = (len(ids) - 1) // BLOCK
        tail_tokens += len(ids) - BLOCK * reusable_blocks
        parent = 0
        for block_index in range(reusable_blocks):
            start = BLOCK * block_index
            key = (parent, tuple(ids[start:start + BLOCK]))
            if key not in nodes:
                nodes[key] = len(nodes) + 1
            parent = nodes[key]
    unique_blocks = len(nodes)
    return unique_blocks, tail_tokens, BLOCK * unique_blocks + tail_tokens


def prefix_relation_audit(rows: list[dict]) -> dict:
    """Reject cohorts where a computed terminal block can populate the trie."""
    prompts = [tuple(row["prompt_token_ids"]) for row in rows]
    same: dict[tuple[int, ...], list[int]] = {}
    for index, prompt in enumerate(prompts):
        same.setdefault(prompt, []).append(index)
    duplicate_groups = [indices for indices in same.values() if len(indices) > 1]
    strict_prefix_pairs = []
    terminal_block_reuse_pairs = []
    for short_index, short in enumerate(prompts):
        for long_index, long in enumerate(prompts):
            if len(short) < len(long) and long[:len(short)] == short:
                strict_prefix_pairs.append([short_index, long_index])
                if len(short) % BLOCK == 0:
                    terminal_block_reuse_pairs.append([short_index, long_index])
    if terminal_block_reuse_pairs:
        raise ValueError("terminal prompt block also populates another reusable prefix; simple trie sum would double count")
    return dict(exact_duplicate_prompt_groups=duplicate_groups,
                strict_full_prompt_prefix_pairs=strict_prefix_pairs,
                terminal_block_reuse_pairs=terminal_block_reuse_pairs,
                full_block_terminal_rows=[i for i, p in enumerate(prompts) if len(p) % BLOCK == 0])


def account(input_dir: Path, run_dir: Path) -> dict:
    workload_path = input_dir / "workload.json"
    outputs_path = run_dir / "measured-outputs.json"
    steps_path = run_dir / "measured-steps.json"
    eos_path = run_dir / "resolved-eos.json"
    if sha(workload_path) != WORKLOAD_SHA256:
        raise ValueError("not the frozen full-150 workload")
    workload, outputs, steps, eos = map(read,
        (workload_path, outputs_path, steps_path, eos_path))
    rows = workload["requests"]
    if (len(rows) != ROWS or len(outputs) != ROWS
            or [row["source_index"] for row in rows] != list(range(ROWS))
            or [row["request_id"] for row in rows]
               != [row["request_id"] for row in outputs]
            or not all(row["prompt_token_ids"] == out["prompt_token_ids"]
                       for row, out in zip(rows, outputs))):
        raise ValueError("input/output request identity or prompt IDs differ")
    for row in rows:
        ids = row["prompt_token_ids"]
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("invalid prompt IDs")
    if (eos.get("qualification_status") != "QUALIFIED"
            or eos.get("qualified_eos_token_ids") != [50279]):
        raise ValueError("EOS receipt differs")
    reasons = Counter()
    for out in outputs:
        ids, reason = out.get("output_token_ids"), out.get("finish_reason")
        if (out.get("finished") is not True or not isinstance(ids, list) or not ids
                or any(type(token) is not int or token < 0 for token in ids)
                or reason not in ("stop", "length")
                or out.get("stop_reason") is not None
                or (reason == "stop" and ids[-1] != 50279)
                or (reason == "length" and (len(ids) != 64 or 50279 in ids))):
            raise ValueError("output length/EOS accounting differs")
        reasons[reason] += 1
    if (steps.get("preempted_request_ids") != []
            or steps.get("allocation_failures") != []):
        raise ValueError("recompute-free accounting requires zero preemption/failure")
    calls = steps.get("scheduler_calls")
    if (not isinstance(calls, list) or not calls
            or any(type(call.get("scheduled_tokens_total")) is not int
                   or call["scheduled_tokens_total"] < 0
                   or call.get("preempted_request_ids") != [] for call in calls)):
        raise ValueError("invalid native schedule totals")

    prompt_tokens = sum(len(row["prompt_token_ids"]) for row in rows)
    output_tokens = sum(len(out["output_token_ids"]) for out in outputs)
    # Prefill of P prompt tokens yields the first output token. Each subsequent
    # output token, including a final EOS when present, needs one decode step.
    decode_scheduled = output_tokens - ROWS
    no_apc_scheduled = prompt_tokens + decode_scheduled
    actual_scheduled = sum(call["scheduled_tokens_total"] for call in calls)
    avoided_scheduled = no_apc_scheduled - actual_scheduled
    if avoided_scheduled < 0:
        raise ValueError("actual scheduling exceeds the no-APC one-pass account")
    relations = prefix_relation_audit(rows)
    unique_blocks, tail_tokens, ideal_prompt = ideal_prefix_work(rows)
    ideal_total = ideal_prompt + decode_scheduled
    return {
        "schema": "c-longbench-multifieldqa-en-cache-headroom-v1",
        "scope": "CPU accounting of frozen full-150 native trace; prompt-prefix-only and fixed observed output work; no online counterfactual or capacity claim",
        "input_sha256": {"workload.json": sha(workload_path)},
        "run_sha256": {name: sha(run_dir / name) for name in
                       ("measured-outputs.json", "measured-steps.json", "resolved-eos.json")},
        "requests": ROWS, "block_tokens": BLOCK,
        "finish_reason_counts": dict(reasons), "eos_token_id": 50279,
        "eos_accounting": "EOS is last output_token_id for every stop; length outputs have 64 IDs and no EOS; all output IDs include final generated token",
        "prompt_tokens_no_apc_one_pass": prompt_tokens,
        "output_token_ids_including_eos": output_tokens,
        "decode_scheduled_tokens_one_pass": decode_scheduled,
        "no_apc_one_pass_scheduled_tokens": no_apc_scheduled,
        "native_schedule_calls": len(calls),
        "native_scheduled_tokens": actual_scheduled,
        "avoided_scheduled_tokens_vs_no_apc": avoided_scheduled,
        "avoided_full_block_equivalents": avoided_scheduled / BLOCK,
        "ideal_prefix_unique_reusable_blocks": unique_blocks,
        "prompt_relation_audit": relations,
        "ideal_prefix_unique_block_tokens": BLOCK * unique_blocks,
        "ideal_prefix_mandatory_per_request_tail_tokens": tail_tokens,
        "ideal_prompt_work_lower_bound_tokens": ideal_prompt,
        "ideal_total_work_lower_bound_tokens": ideal_total,
        "ideal_max_prompt_savings_vs_no_apc_tokens": prompt_tokens - ideal_prompt,
        "native_minus_ideal_total_tokens": actual_scheduled - ideal_total,
        "interpretation": [
            "The no-APC number is the P plus O-minus-one one-pass schedule account, not a measured no-APC run.",
            "The aggregate difference is avoided scheduled token work consistent with 16-token APC reuse; aggregate logs do not identify individual cache hits, completed-prefix eviction, or causes of the remaining gap.",
            "For this audited cohort only, the trie lower bound assumes infinite cache and perfect request order, shares full 16-token prompt prefixes, leaves at least one prompt token uncached per request, and has no terminal-block double count.",
            "It ignores generated-prefix sharing and holds observed output work fixed; it is not a general formula or a global bound over different generation trajectories.",
            "The cohort-conditional lower bound is not achievable capacity evidence for 4096 blocks or an online latency/goodput counterfactual.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = account(args.input_dir, args.run_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in
                      ("no_apc_one_pass_scheduled_tokens", "native_scheduled_tokens",
                       "avoided_scheduled_tokens_vs_no_apc", "ideal_total_work_lower_bound_tokens",
                       "native_minus_ideal_total_tokens")}))


if __name__ == "__main__":
    main()

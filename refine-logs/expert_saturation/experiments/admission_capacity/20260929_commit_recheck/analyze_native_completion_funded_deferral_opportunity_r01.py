#!/usr/bin/env python3
"""Observed numeric no-new-page completion witnesses at full-running pressure steps."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


DEFAULT_SESSION = Path("moe-a-native-bidkv-full-running-session-r01-20261002")
DEFAULT_OUTPUT = Path("A_NATIVE_COMPLETION_FUNDED_DEFERRAL_OPPORTUNITY_R01_20261002.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(session):
    cells = list(session.glob("cell-*-bidkv_full_break/archive"))
    require(len(cells) == 1, "Expected exactly one full-running arm archive")
    archive = cells[0]
    paths = {name: archive / f"{name}.json" for name in
             ("raw", "config", "safe-cap-qualification", "selective-store")}
    raw, config, qualification, store = (json.loads(paths[name].read_text())
                                         for name in paths)
    block_size = qualification.get("block_size")
    require(block_size == qualification.get("scheduler_block_size")
            == qualification.get("single_type_block_size") == 16,
            "Runtime KV block size differs from 16 tokens")
    require(qualification.get("num_lookahead_tokens") == 0
            and qualification.get("num_spec_tokens") == 0
            and qualification.get("prefix_caching") is False,
            "Lookahead, speculative tokens, or prefix caching change page-capacity reading")
    require(config.get("output_tokens") == config.get("max_output_tokens") == 1024,
            "Frozen per-request hard output cap differs")
    require(config.get("native_victim_full_running") is True
            and store.get("native_victim_full_running_enabled") is True
            and store.get("native_victim_rule") == "bidkv_score",
            "Archive is not the full-running BidKV arm")
    require(store.get("current_victim_guard_enabled") is False
            and store.get("self_preempt_continue_enabled") is False,
            "Guard or continuation changed this observed candidate set")
    request_rows = raw.get("requests")
    require(isinstance(request_rows, list) and len(request_rows) == 128,
            "Raw complete request cohort missing")
    requests = {row["internal_request_id"]: row for row in request_rows}
    require(len(requests) == 128, "Duplicate internal request IDs")
    require(all(row.get("max_output_tokens") == config["max_output_tokens"]
                for row in request_rows),
            "Raw static per-request maximum differs from frozen config")
    decisions = store.get("victim_decisions")
    require(isinstance(decisions, list), "Victim decision log missing")

    records = []
    for ordinal, decision in enumerate(decisions):
        require(decision.get("rule") == "bidkv_score"
                and decision.get("full_running_candidate_set") is True,
                "Decision lacks the recorded full-running candidate set")
        witnesses = []
        unknown_candidates = []
        qualified_noncurrent = 0
        for row in decision["candidates"]:
            request_id = row["request"]
            if request_id == decision["failed_request"] or row.get("qualified") is not True:
                continue
            qualified_noncurrent += 1
            request = requests.get(request_id)
            score = row.get("bidkv") or {}
            cap = request.get("max_output_tokens") if request else None
            fraction = score.get("completion")
            computed = score.get("computed_tokens")
            held = row.get("held_blocks")
            if (type(cap) is not int or cap <= 0
                    or not isinstance(fraction, (float, int))
                    or not 0 <= fraction <= 1
                    or type(computed) is not int or computed < 0
                    or type(held) is not int or held <= 0):
                unknown_candidates.append(request_id)
                continue
            output_count_float = fraction * cap
            output_count = round(output_count_float)
            # The frozen cap is 1024, so the logged ratio should reconstruct
            # its exact integer numerator without any future output tokens.
            if abs(output_count_float - output_count) > 1e-9:
                unknown_candidates.append(request_id)
                continue
            remaining = cap - output_count
            capacity = held * block_size
            if remaining <= 0 or computed + remaining > capacity:
                continue
            witnesses.append(dict(
                internal_request_id=request_id,
                source_request_id=request["request_id"],
                running_index=row["index"],
                scheduled_prefix=row.get("scheduled_prefix"),
                output_tokens_at_decision=output_count,
                hard_output_cap=cap,
                remaining_output_cap=remaining,
                computed_tokens_at_decision=computed,
                held_blocks_at_decision=held,
                existing_page_token_capacity=capacity,
                capacity_slack_tokens=capacity - computed - remaining,
            ))
        witnesses.sort(key=lambda row: (row["remaining_output_cap"], row["internal_request_id"]))
        records.append(dict(
            ordinal=ordinal, step=decision["step"],
            failed_current=decision["failed_request"],
            selected_native_victim=decision["selected"],
            native_tail=decision["native_tail"],
            raw_free_blocks_at_decision=decision.get("free_blocks"),
            candidate_rows=len(decision["candidates"]),
            qualified_noncurrent_candidates=qualified_noncurrent,
            unknown_candidate_ids=unknown_candidates,
            witness_count=len(witnesses),
            minimum_remaining_output_steps=(witnesses[0]["remaining_output_cap"]
                                            if witnesses else None),
            witness_ids=[row["internal_request_id"] for row in witnesses],
            witnesses=witnesses,
            unknown_gates={
                "post_deferral_native_admission_and_progress": "NOT_EXECUTED",
                "native_transfer_and_inflight_reservations": "NOT_RECORDED_PER_CANDIDATE_HERE",
                "actual_EOS_before_hard_cap": "UNKNOWN_AND_NOT_USED",
                "net_free_capacity_after_a_witness_finishes": "NOT_OBSERVED_UNDER_DEFERRAL",
            },
        ))
    all_witnesses = [w for record in records for w in record["witnesses"]]
    return dict(
        status="OBSERVED_NUMERIC_OPPORTUNITY_ONLY",
        session=str(session), full_arm_archive=str(archive),
        source_sha256={name: sha256(path) for name, path in paths.items()},
        static_identity=dict(block_size_tokens=block_size,
                             config_hard_output_cap=config["max_output_tokens"],
                             raw_request_caps_identical=True,
                             request_count=len(requests),
                             source_workload_sha256=config["workload_sha256"]),
        definition=("At the recorded decision only: non-failed-current qualified pure-decode "
                    "candidate, positive remaining_output_cap=max_tokens-num_output_tokens, "
                    "and computed_tokens+remaining_output_cap <= held_blocks*block_size. "
                    "The candidate's logged BidKV completion fraction reconstructs its "
                    "integer output count against the static 1024-token cap; no later outputs are used."),
        summary=dict(
            pressure_decisions=len(records),
            decisions_with_witness=sum(row["witness_count"] > 0 for row in records),
            total_witness_occurrences=len(all_witnesses),
            distinct_witness_requests=len({w["internal_request_id"] for w in all_witnesses}),
            witness_count_distribution=dict(sorted(Counter(
                row["witness_count"] for row in records).items())),
            remaining_output_steps_min=min((w["remaining_output_cap"]
                                            for w in all_witnesses), default=None),
            remaining_output_steps_max=max((w["remaining_output_cap"]
                                            for w in all_witnesses), default=None),
            scheduled_prefix_witnesses=sum(w["scheduled_prefix"] is True
                                           for w in all_witnesses),
            unknown_candidate_occurrences=sum(len(row["unknown_candidate_ids"])
                                              for row in records),
        ),
        decisions=records,
        interpretation=("A possible ordinary deferral would hold the page-needing failed decode "
                        "and continue a separately qualified completion-funded request, avoiding "
                        "an immediate victim preemption if native scheduling actually permits it. "
                        "This differs from the current guard, which changes the victim after "
                        "allocation failure. No such deferral was run or timed here."),
        limitations=[
            "The inequality is a conservative sufficient page-capacity condition under the fixed hard cap and zero lookahead; EOS may stop earlier. It is not an observed alternative schedule.",
            "Existing held-block capacity is physical ownership recorded at the decision; it does not measure future EOS, transfers, or net released blocks.",
            "Already-scheduled prefix token work is not added again to computed_tokens+remaining_output_cap; this is the specified conservative bound.",
            "All per-request final outputs in raw are deliberately ignored except static max_output_tokens and identity.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), "Output file already exists")
    result = analyze(args.session)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

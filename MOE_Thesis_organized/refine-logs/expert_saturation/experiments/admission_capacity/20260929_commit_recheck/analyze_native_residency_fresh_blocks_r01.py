#!/usr/bin/env python3
"""Summarize two fixed-input native victim triplets without pooling denominators."""

import argparse
import json
from pathlib import Path

from analyze_native_residency_victim_triplet_r01 import analyze, require


PAIRS = {"tail": "density_vs_tail", "arrival": "density_vs_arrival"}


def direction_counts(block, pair_name):
    frontier = block["pairwise_comparisons"][pair_name]["full_cohort_pair"]["frontier"]
    require(len(frontier) == 20, f"Expected 20 fixed goodput points: {pair_name}")
    values = [point["goodput_difference_requests_s"] for point in frontier]
    require(all(isinstance(value, (int, float)) for value in values),
            f"Missing goodput difference: {pair_name}")
    return dict(higher=sum(value > 0 for value in values),
                equal=sum(value == 0 for value in values),
                lower=sum(value < 0 for value in values), total=len(values),
                direction="service_density minus control, in requests/s")


def block_comparison(block, pair_name):
    score = block["pairwise_criterion_values"][pair_name]
    reference_gap = score["max_gap_reference_s"]
    candidate_gap = score["max_gap_candidate_s"]
    require(reference_gap is not None and reference_gap > 0 and candidate_gap is not None,
            f"Max-gap ratio unavailable: {pair_name}")
    return dict(rate_ratio=score["rate_ratio"],
                mean_flow_ratio=score["mean_flow_ratio"],
                max_gap_ratio=candidate_gap / reference_gap,
                max_gap_control_s=reference_gap,
                max_gap_density_s=candidate_gap,
                goodput_directions=direction_counts(block, pair_name))


def analyze_blocks(first_session, second_session):
    require(first_session.resolve() != second_session.resolve(),
            "First and second blocks must be distinct sessions")
    blocks = {"first": analyze(first_session), "second": analyze(second_session)}
    criteria = {name: block["predeclared_criteria"] for name, block in blocks.items()}
    require(all(len(values) == 12 for values in criteria.values()),
            "Each block must retain the original 12 predeclared criteria")
    require(set(criteria["first"]) == set(criteria["second"]),
            "The blocks have different predeclared criteria")
    comparisons = {
        control: {name: block_comparison(block, pair_name)
                  for name, block in blocks.items()}
        for control, pair_name in PAIRS.items()
    }
    extrema = {
        control: {
            metric: dict(min=min(comparisons[control][name][metric] for name in blocks),
                         max=max(comparisons[control][name][metric] for name in blocks))
            for metric in ("rate_ratio", "mean_flow_ratio", "max_gap_ratio")
        }
        for control in PAIRS
    }
    passed = {name: (block["status"] == "COMPLETE_TRIPLET"
                     and block["all_criteria_met"] is True
                     and all(criteria[name].values()))
              for name, block in blocks.items()}
    return dict(
        status="BOTH_BLOCKS_PASS" if all(passed.values()) else "BLOCK_CRITERION_FAILED",
        both_blocks_original_12_criteria_pass=all(passed.values()),
        block_pass=passed, blocks=blocks, density_vs_control=comparisons,
        within_block_ratio_min_max=extrema,
        interpretation="Each block keeps its complete original analysis; ratios use only that block's control denominator. Min/max summarize two ratios without pooling request counts, token rates, or elapsed time.",
        limitations=["One serial triplet per block does not establish statistical stability or same-state causality.",
                     "No best block or best control is selected."],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be a new file")
    result = analyze_blocks(args.first, args.second)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print(json.dumps(dict(status=result["status"],
                          block_pass=result["block_pass"])))


if __name__ == "__main__":
    main()

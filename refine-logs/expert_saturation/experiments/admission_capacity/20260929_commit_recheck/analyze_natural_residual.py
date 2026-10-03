"""Read-only descriptive G/H recovery analysis; no policy counterfactuals."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ("native_full_native", "current", "eager")
DOCUMENT = "memory-train-article-0018772"


def load(path):
    data = path.read_bytes()
    result = json.loads(data)
    expected = {f"block{block}-{arm}" for block in (0, 1) for arm in ARMS}
    if set(result["cells"]) != expected:
        raise ValueError(f"unexpected cells in {path}")
    return result, hashlib.sha256(data).hexdigest()


def gaps(cell):
    rows = cell["requests"]["requests"]
    result = {row["request_id"]: row["max_engine_return_gap_s"] for row in rows}
    if len(result) != len(rows) or any(value is None for value in result.values()):
        raise ValueError("request identities or generation gaps are incomplete")
    return result


def pearson(left, right):
    if set(left) != set(right):
        raise ValueError("request cohorts differ")
    ids = sorted(left)
    x = [left[rid] for rid in ids]
    y = [right[rid] for rid in ids]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def analyze(g, h):
    short_service = {}
    for name, data in (("G", g), ("H", h)):
        short_service[name] = {
            cell: {
                key: data["cells"][cell]["sparse_recovery"]["counts"][key]
                for key in ("zero_new_outputs_then_repreempted",
                            "one_two_new_outputs_then_repreempted")
            }
            for cell in sorted(data["cells"])
        }

    repeats = {}
    for arm in ARMS:
        left = gaps(h["cells"][f"block0-{arm}"])
        right = gaps(h["cells"][f"block1-{arm}"])
        repeats[arm] = {
            "per_request_max_gap_pearson": pearson(left, right),
            "both_blocks_over_3s": sum(left[rid] > 3 and right[rid] > 3 for rid in left),
        }

    document = {}
    for arm in ARMS:
        cell = h["cells"][f"block1-{arm}"]
        request = next(row for row in cell["requests"]["requests"]
                       if row["request_id"] == DOCUMENT)
        segments = [row for row in cell["sparse_recovery"]["segments"]
                    if row["request_id"] == DOCUMENT]
        document[arm] = {
            "completed_outputs": request["outputs"],
            "max_generation_gap_s": request["max_engine_return_gap_s"],
            "preemptions": len(segments),
            "segments": [{
                "preemption_index": row["preemption_index"],
                "preempt_entered_s": row["preempt_entered_s"],
                "output_age_at_preempt_ms": 1000 * (
                    row["preempt_entered_s"] - row["last_new_output_before_preempt_s"]),
                "first_new_output_after_preempt_s": row["first_new_output_after_preempt_s"],
                "gap_s": row["last_to_first_new_output_gap_s"],
                "new_outputs_until_next_preempt_or_completion": row["returned_new_outputs"],
                "end": row["end"],
            } for row in segments],
        }
    return {
        "evidence_layer": "read-only analysis of retained native-serving trajectories",
        "short_service_counts": short_service,
        "H_repeat_correlations": repeats,
        "H_block1_document_example": document,
        "interpretation_limit": (
            "Each policy has its own execution trajectory. Correlations, short-service counts, "
            "and the selected document are descriptive; none estimates the effect of a new action."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g-analysis", type=Path, required=True)
    parser.add_argument("--h-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    g, g_sha = load(args.g_analysis)
    h, h_sha = load(args.h_analysis)
    result = analyze(g, h)
    result["sources"] = {
        "G": {"path": str(args.g_analysis), "sha256": g_sha},
        "H": {"path": str(args.h_analysis), "sha256": h_sha},
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

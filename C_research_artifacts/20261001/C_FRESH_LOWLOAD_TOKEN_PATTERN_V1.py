#!/usr/bin/env python3
"""Reuse the existing posthoc token-pattern diagnostic on all fresh/low-load cells."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from C_SUSTAINED_TOKEN_PATTERN_DIAGNOSTIC import longest_short_period

FRESH = [("native_1", "native_full"), ("bound_fifo_1", "native_max_bound"),
         ("retirement_1", "native_retirement"), ("retirement_2", "native_retirement"),
         ("bound_fifo_2", "native_max_bound"), ("native_2", "native_full")]
LOWLOAD = [("native_1", "native_full"), ("retirement_1", "native_retirement"),
           ("retirement_2", "native_retirement"), ("native_2", "native_full")]


def cell(path, group, label, arm):
    raw_path = path / "raw.json"
    data = raw_path.read_bytes()
    raw = json.loads(data)
    assert raw["status"] == "COMPLETE" and len(raw["requests"]) == 128
    rows = []
    for r in raw["requests"]:
        assert r["status"] == "completed" and r["max_output_tokens"] == 1024
        ids = r["output_token_ids"]
        span, period = longest_short_period(ids)
        constant = bool(ids) and len(set(ids)) == 1
        rows.append(dict(request_id=r["request_id"], document_id=r["document_id"],
            prompt_sha256=r["prompt_token_ids_sha256"], prompt_tokens=r["prompt_tokens"],
            output_sha256=r["output_token_ids_sha256"], output_tokens=len(ids),
            stop_reason=r["stop_reason"], longest_repeat_tokens=span,
            period_tokens=period, repeat_ge512=span >= 512,
            entire_output_single_token=constant,
            constant_1024=constant and len(ids) == 1024,
            single_token_id=ids[0] if constant else None))
    assert len({r["request_id"] for r in rows}) == 128
    result = dict(label=f"{group}/{label}", group=group, arm=arm,
        raw_path=str(raw_path), raw_sha256=hashlib.sha256(data).hexdigest(),
        arrival_scale=raw["arrival_scale"], preemptions=raw["actual_preemption_count"],
        completed_requests=len(rows), output_tokens=sum(r["output_tokens"] for r in rows),
        finish_reasons=dict(Counter(r["stop_reason"] for r in rows)),
        outputs_at_1024=sum(r["output_tokens"] == 1024 for r in rows),
        repeat_ge512=sum(r["repeat_ge512"] for r in rows),
        repeat_ge512_finish_reasons=dict(Counter(r["stop_reason"] for r in rows if r["repeat_ge512"])),
        entire_output_single_token=sum(r["entire_output_single_token"] for r in rows),
        constant_1024=sum(r["constant_1024"] for r in rows),
        constant_1024_token_ids=dict(Counter(r["single_token_id"] for r in rows if r["constant_1024"])),
        requests=rows)
    gate = path / "retirement-envelope.json"
    if gate.exists():
        g = json.loads(gate.read_text())
        result["retirement_pressure"] = {k: g[k] for k in
            ["incremental_admissions", "hold_calls", "full_bound_sum_peak", "physical_blocks_peak"]}
    return result


def compare(a, b):
    aa = {r["request_id"]: r for r in a["requests"]}
    bb = {r["request_id"]: r for r in b["requests"]}
    assert set(aa) == set(bb)
    changes = []
    for rid in sorted(aa):
        x, y = aa[rid], bb[rid]
        assert (x["document_id"], x["prompt_sha256"], x["prompt_tokens"]) == (
            y["document_id"], y["prompt_sha256"], y["prompt_tokens"])
        changes.append(dict(request_id=rid, exact_output_changed=x["output_sha256"] != y["output_sha256"],
            output_tokens_before=x["output_tokens"], output_tokens_after=y["output_tokens"],
            stop_reason_before=x["stop_reason"], stop_reason_after=y["stop_reason"],
            repeat_ge512_before=x["repeat_ge512"], repeat_ge512_after=y["repeat_ge512"],
            constant_1024_before=x["constant_1024"], constant_1024_after=y["constant_1024"],
            longest_repeat_tokens_before=x["longest_repeat_tokens"],
            longest_repeat_tokens_after=y["longest_repeat_tokens"]))
    transitions = {}
    for flag in ["repeat_ge512", "constant_1024"]:
        transitions[flag] = {name: [rid for rid in sorted(aa)
            if (aa[rid][flag], bb[rid][flag]) == pair] for name, pair in
            [("persists", (True, True)), ("leaves", (True, False)),
             ("enters", (False, True)), ("neither", (False, False))]}
    return dict(reference=a["label"], candidate=b["label"], transitions=transitions,
        exact_output_changed=sum(r["exact_output_changed"] for r in changes),
        output_length_changed=sum(r["output_tokens_before"] != r["output_tokens_after"] for r in changes),
        finish_reason_changed=sum(r["stop_reason_before"] != r["stop_reason_after"] for r in changes),
        per_request_changes=changes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cells = []
    for group, folder, order in [("fresh", "c-native-retirement-fresh-v1", FRESH),
                                 ("lowload", "c-native-retirement-lowload-v1", LOWLOAD)]:
        for label, arm in order:
            cells.append(cell(args.artifact_root / folder / label / arm, group, label, arm))
    # Retain the single later donor pilot separately; no cross-load donor pair exists.
    cells.append(cell(args.artifact_root / "c-native-temporal-cap-donor-pilot-v1" /
        "native_temporal_donor_1", "donor", "native_temporal_donor_1", "native_temporal_donor"))
    lookup = {c["label"]: c for c in cells}
    identity = {(r["request_id"], r["document_id"], r["prompt_sha256"], r["prompt_tokens"])
                for r in cells[0]["requests"]}
    assert all({(r["request_id"], r["document_id"], r["prompt_sha256"], r["prompt_tokens"])
                for r in c["requests"]} == identity for c in cells)
    pairs = [compare(lookup[f"fresh/{label}"], lookup[f"lowload/{label}"])
             for label, _ in LOWLOAD]
    matrix = []
    indexed = {c["label"]: {r["request_id"]: r for r in c["requests"]} for c in cells}
    for rid in sorted(r[0] for r in identity):
        matrix.append(dict(request_id=rid,
            repeat_ge512_cells=[c["label"] for c in cells if indexed[c["label"]][rid]["repeat_ge512"]],
            constant_1024_cells=[c["label"] for c in cells if indexed[c["label"]][rid]["constant_1024"]]))
    groups = {}
    for group in ["fresh", "lowload", "donor"]:
        selected = [c for c in cells if c["group"] == group]
        groups[group] = dict(cells=len(selected), executions=sum(c["completed_requests"] for c in selected),
            repeat_ge512_executions=sum(c["repeat_ge512"] for c in selected),
            constant_1024_executions=sum(c["constant_1024"] for c in selected),
            finish_reasons=dict(sum((Counter(c["finish_reasons"]) for c in selected), Counter())))
        for flag in ["repeat_ge512", "constant_1024"]:
            sets = [{r["request_id"] for r in c["requests"] if r[flag]} for c in selected]
            groups[group][f"{flag}_all_cells_request_ids"] = sorted(set.intersection(*sets))
            groups[group][f"{flag}_any_cell_request_ids"] = sorted(set.union(*sets))
    result = dict(schema="c-fresh-lowload-token-pattern-v1", status="POST_HOC_SAME_COHORT_DIAGNOSTIC_ONLY",
        algorithm="Imported unchanged longest_short_period from C_SUSTAINED_TOKEN_PATTERN_DIAGNOSTIC.py",
        criteria="Period k=1..16; longest contiguous span of at least two copies; count span>=512 tokens. Constant-1024 means all 1024 output IDs identical. Existing posthoc criteria, not a semantic-quality score.",
        coverage="All six fresh cells, all four lowload cells; later donor pilot separately retained. All 128 request/document/prompt identities align.",
        cells=cells, groups=groups, same_arm_fresh_to_lowload=pairs,
        request_pattern_matrix=matrix,
        limits="Same viewed articles, fixed generation cap and decoding; arrival scale changes concurrency and numerical/output trajectories. Temporal separation and repeated executions are confounded. No independent quality judgement, causal pressure attribution, useful-output goodput, request exclusion, or new threshold.")
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    for c in cells:
        print(c["label"], c["repeat_ge512"], c["constant_1024"], c["finish_reasons"], c["output_tokens"])
    for x in pairs:
        print(x["reference"], "->", x["candidate"],
              {k: {a: len(b) for a, b in v.items()} for k, v in x["transitions"].items()},
              "output/length/stop changes", x["exact_output_changed"], x["output_length_changed"], x["finish_reason_changed"])


if __name__ == "__main__":
    main()

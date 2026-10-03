#!/usr/bin/env python3
"""Describe retained step-six reruns; no fit, confidence interval, or exact KV claim."""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "experiments/admission_capacity"))
from analyze_capacity import analyze, read_json

TRIALS = "a0_c0_hold a0_c32_hold b0_c32_up b0_c0_up b1_c0_up b1_c32_up a1_c32_hold a1_c0_hold".split()
TIMING = ("recent_model_call_s", "recent_iteration_s", "recent_itl_s", "oldest_wait_s")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def differences(a, b):
    return {k: {"a": a[k], "b": b[k], "b_minus_a": b[k] - a[k],
                "relative_to_a": (b[k] - a[k]) / a[k] if a[k] else None}
            for k in a.keys() & b.keys() if type(a[k]) in (int, float) and type(b[k]) in (int, float)}


def capture(raw, plan, workload):
    actions = [a for a in raw["actions"] if a.get("trigger") == "completed_decode_steps"]
    require(len(actions) == 1, "need exactly one completed-step action")
    action = actions[0]
    require(action["requested_completed_steps"] == action["completed_steps"] == 6, "wrong action frontier")
    s, steps = action["pre_action"], raw["steps"]
    t, rows = s["observed_s"], {r["request_id"]: r for r in raw["requests"]}
    slots = {r["request_id"]: i for i, r in enumerate(workload["source_requests"])}
    active = [rid for rid, r in rows.items() if r["admission_s"] is not None
              and r["admission_s"] <= t and (r["completion_s"] is None or r["completion_s"] > t)]
    waiting = [rid for rid, r in rows.items() if r["arrival_s"] <= t
               and (r["admission_s"] is None or r["admission_s"] > t)]
    future = [rid for rid, r in rows.items() if r["arrival_s"] > t]
    require((active, waiting, future) == (s["active_request_ids"], s["waiting_request_ids"], s["future_request_ids"]), "snapshot queue/state mismatch")
    require((len(active), len(waiting), len(future)) == (s["actual_active"], s["waiting_requests"], s["future_requests"]), "snapshot count mismatch")
    counts = {rid: 0 for rid in rows}
    for i, step in enumerate(steps):
        expected = {rid for rid, r in rows.items() if r["admission_s"] is not None
                    and r["admission_s"] <= step["start_s"] < r["completion_s"]}
        require(step["step"] == i and set(step["request_ids"]) == expected, "step/full-active identity mismatch")
        require(len(step["request_ids"]) == len(expected) == step["ordinary"]["actual_active"] == step["ordinary"]["decode_requests"], "active width mismatch")
        if i < 6:
            for rid, number in zip(step["request_ids"], step["decode_steps"]):
                counts[rid] += 1
                require(number == counts[rid], "prefix decode progress mismatch")
    require(s["active_decode_steps"] == [counts[rid] for rid in active], "snapshot decode progress mismatch")
    require(s["kv_lengths"] == [rows[rid]["prompt_tokens"] + counts[rid] for rid in active], "snapshot KV lengths mismatch")
    require(s["history_step_indices"] == [2, 3, 4, 5], "history cutoff mismatch")
    latest = s["latest_completed_step"]
    require(all(latest[k] == steps[5][k] for k in ("step", "request_ids", "decode_steps", "completed_s", "pressure")), "latest pressure/state mismatch")
    require(latest["decode_requests"] == len(steps[5]["request_ids"]), "latest width mismatch")
    require(steps[5]["completed_s"] <= t <= action["applied_s"], "pre-action time order mismatch")
    pressure = latest["pressure"]
    require(([p["layer"] for p in pressure] == list(range(16))) if plan["telemetry"] else not pressure, "pressure layer coverage mismatch")
    require(all(p[k] is not None and math.isfinite(p[k]) for p in pressure for k in ("U", "C")), "invalid pressure")
    ordinary = {k: s[k] for k in (*TIMING, "target_cap", "actual_active", "waiting_requests", "future_requests")}
    ordinary.update(decode_batch_width=latest["decode_requests"], logical_kv_tokens=sum(s["kv_lengths"]),
                    padded_kv_tokens=max(s["kv_lengths"], default=0) * len(active))
    prefix = {rid: r["output_token_ids"][:counts[rid] + 1] for rid, r in rows.items()
              if r["admission_s"] is not None and r["admission_s"] <= t}
    shape = dict(active_slots=[slots[r] for r in active], decode_steps=s["active_decode_steps"], kv_lengths=s["kv_lengths"],
                 waiting_slots=[slots[r] for r in waiting], future_slots=[slots[r] for r in future],
                 last_batch_slots=[slots[r] for r in latest["request_ids"]], last_decode_steps=latest["decode_steps"],
                 history_widths=[len(steps[i]["request_ids"]) for i in s["history_step_indices"]])
    return dict(action=action, ordinary=ordinary, slot_state=shape, token_prefix=prefix,
                prefix_membership=[s["request_ids"] for s in steps[:6]], pressure_layers=pressure,
                pressure_means={k: sum(p[k] for p in pressure) / 16 if pressure else None for k in ("U", "C")})


def summarize(root):
    cells, validations, documents = [], [], set()
    for trial in TRIALS:
        run = root / "gpu_results" / trial
        config, workload = read_json(run / "config.json"), read_json(run / "workload.json")
        documents.update(r["document_id"] for r in workload["source_requests"])
        checked = analyze(run)
        require(checked["complete_scan"] and checked["identity_validation"] == "passed_present_cells_only", f"invalid/incomplete run: {trial}")
        require(not checked["static_comparisons"] and config.get("cap_schedule"), "dynamic run entered static comparison")
        require(len(checked["cells"]) == 4, "expected four regime/telemetry cells")
        validations.append(dict(trial=trial, verdict=checked["verdict"], issues=checked["issues"]))
        block, cohort, role = trial.split("_")
        expected_target = 6 if role == "hold" else 8
        configured = json.loads(config["step_action"]) if isinstance(config["step_action"], str) else config["step_action"]
        require(configured == [6, expected_target], "configured action differs from trial role")
        for c in checked["cells"]:
            index, plan = c["cell"], c["plan"]
            require(read_json(run / f"checks-{index:03d}.json").get("status") == "PASS", "isolation sidecar not PASS")
            raw, metrics = read_json(run / f"cell-{index:03d}.json"), c["metrics"]
            frontier = capture(raw, plan, workload)
            require(frontier["action"]["target"] == expected_target and frontier["action"]["previous_target"] == 6, "applied action differs from trial role")
            numeric = {k: metrics[k] for k in ("goodput_rps", "throughput_rps", "observation_duration_s", "n_slo_pass", "slo_attainment")}
            numeric.update({f"{metric}_{q}_s": metrics["latency_s"][metric][q] for metric in ("ttft", "tpot") for q in ("p50", "p95", "p99")})
            numeric.update({f"n_{metric}_pass": sum(r[f"{metric}_pass"] for r in metrics["per_request"]) for metric in ("ttft", "tpot")})
            numeric["effect_delay_s"] = frontier["action"]["effect_delay_s"]
            cells.append(dict(id=f"{trial}/cell-{index:03d}", cohort=cohort, repeat=int(block[1:]), role=role,
                              regime=plan["regime"], telemetry=plan["telemetry"], numeric=numeric, metrics=metrics,
                              exposure=c["execution_exposure"], **frontier))
    lookup = {(c["cohort"], c["repeat"], c["regime"], c["telemetry"], c["role"]): c for c in cells}
    require(len(lookup) == 32, "missing or duplicate planned cell")
    pairs, cohorts = [], []
    for key, a in lookup.items():
        if a["role"] == "hold":
            b = lookup[(*key[:-1], "up")]
            pairs.append(dict(hold=a["id"], up=b["id"], token_prefix_equal=a["token_prefix"] == b["token_prefix"],
                prefix_membership_equal=a["prefix_membership"] == b["prefix_membership"],
                snapshot_fields_equal={k: a["action"]["pre_action"][k] == b["action"]["pre_action"][k] for k in
                    ("active_request_ids", "active_decode_steps", "kv_lengths", "waiting_request_ids", "future_request_ids")},
                state_fields_equal={k: a["slot_state"][k] == b["slot_state"][k] for k in a["slot_state"]},
                ordinary_up_minus_hold=differences(a["ordinary"], b["ordinary"]),
                pressure_up_minus_hold=differences(a["pressure_means"], b["pressure_means"]),
                outcomes_up_minus_hold=differences(a["numeric"], b["numeric"])))
        if a["cohort"] == "c0":
            b = lookup[("c32", *key[1:])]
            cohorts.append(dict(c0=a["id"], c32=b["id"],
                slot_state_equal={k: a["slot_state"][k] == b["slot_state"][k] for k in a["slot_state"]},
                ordinary_c32_minus_c0=differences(a["ordinary"], b["ordinary"]),
                pressure_c32_minus_c0=differences(a["pressure_means"], b["pressure_means"])))
    return dict(evidence_type="CUSTOM_CONTINUOUS_RUNTIME", interpretation="Descriptive independent reruns; matching observable prefixes does not prove identical KV tensors. No fitting or mechanism verdict. OFF has no pressure measurement; ON pressure is never assigned to OFF.",
                pressure_rule="Latest completed step only, arithmetic mean across the fixed 16 layers; history widths retained separately.",
                unique_document_ids=len(documents), cells=cells, validations=validations, hold_up_pairs=pairs, cross_cohort_pairs=cohorts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "analysis.json")
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite an existing analysis")
    result = summarize(args.run_dir)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Wrote {len(result['cells'])} retained cells: {args.output}")


if __name__ == "__main__":
    main()

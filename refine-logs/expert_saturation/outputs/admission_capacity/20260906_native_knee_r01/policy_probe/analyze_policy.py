#!/usr/bin/env python3
"""Retained native static/shadow/feedback episodes; raw request and causal checks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from statistics import median

PARENT = Path(__file__).resolve().parent.parent / "analyze_knee.py"
spec = importlib.util.spec_from_file_location("knee", PARENT)
knee = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knee)
read, same, require = knee.read, knee.same, knee.require


def causal_check(raw, config):
    """Validate actual state/observation timing without generating a policy trace."""
    steps = raw["scheduler_steps"]
    rows = {r["request_id"]: r for r in raw["requests"]}
    decisions, observations = raw.get("feedback_decisions", []), raw.get("feedback_observations", [])
    policy = raw.get("policy", "static")
    controlled = policy in ("shadow", "feedback")
    require(len(decisions) == (len(steps) if controlled else 0), "decision/scheduler count mismatch")
    if not controlled:
        require(not observations and not raw.get("actions"), "static arm contains feedback activity")
    observed_itls = defaultdict(dict)
    for rid, row in rows.items():
        for a, b in zip(row["token_times_s"], row["token_times_s"][1:]):
            observed_itls[b][rid] = b - a
    prior_completed = 0
    for obs in observations:
        require(prior_completed < obs["completed_step"] <= len(steps), "observation step order mismatch")
        prior_completed = obs["completed_step"]
        require(obs["received_s"] <= obs["available_s"] <= raw["observation_end_s"], "invalid observation availability")
        require(same(obs["request_itls_s"], observed_itls[obs["received_s"]]), "ITL observation differs from host token data")
        require(same(obs["step_median_itl_s"], median(obs["request_itls_s"].values())), "ITL median mismatch")
        source_step = steps[obs["completed_step"] - 1]
        require(source_step["end_s"] <= obs["received_s"], "observation precedes completed scheduling")
        if obs["completed_step"] < len(steps):
            require(obs["available_s"] <= steps[obs["completed_step"]]["start_s"], "observation availability crossed next step")
    computed, ever_scheduled = {}, set()
    checked_existing = 0
    for index, step in enumerate(steps):
        require(step["step"] == index, "step identity mismatch")
        require(not step["preempted_request_ids"], "preemption violates action scope")
        require(all(r["computed_adjustment"] == 0 for r in step["scheduled"]), "KV/computed adjustment violates action scope")
        existing = {rid for rid in ever_scheduled if computed[rid] >= rows[rid]["prompt_tokens"] and
                    (rows[rid]["completion_s"] is None or rows[rid]["completion_s"] > step["start_s"])}
        decoded = {r["request_id"] for r in step["scheduled"] if r["decode_tokens"] > 0}
        require(existing <= decoded, "an existing decode request did not advance")
        checked_existing += len(existing)
        if controlled:
            d = decisions[index]
            require(d["decision_index"] == index == step["decision_index"] and d["completed_steps"] == index,
                    "decision/step alignment mismatch")
            require(set(step["existing_decode_request_ids"]) == existing and step["existing_decode_all_scheduled"],
                    "recorded existing decode set differs from independently reconstructed set")
            require(step["start_s"] == d["decision_start_s"] <= d["decision_end_s"] <= d["applied_s"] <= step["end_s"],
                    "decision/application/scheduler time order mismatch")
            window = [o for o in observations if o["completed_step"] <= index][-4:]
            require(d["window_completed_steps"] == [o["completed_step"] for o in window], "observer cutoff used wrong history window")
            expected_recent = median(o["step_median_itl_s"] for o in window) if len(window) == 4 else None
            require(same(d["recent_step_median_itl_s"], expected_recent), "decision signal differs from available window")
            require(d["signal_cutoff_s"] == (window[-1]["received_s"] if window else None) and
                    d["signal_available_s"] == (window[-1]["available_s"] if window else None), "signal cutoff mismatch")
            if window:
                require(window[-1]["available_s"] <= d["decision_start_s"], "future observation selected an action")
            require(d["active_before"] == step["running_before"] and d["waiting_before"] == step["waiting_before"],
                    "decision state differs from actual scheduler")
            target = d["target_cap"]
            require(target == step["target_cap"], "decision target differs from scheduler target")
            limit = max(target, step["running_before"])
            require(d["effective_scheduler_limit"] == step["effective_scheduler_limit"] == limit and
                    step["actual_active"] <= limit <= config["engine_max_num_seqs"], "nonpreemptive clamp mismatch")
            if policy == "shadow":
                require(target == raw["target_cap"] and not d["applied_change"], "shadow altered admission target")
        else:
            require(step["target_cap"] == raw["target_cap"] and step["actual_active"] <= raw["target_cap"],
                    "static cap changed or was exceeded")
        for row in step["scheduled"]:
            computed[row["request_id"]] = row["computed_after"]
            ever_scheduled.add(row["request_id"])
    require(raw.get("actions", []) == [d for d in decisions if d["intent_changed"]], "action list differs from actual decisions")
    action_records = []
    changes = [d for d in decisions if d["applied_change"]]
    for i, action in enumerate(raw.get("actions", [])):
        following = next((d for d in changes if d["decision_index"] > action["decision_index"]), None)
        segment = [s for s in steps if s["step"] >= action["decision_index"] and
                   (following is None or s["step"] < following["decision_index"])]
        target = action["target_cap"]
        down = action["intent_target"] < action["intent_before"]
        reached = next((s for s in segment if (s["actual_active"] <= target if down else s["actual_active"] >= target)), None)
        binding = next((s for s in segment if s["waiting_before"] > 0 and
                        s["running_before"] >= target and target < raw["target_cap"]), None)
        applied = action["applied_change"]
        action_records.append(dict(decision_index=action["decision_index"], intent_before=action["intent_before"],
            intent_target=action["intent_target"], actual_target=target, applied_change=applied,
            applied_s=action["applied_s"], active_before=action["active_before"], waiting_before=action["waiting_before"],
            actual_bound_reached_s=reached["end_s"] if applied and reached else None,
            delay_to_actual_bound_s=reached["end_s"] - action["applied_s"] if applied and reached else None,
            first_binding_opportunity_s=binding["start_s"] if applied and binding else None,
            natural_drain_steps=sum(s["actual_active"] > target for s in segment) if applied else 0,
            status="SHADOW_INTENT_ONLY" if not applied else "BOUND_REACHED" if reached else
                   "SUPERSEDED_BEFORE_BOUND" if following else "EPISODE_ENDED_BEFORE_BOUND"))
    return dict(existing_decode_advances_checked=checked_existing,
        decision_count=len(decisions), observation_count=len(observations),
        applied_action_count=len(changes), shadow_intent_count=sum(not d["applied_change"] for d in raw.get("actions", [])),
        decision_compute_s=sum(d["decision_end_s"] - d["decision_start_s"] for d in decisions),
        decision_and_apply_s=sum(d["applied_s"] - d["decision_start_s"] for d in decisions),
        output_processing_to_signal_available_s=sum(o["available_s"] - o["received_s"] for o in observations),
        target_step_counts=dict(sorted(Counter(s["target_cap"] for s in steps).items())),
        active_above_target_steps=sum(s["actual_active"] > s["target_cap"] for s in steps),
        actions=action_records,
        semantics="Bound reached describes actual active <= down target or >= up target; binding opportunity is not a counterfactual effect. Cost fields already reside inside request wall time.")


def summarize(raw, config, workload, plan, saved):
    require(raw["plan"] == plan and raw["policy"] == plan["policy"], "raw/config policy plan mismatch")
    require(raw["target_cap"] == plan["cap"] == raw["admission_cap_state"]["admission_cap"], "initial cap mismatch")
    require(raw["admission_cap_state"]["engine_max_num_seqs"] == config["engine_max_num_seqs"], "engine capacity mismatch")
    require(len(raw["requests"]) == config["requests"] == len(workload["source_requests"]), "cohort size mismatch")
    indexed = {r["request_id"]: r for r in raw["requests"]}
    require(len(indexed) == config["requests"], "duplicate request identity")
    for source, arrival in zip(workload["source_requests"], workload["arrival_traces_s"][plan["regime"]]):
        row = indexed[source["request_id"]]
        require(row["document_id"] == source["document_id"] and row["prompt_token_ids_sha256"] == source["prompt_token_ids_sha256"], "input identity mismatch")
        require(same(row["arrival_s"], arrival * plan["arrival_scale"]), "arrival mismatch")
        require(row["status"] != "completed" or len(row["output_token_ids"]) == config["output_tokens"], "fixed output length mismatch")
    result = knee.legacy.summarize_cell(raw, config)
    metrics = result["metrics"]
    require(saved is not None and same(metrics, {k: saved.get(k) for k in metrics}), "saved main metrics differ from raw")
    reference = knee.summarize_episode_requests(raw["requests"], observation_end_s=raw["observation_end_s"], **config["reference_slo"])
    require(same(reference, saved.get("reference_slo_metrics")), "saved reference metrics differ from raw")
    result.update(saved_main_and_reference_equal=True, reference_slo_metrics=reference, n_slo_pass=metrics["n_slo_pass"],
        violation_counts=dict(ttft_only=sum(not r["ttft_pass"] and r["tpot_pass"] for r in metrics["per_request"]),
            tpot_only=sum(r["ttft_pass"] and not r["tpot_pass"] for r in metrics["per_request"]),
            both=sum(not r["ttft_pass"] and not r["tpot_pass"] for r in metrics["per_request"])),
        waiting_after_positive_steps=sum(s["waiting_requests"] > 0 for s in raw["scheduler_steps"]),
        decode_width_step_counts=dict(sorted(Counter(s["decode_requests"] for s in raw["scheduler_steps"]).items())),
        causal_and_action=causal_check(raw, config))
    result["slo_grid_descriptive_only"] = []
    for ttft, tpot in knee.GRID:
        m = knee.summarize_episode_requests(raw["requests"], observation_end_s=raw["observation_end_s"], ttft_slo_s=ttft, tpot_slo_s=tpot)
        result["slo_grid_descriptive_only"].append(dict(ttft_slo_s=ttft, tpot_slo_s=tpot, n_slo_pass=m["n_slo_pass"], goodput_rps=m["goodput_rps"]))
    return result


def analyze(root, campaign):
    cells, configs, engines, issues = [], {}, {}, []
    for trial in campaign["trials"]:
        group, directory = trial["group"], root / trial["group"]
        if not (directory / "config.json").exists():
            issues.append(f"UNRUN trial {group}")
            continue
        config = configs[group] = read(directory / "config.json")
        workload = read(directory / "workload.json")
        workload_ok = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() == config["workload_sha256"]
        engines[group] = read(directory / "engine_args.json") if (directory / "engine_args.json").exists() else None
        for index, plan in enumerate(config["plans"]):
            cell = dict(id=f"{group}/cell-{index:03d}", group=group, block=trial["block"], plan=plan,
                        cohort=config["workload_sha256"], arm=f"{plan['policy']}{plan['cap']}", status="MISSING", qualified=False)
            cells.append(cell)
            path = directory / f"cell-{index:03d}.json"
            if not path.exists():
                continue
            try:
                require(workload_ok, "workload hash changed")
                raw = read(path)
                saved = directory / f"metrics-{index:03d}.json"
                cell.update(summarize(raw, config, workload, plan, read(saved) if saved.exists() else None), status=raw["status"], error=raw.get("error"))
                check = directory / f"checks-{index:03d}.json"
                cell["gpu_boundary_check"] = read(check).get("status") if check.exists() else "MISSING"
                cell["qualified"] = (cell["status"] == "COMPLETE" and cell["metrics"]["n_completed"] == config["requests"] and
                                      cell["gpu_boundary_check"] == "PASS" and cell["token_level_timing_resolved"])
            except (KeyError, ValueError, TypeError) as exc:
                cell.update(status="INVALID", error=str(exc), qualified=False)
    paired, conditions = [], []
    condition_keys = sorted({(c["cohort"], c["plan"]["arrival_scale"], c["plan"]["regime"]) for c in cells})
    for cohort, scale, regime in condition_keys:
        selected = [c for c in cells if (c["cohort"], c["plan"]["arrival_scale"], c["plan"]["regime"]) == (cohort, scale, regime)]
        valid = [c for c in selected if c["qualified"]]
        arm_order = {c["arm"]: ({"static": 0, "shadow": 1, "feedback": 2}[c["plan"]["policy"]], c["plan"]["cap"])
                     for c in selected}
        arms = sorted(arm_order, key=arm_order.get)
        conditions.append(dict(cohort=cohort, arrival_scale=scale, regime=regime,
            arm_ranges={arm: {k: knee.legacy.interval([c[k] for c in valid if c["arm"] == arm]) for k in (*knee.KEYS, "n_slo_pass")} for arm in arms}))
        for block, repeat in sorted({(c["block"], c["plan"]["repeat"]) for c in selected}):
            rows = [c for c in selected if (c["block"], c["plan"]["repeat"]) == (block, repeat)]
            for base_arm, action_arm in itertools.combinations(arms, 2):
                # Preserve every arm comparison; direction is explicit in the names.
                a, b = [c for c in rows if c["arm"] == base_arm], [c for c in rows if c["arm"] == action_arm]
                pair = dict(block=block, repeat=repeat, cohort=cohort, arrival_scale=scale, regime=regime,
                            base_arm=base_arm, action_arm=action_arm, status="MISSING_OR_UNQUALIFIED")
                if len(a) == len(b) == 1:
                    x, y = a[0], b[0]
                    gx, gy = x["group"], y["group"]
                    compatible = all(configs[gx].get(k) == configs[gy].get(k) for k in knee.FAIR_CONFIG)
                    compatible = compatible and engines[gx] is not None and engines[gx] == engines[gy]
                    pair.update(base_id=x["id"], action_id=y["id"], fair_config_equal=compatible)
                    if compatible and x["qualified"] and y["qualified"]:
                        pair.update(status="DESCRIPTIVE_PAIRED_RERUN", action_minus_base=knee.delta(x, y))
                paired.append(pair)
    expected = campaign.get("expected_episodes", 24)
    complete = len(cells) == expected and all(c["qualified"] for c in cells) and len(configs) == len(campaign["trials"])
    return dict(status="MEASUREMENT_ONLY" if complete else "PARTIAL_OR_UNRUN", complete_expected_capture=complete,
        expected_episodes=expected, completed_episodes=sum(c["status"] == "COMPLETE" for c in cells),
        qualified_episodes=sum(c["qualified"] for c in cells),
        request_executions=sum(c.get("metrics", {}).get("n_completed", 0) for c in cells),
        engine_args_equal=bool(engines) and all(e is not None and e == next(iter(engines.values())) for e in engines.values()),
        cells=cells, conditions=conditions, all_arm_pairs=paired, issues=issues,
        limits=["All main and reference request metrics are recomputed from retained host timestamps; capture and feedback costs remain in their measured wall-time denominator.",
            "Static 8/12/16/32 are all reported. The best observed static point is a descriptive comparison, not a trained or held-out policy.",
            "Shadow uses the feedback observer/decision code but keeps admission at 32; shadow intent changes are not applied actions.",
            "Existing decode sets are reconstructed from prior scheduled tokens and request completion, then checked for advancement at every step.",
            "Recomputed observation windows contain only completed steps available before each decision; action application precedes that scheduler result.",
            "Request-level mean TPOT, step median ITL feedback, and worst token ITL are distinct quantities.",
            "Action-bound timing and binding opportunities do not identify an exact counterfactual admission-delay effect.",
            "All arms independently execute future state; neither identical hidden state nor equal output semantics is asserted.",
            "Two counterbalanced repeats of the same finite input are descriptive; no step-level independent-sample inference, dynamic Oracle, or expert-signal claim.",
            "The frozen nine-point SLO grid cannot select the canonical policy or change the feedback threshold."])


def report(result):
    def fmt(x):
        return "—" if x is None else f"{x:.5f}"
    lines = [f"# Native simple policy probe: {result['status']}", "",
        f"Complete {result['completed_episodes']}/{result['expected_episodes']}; qualified {result['qualified_episodes']}; requests {result['request_executions']}.", "",
        "| Cell | Regime | Arm | Status | Pass | Throughput | Goodput | TTFT / TPOT p50 s | Max active / decode / queue | Applied actions |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in result["cells"]:
        lead = f"| {c['id']} | {c['plan']['regime']} | {c['arm']} | {c['status']}"
        if "metrics" not in c:
            lines.append(lead + " | — | — | — | — | — | — |")
        else:
            lines.append(lead + f" | {c['n_slo_pass']}/{c['metrics']['n_planned']} | {fmt(c['throughput_rps'])} | {fmt(c['goodput_rps'])} | "
                f"{fmt(c['ttft_p50_s'])} / {fmt(c['tpot_p50_s'])} | {c['max_active']} / {c['max_decode_requests']} / {c['max_native_waiting_after']} | "
                f"{c['causal_and_action']['applied_action_count']} |")
    lines += ["", "All arm pair comparisons and descriptive SLO grid are retained in analysis.json.", "", *result["limits"], "", *result["issues"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--campaign-plan", type=Path, default=Path(__file__).with_name("campaign_plan.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.results_dir, read(args.campaign_plan))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(report(result))


if __name__ == "__main__":
    main()

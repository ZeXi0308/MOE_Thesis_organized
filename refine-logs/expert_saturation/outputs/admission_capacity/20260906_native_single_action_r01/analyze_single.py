#!/usr/bin/env python3
"""Analyze all twelve frozen one-shot episodes; observed prefixes are not snapshots."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

PRIOR = Path(__file__).resolve().parent.parent / "20260906_native_knee_r01/policy_probe/analyze_policy.py"
spec = importlib.util.spec_from_file_location("single_action_prior_summary", PRIOR)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
read, require = prior.read, prior.require
ARMS = ((32, "single_shadow"), (32, "single_down"), (16, "static"))


def single_semantics(raw, config):
    policy = raw["policy"]
    require(policy in {p for _, p in ARMS}, "unexpected one-shot policy")
    if policy == "static":
        require(raw["target_cap"] == 16, "static comparator must be static16")
        return dict(status="STATIC_BASELINE", trigger=None)
    require(raw["target_cap"] == 32 and config["single_down_target"] == 16, "one-shot must be 32->16")
    trigger, intent, last_change = None, 32, 0
    for d in raw["feedback_decisions"]:
        recent = d["recent_step_median_itl_s"]
        cooldown = d["completed_steps"] - last_change
        fires = trigger is None and recent is not None and cooldown >= 4 and recent > config["tpot_slo_s"]
        previous = intent
        if fires:
            trigger, intent, last_change = d, 16, d["completed_steps"]
        require(d["policy"] == policy and d["cooldown_steps"] == cooldown, "single decision identity/cooldown mismatch")
        require(d["intent_before"] == previous and d["intent_target"] == intent and
                d["intent_changed"] == fires, "single intent did not latch at the first available trigger")
        require(d["applied_change"] == (fires and policy == "single_down") and
                d["target_cap"] == (intent if policy == "single_down" else 32), "shadow applied an action or single target changed")
    return dict(status="TRIGGER_NOT_OBSERVED" if trigger is None else
                "SHADOW_INTENT_ONLY" if policy == "single_shadow" else "ONE_ACTION_APPLIED", trigger=trigger)


def summarize(raw, config, workload, plan, saved):
    require(raw["plan"] == plan and raw["policy"] == plan["policy"], "actual one-shot plan mismatch")
    # Adapt only the legacy check's observer category on a shallow analysis copy.
    # Tokens, timing, decisions and applied-change flags remain the actual raw data.
    category = {"single_shadow": "shadow", "single_down": "feedback", "static": "static"}[raw["policy"]]
    legacy_plan = dict(plan, policy=category)
    result = prior.summarize(dict(raw, policy=category, plan=legacy_plan), config, workload, legacy_plan, saved)
    result["single_action"] = single_semantics(raw, config)
    return result


def observed_prefix(raw, trigger):
    cutoff, index = trigger["decision_start_s"], trigger["decision_index"]
    return dict(decision_index=index, active=trigger["active_before"], waiting=trigger["waiting_before"],
        window_steps=trigger["window_completed_steps"],
        schedule=[[(r["request_id"], r["computed_before"], r["computed_after"], r["scheduled_tokens"])
                   for r in step["scheduled"]] for step in raw["scheduler_steps"][:index]],
        submitted=sorted(r["request_id"] for r in raw["requests"]
                         if r["admission_s"] is not None and r["admission_s"] <= cutoff),
        delivered=sorted((r["request_id"], [token for token, at in zip(r["output_token_ids"], r["token_times_s"])
                                           if at <= cutoff]) for r in raw["requests"]))


def analyze(root):
    cells, raw_by_id, configs, engines, issues = [], {}, {}, {}, []
    for group, reverse in (("forward", False), ("reverse", True)):
        directory = root / group
        available = (directory / "config.json").exists() and (directory / "workload.json").exists()
        config, workload = (read(directory / "config.json"), read(directory / "workload.json")) if available else ({}, {})
        configs[group] = config
        engines[group] = read(directory / "engine_args.json") if (directory / "engine_args.json").exists() else None
        plans = [dict(repeat=0, arrival_scale=1.0, regime=regime, cap=cap, policy=policy)
                 for cap, policy in (reversed(ARMS) if reverse else ARMS)
                 for regime in (("bursty", "steady") if reverse else ("steady", "bursty"))]
        for index, plan in enumerate(plans):
            cell = dict(id=f"{group}/cell-{index:03d}", group=group, plan=plan,
                        arm=plan["policy"], status="MISSING", qualified=False)
            cells.append(cell)
            path = directory / f"cell-{index:03d}.json"
            if not available or not path.exists():
                continue
            try:
                require(config["plans"] == plans and config["single_action_probe"], "frozen six-cell plan drift")
                require(config["engine_max_num_seqs"] == 32 and config["requests"] == 32 and
                        config["prompt_tokens"] == config["output_tokens"] == 128, "frozen engine/workload shape drift")
                require(config["ttft_slo_s"] == 0.2 and config["tpot_slo_s"] == 0.009, "frozen primary SLO drift")
                require(hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() == config["workload_sha256"],
                        "prepared workload hash changed")
                raw = raw_by_id[cell["id"]] = read(path)
                cell.update(status=raw["status"], capture_status=raw["status"], error=raw.get("error"))
                saved_path = directory / f"metrics-{index:03d}.json"
                cell.update(summarize(raw, config, workload, plan, read(saved_path) if saved_path.exists() else None))
                check = directory / f"checks-{index:03d}.json"
                cell["gpu_boundary_check"] = read(check).get("status") if check.exists() else "MISSING"
                cell["qualified"] = (raw["status"] == "COMPLETE" and cell["metrics"]["n_completed"] == 32 and
                                     cell["gpu_boundary_check"] == "PASS" and cell["token_level_timing_resolved"])
            except (KeyError, ValueError, TypeError) as exc:
                cell.update(status=cell["capture_status"] if cell.get("capture_status") not in (None, "COMPLETE") else "INVALID",
                            analysis_error=str(exc), qualified=False)
    compatible = all(configs["forward"].get(k) == configs["reverse"].get(k) for k in prior.knee.FAIR_CONFIG)
    compatible = compatible and engines["forward"] is not None and engines["forward"] == engines["reverse"]
    if not compatible:
        issues.append("engine/fair configuration equality unavailable or failed")
    comparisons = []
    for group in ("forward", "reverse"):
        for regime in ("steady", "bursty"):
            selected = {c["arm"]: c for c in cells if c["group"] == group and c["plan"]["regime"] == regime}
            comparison = dict(group=group, regime=regime, status="INCOMPLETE_OR_UNQUALIFIED",
                              exact_prestate_snapshot=False, dynamic_oracle="UNRUN")
            if compatible and all(c["qualified"] for c in selected.values()):
                hold, down, static = (selected[p] for p in ("single_shadow", "single_down", "static"))
                comparison.update(status="DESCRIPTIVE_PAIRED_RERUN",
                    down_minus_hold=prior.knee.delta(hold, down), down_minus_static16=prior.knee.delta(static, down))
                triggers = [c["single_action"]["trigger"] for c in (hold, down)]
                comparison["trigger_status"] = [c["single_action"]["status"] for c in (hold, down)]
                if any(t is None for t in triggers):
                    comparison["action_qualification"] = "TRIGGER_NOT_OBSERVED_IN_BOTH_ARMS"
                else:
                    prefixes = [observed_prefix(raw_by_id[c["id"]], t) for c, t in zip((hold, down), triggers)]
                    matches = {key: prefixes[0][key] == prefixes[1][key] for key in prefixes[0]}
                    comparison.update(observed_prefix_fields_equal=matches,
                        action_qualification="OBSERVED_PREFIX_ALIGNED_NOT_SNAPSHOT" if all(matches.values()) else "PREFIX_NOT_ALIGNED",
                        trigger_times_s=[t["decision_start_s"] for t in triggers],
                        trigger_signals_s=[t["recent_step_median_itl_s"] for t in triggers])
            comparisons.append(comparison)
    complete = compatible and all(c["qualified"] for c in cells)
    return dict(status="MEASUREMENT_ONLY" if complete else "PARTIAL_OR_UNRUN", expected_episodes=12,
        completed_episodes=sum(c["status"] == "COMPLETE" for c in cells), qualified_episodes=sum(c["qualified"] for c in cells),
        complete_expected_capture=complete, engine_args_equal=compatible, cells=cells, comparisons=comparisons, issues=issues,
        limits=["Every planned forward/reverse arm is retained; no best-repeat selection.",
                "Hold32 is a shadow observer: a recorded intention is not an applied admission action.",
                "Observed schedule/token/submission prefixes do not establish equal KV tensors or an exact snapshot.",
                "Untriggered and nonaligned arms retain request measurements but cannot qualify a matched one-shot action.",
                "A binding opportunity is not an exact counterfactual divergence; target reach is not request benefit.",
                "No dynamic Oracle, expert-signal increment, production result or method GO is inferred."])


def report(result):
    lines = [f"# One-shot admission qualification: {result['status']}", "",
             f"Completed {result['completed_episodes']}/12; qualified {result['qualified_episodes']}.", "",
             "| Cell | Arm | Status | Trigger | Applied actions | Goodput |",
             "|---|---|---|---|---|---|"]
    for c in result["cells"]:
        lines.append(f"| {c['id']} | {c['arm']} | {c['status']} | {c.get('single_action', {}).get('status', 'UNAVAILABLE')} | "
                     f"{c.get('causal_and_action', {}).get('applied_action_count', '—')} | {c.get('goodput_rps', '—')} |")
    for pair in result["comparisons"]:
        lines += ["", f"{pair['group']} / {pair['regime']}: {pair.get('action_qualification', pair['status'])}."]
    return "\n".join(lines + ["", *result["limits"], "", *result["issues"]]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(report(result))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""F5 offline EWMA tracking and F6 measured CPU/truncation evidence.

  python code/analyze_acceptance_tracking.py runs/olmoe_ngram_matrix_r02 \
    --output figures/F5F6_tracking_olmoe_ngram_r02 --plot

Reads full performance populations only. Replays AcceptanceEWMA with a declared
cold prior at each run, then continues across its waves in recorded order. The
predictions are computed BEFORE each step; the same step is then used to update
the estimator. No warmup state is invented. This is an offline one-step tracking
diagnostic on an executed static trajectory, not an executed adaptive policy or
a counterfactual policy-performance estimate. Logged native alpha_after is a
cumulative accepted/tested-position ratio, not an EWMA snapshot.

Conditional acceptance uses sum(accepted)/sum(min(accepted+1,verified_drafts));
later slots after a rejection are untested. Availability is P(full proposal
length >= rank) over ALL eligible scheduled decode request-steps, including
zero proposals. Terminal requests are censored using plugin flags OR the exact
last client-chunk timestamp, including old zero-draft terminal omissions.

Missing per-request verified lengths are recovered from scheduled lengths only
when native aggregate verified == scheduled on a pure-decode step (zero invalid
slots). Otherwise those request observations remain unavailable. Full-budget
availability exposure is used only for fixed native budgets or explicit max
proposal mode. No unobserved higher-rank exposure is created.

F6 CPU timers are native/plugin host work; host draft calls are NOT GPU draft
kernel time. No adaptive selection timer => pending, never a fabricated zero.
Run-level errors and raw repeats are retained. Request-steps are not independent
system repetitions. Optional plots use the installed PGFPlots/Poppler locally.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import unittest

from cost_model import AcceptanceEWMA
from plot_results import REGIME_FIELDS, _context, _finite, resolve_regime, route_instrumented, write_csv as _write_csv
from summarize_draft_progress import _decode_ids, _external_id


CPU_FIELDS = {"control_ns": "wrapper_cpu_us", "acceptance_update_ns": "acceptance_update_cpu_us",
              "schedule_instrumentation_ns": "schedule_instrumentation_cpu_us"}
ERRORS = ("conditional_alpha", "availability_rank1", "progress_tokens")


def write_csv(path, rows):
    _write_csv(path, [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                       for k, v in row.items()} for row in rows])


def mean_or_none(values):
    return statistics.mean(values) if values else None


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def iter_steps(record):
    """Yield request-level observations using the established terminal boundary."""
    for wave_index, episode in enumerate(record.get("episodes", [record])):
        hosts, steps = episode.get("host_steps", []), episode.get("plugin", {}).get("steps", [])
        if not hosts or len(hosts) != len(steps):
            raise ValueError("missing/misaligned host and plugin step arrays")
        requests = {r["request_id"]: r for r in episode.get("requests", [])}
        external_ids = set(requests)
        finals = {rid: float(r["chunks"][-1]["elapsed_s"]) for rid, r in requests.items()
                  if r.get("finished") and r.get("chunks")}
        chunks = {rid: {float(c["elapsed_s"]): int(c["tokens"]) for c in r.get("chunks", [])}
                  for rid, r in requests.items()}
        id_cache = {}
        config = dict(episode.get("plugin", {}).get("config", {}))
        config["_acceptance_snapshot"] = episode.get("plugin", {}).get("acceptance_snapshot")
        for index, (host, step) in enumerate(zip(hosts, steps)):
            if int(host.get("index", -1)) != index or int(step.get("step", -1)) != index:
                raise ValueError("host/plugin step index mismatch")
            if step.get("episode") not in (None, episode.get("label")):
                raise ValueError("plugin step belongs to another episode")
            ids, identity_source = _decode_ids(step)
            if ids is not None and len(ids) != int(step.get("decode_batch", len(ids))):
                raise ValueError("decode request identity count differs from native batch")
            elapsed = _finite(host.get("elapsed_s"))
            censored = set(step.get("ewma_censored_requests", []))
            available = step.get("proposed_before_by_request")
            scheduled = step.get("scheduled_drafts_by_request")
            accepted = step.get("accepted_by_request")
            verified = step.get("observed_drafts_by_request")
            native_s = _finite(step.get("scheduled_drafts"))
            native_v = _finite(step.get("observed_drafts"))
            verified_from_zero_invalid = (step.get("phase") == "decode" and native_s is not None
                and native_v is not None and native_s == native_v)
            observations = []
            for rid in ids or []:
                if rid not in id_cache:
                    id_cache[rid] = _external_id(rid, external_ids)
                external = id_cache[rid]
                runner_known = external is not None and elapsed is not None
                terminal_runner = runner_known and finals.get(external) == elapsed
                terminal_plugin = rid in censored
                p = int(scheduled.get(rid, 0)) if scheduled is not None else None
                a = int(accepted[rid]) if accepted is not None and rid in accepted else (0 if p == 0 else None)
                full = int(available[rid]) if available is not None and rid in available else None
                if verified is not None and rid in verified:
                    v, v_source = int(verified[rid]), "native_per_request"
                elif p == 0:
                    v, v_source = 0, "zero_scheduled"
                elif verified_from_zero_invalid:
                    v, v_source = p, "scheduled_exact_from_zero_native_invalid_aggregate"
                else:
                    v, v_source = None, "unavailable"
                if a is not None and (a < 0 or v is not None and a > v or p is not None and a > p):
                    raise ValueError("invalid native accepted/verified/scheduled counts")
                observations.append({"id": rid, "available": full, "scheduled": p, "accepted": a,
                    "verified": v, "verified_source": v_source,
                    "emitted": chunks[external].get(elapsed, 0) if runner_known else None,
                    "terminal": bool(terminal_runner or terminal_plugin),
                    "terminal_known": runner_known or terminal_plugin,
                    "terminal_recovered": bool(terminal_runner and not terminal_plugin)})
            yield wave_index, episode.get("label"), host, step, config, observations, identity_source


def analyze_run(record, source, context=None, alpha=.7, decay=.9,
                alpha_estimator="exposure_ratio", prior_tested_count=1.0):
    if route_instrumented(record, context):
        raise ValueError("route-instrumented runs are not performance populations")
    if record.get("all_finished") is False:
        raise ValueError("unfinished run; analyze separately")
    mode = "native_batch_table" if (context or {}).get("native_batch_table") else record.get("mode", "native")
    gamma, budget = int(record["gamma"]), int(record.get("proposal_budget", record["gamma"]))
    identity = {**resolve_regime(record, context or {}), "source": source, "run_id": record.get("label"),
                "batch": int(record["batch"]), "gamma": gamma, "proposal_budget": budget, "mode": mode,
                "replay_alpha_estimator": alpha_estimator, "replay_prior_tested_count": prior_tested_count}
    estimator = AcceptanceEWMA(alpha=alpha, decay=decay, alpha_estimator=alpha_estimator,
                               prior_tested_count=prior_tested_count)
    tracking, cpu_by_phase = [], defaultdict(list)
    inventory, diagnostics = Counter(), Counter()
    rank_counts = {j: Counter() for j in range(1, budget + 1)}
    errors = {name: Counter() for name in ERRORS}
    totals = Counter()
    decode_index = 0
    for wave_index, wave, host, step, config, obs, identity_source in iter_steps(record):
        inventory.update(step.keys())
        inventory.update("decision." + key for key in (step.get("decision") or {}))
        totals["host_steps"] += 1
        if int(step.get("scheduled_tokens", 0)) > 0:
            cpu = {"step": int(step["step"]), "host_step_ms": _finite(host.get("host_step_ms"))}
            for field, name in CPU_FIELDS.items():
                val = _finite(step.get(field))
                cpu[name] = val / 1000 if val is not None else None
            cpu["policy_choose_cpu_us"] = _finite((step.get("decision") or {}).get("decision_cpu_us"))
            for field in ("drafts_available_before", "truncated_by_controller", "scheduled_drafts"):
                cpu[field] = _finite(step.get(field))
            cpu_by_phase[step.get("phase", "unknown")].append(cpu)
        if not obs:
            if int(step.get("decode_batch", 0)):
                diagnostics["decode_identity_unavailable_steps"] += 1
            continue
        totals["decode_steps"] += 1
        totals["decode_request_steps"] += len(obs)
        diagnostics["identity_" + identity_source] += 1
        eligible = []
        for o in obs:
            diagnostics["verified_" + o["verified_source"]] += 1
            if o["terminal"]:
                totals["terminal_request_steps"] += 1
                totals["terminal_recovered_from_runner"] += o["terminal_recovered"]
            elif not o["terminal_known"]:
                totals["terminal_status_unknown_request_steps"] += 1
            else:
                eligible.append(o)
        if not eligible:
            decode_index += 1
            continue
        totals["uncensored_request_steps"] += len(eligible)
        next_budget = int(step.get("proposal_budget_next", budget))
        full_budget_known = ((mode in ("native", "static") and gamma == budget and next_budget == budget)
                             or (config.get("proposal_budget") == "max" and mode != "native_batch_table"))
        available_obs = [o for o in eligible if o["available"] is not None] if full_budget_known else []
        valid = [o for o in eligible if o["accepted"] is not None and o["verified"] is not None]
        diagnostics["unavailable_acceptance_request_steps"] += len(eligible) - len(valid)
        diagnostics["unavailable_full_budget_request_steps"] += len(eligible) - len(available_obs)
        totals["availability_request_steps"] += len(available_obs)
        totals["acceptance_request_steps"] += len(valid)
        accepted = sum(o["accepted"] for o in valid)
        tested = sum(min(o["accepted"] + 1, o["verified"]) for o in valid)
        totals["accepted_drafts"] += accepted
        totals["tested_positions"] += tested
        conditional = accepted / tested if tested else None
        q_obs = sum(o["available"] >= 1 for o in available_obs) / len(available_obs) if available_obs and budget else None
        complete_progress = len(valid) == len(eligible) and all(o["emitted"] is not None for o in eligible)
        actual_progress = statistics.mean(o["emitted"] for o in eligible) if complete_progress else None
        proxy_progress = 1 + accepted / len(valid) if valid else None
        if complete_progress:
            diagnostics["nonterminal_proxy_emission_mismatches"] += sum(o["emitted"] != 1 + o["accepted"] for o in eligible)
        actual_gamma = int((step.get("decision") or {}).get("gamma", gamma))
        if actual_gamma < 0 or actual_gamma > budget:
            raise ValueError("selected gamma outside declared proposal budget")
        prediction = estimator.expected_tokens(actual_gamma)
        q_before = estimator.availability_survival(budget)
        s_before = estimator.prefix_survival(budget)
        logged_snapshot = config.get("_acceptance_snapshot") or {}
        if logged_snapshot.get("alpha_estimator"):
            native_alpha_semantics = "logged_" + logged_snapshot["alpha_estimator"]
        elif config.get("acceptance_class") or config.get("policy_class"):
            native_alpha_semantics = "configured_estimator_scalar_type_or_prior_not_recorded"
        else:
            native_alpha_semantics = "native_fallback_cumulative_conditional_prior_state_not_reset"
        row = {"source": source, "wave": wave, "wave_index": wave_index, "step": int(step["step"]),
            "decode_index": decode_index, "actual_decode_batch": len(obs), "eligible_requests": len(eligible),
            "selected_gamma": actual_gamma, "proposal_budget": budget,
            "conditional_alpha_before": estimator.alpha, "conditional_alpha_observed": conditional,
            "alpha_estimator": alpha_estimator, "prior_tested_count": prior_tested_count,
            "alpha_count_accepted_before": estimator.ewma_accepted,
            "alpha_count_tested_before": estimator.ewma_tested,
            "conditional_accepted": accepted, "conditional_tested": tested,
            "availability_rank1_before": q_before[0] if budget else None,
            "availability_rank1_observed": q_obs, "availability_requests": len(available_obs),
            "availability_positive": sum(o["available"] > 0 for o in available_obs),
            "progress_tokens_before": prediction, "progress_tokens_observed": actual_progress,
            "progress_requests": len(eligible) if complete_progress else 0,
            "one_plus_accepted_proxy": proxy_progress,
            "unseen_acceptance_ranks_before": sum(estimator.exposures[j] == 0 for j in range(1, budget + 1)),
            "unseen_availability_ranks_before": sum(estimator.availability_exposures[j] == 0 for j in range(1, budget + 1)),
            "native_logged_alpha_after": _finite(step.get("alpha_after")),
            "native_alpha_semantics": native_alpha_semantics,
            "full_budget_known": full_budget_known, "actual_online_snapshot_recorded": "acceptance_snapshot" in step,
            "evidence_kind": "offline_replayed_pre_step_estimate"}
        for j in range(1, budget + 1):
            rc = rank_counts[j]
            rc["availability_requests"] += len(available_obs)
            rc["available_at_rank"] += sum(o["available"] >= j for o in available_obs)
            exposed = [o for o in valid if o["verified"] >= j]
            rc["verified_at_rank"] += len(exposed)
            rc["accepted_at_rank"] += sum(o["accepted"] >= j for o in exposed)
            rc["acceptance_requests"] += len(valid)
            for name, obs_value, weight, pred in (
                ("availability", sum(o["available"] >= j for o in available_obs) / len(available_obs) if available_obs else None,
                 len(available_obs), q_before[j-1]),
                ("conditional_survival", sum(o["accepted"] >= j for o in exposed) / len(exposed) if exposed else None,
                 len(exposed), s_before[j-1])):
                if obs_value is not None:
                    rc[name + "_error_weight"] += weight
                    rc[name + "_absolute_error_sum"] += weight * abs(pred - obs_value)
                    rc[name + "_squared_error_sum"] += weight * (pred - obs_value)**2
        for name, weight in (("conditional_alpha", tested), ("availability_rank1", len(available_obs)),
                             ("progress_tokens", row["progress_requests"])):
            predicted, observed = row[name + "_before"], row[name + "_observed"]
            if predicted is not None and observed is not None and weight:
                delta = predicted - observed
                errors[name]["weight"] += weight
                errors[name]["absolute"] += weight * abs(delta)
                errors[name]["squared"] += weight * delta * delta
                errors[name]["signed"] += weight * delta
        estimator.update([o["accepted"] for o in valid], [o["verified"] for o in valid])
        if available_obs:
            estimator.update_availability([o["available"] for o in available_obs], budget)
        row["conditional_alpha_after_replay"] = estimator.alpha
        row["availability_rank1_after_replay"] = estimator.availability_survival(budget)[0] if budget else None
        tracking.append(row)
        decode_index += 1
    run = {**identity, **dict(totals), "prior_alpha": alpha, "decay": decay,
        "reset_boundary": "cold prior once per run; continuous recorded wave order; warmup state not reconstructed",
        "conditional_alpha_pooled": totals["accepted_drafts"] / totals["tested_positions"] if totals["tested_positions"] else None,
        "diagnostics": dict(diagnostics), "field_occurrences": dict(inventory),
        "final_replayed_snapshot": estimator.snapshot(),
        "evidence_kind": "offline_tracking_on_actual_static_trajectory"}
    for name, err in errors.items():
        for suffix, numerator in (("mae", "absolute"), ("mse", "squared"), ("bias", "signed")):
            run[name + "_prestep_" + suffix] = err[numerator] / err["weight"] if err["weight"] else None
        run[name + "_prestep_rmse"] = math.sqrt(run[name + "_prestep_mse"]) if err["weight"] else None
        run[name + "_error_weight"] = err["weight"]
    ranks = []
    for j, c in rank_counts.items():
        row = {**identity, "rank": j, **dict(c),
            "availability_survival": c["available_at_rank"] / c["availability_requests"] if c["availability_requests"] else None,
            "conditional_acceptance_survival": c["accepted_at_rank"] / c["verified_at_rank"] if c["verified_at_rank"] else None,
            "unconditional_accepted_survival": c["accepted_at_rank"] / c["acceptance_requests"] if c["acceptance_requests"] else None}
        for name in ("availability", "conditional_survival"):
            weight = c[name + "_error_weight"]
            row[name + "_prestep_mae"] = c[name + "_absolute_error_sum"] / weight if weight else None
            row[name + "_prestep_rmse"] = math.sqrt(c[name + "_squared_error_sum"] / weight) if weight else None
        ranks.append(row)
    controls = []
    for phase, values in cpu_by_phase.items():
        row = {**identity, "phase": phase, "steps": len(values)}
        row["host_step_ms_mean"] = mean_or_none([v["host_step_ms"] for v in values if v["host_step_ms"] is not None])
        for metric in [*CPU_FIELDS.values(), "policy_choose_cpu_us"]:
            observed = [v[metric] for v in values if v[metric] is not None]
            row[metric + "_observed_steps"] = len(observed)
            row[metric + "_mean"] = mean_or_none(observed)
            row[metric + "_p50"] = quantile(observed, .5)
            row[metric + "_p95"] = quantile(observed, .95)
            row[metric + "_max"] = max(observed) if observed else None
            matched = [v for v in values if v[metric] is not None and v["host_step_ms"] is not None]
            denominator = sum(v["host_step_ms"] for v in matched) * 1000
            row[metric + "_total_over_host_fraction"] = sum(v[metric] for v in matched) / denominator if denominator else None
        for field in ("drafts_available_before", "truncated_by_controller", "scheduled_drafts"):
            observed = [v[field] for v in values if v[field] is not None]
            row[field] = sum(observed) if len(observed) == len(values) else None
            row[field + "_observed_steps"] = len(observed)
        a, t = row["drafts_available_before"], row["truncated_by_controller"]
        row["controller_discarded_available_fraction"] = t / a if a and t is not None else None
        row["adaptive_choose_status"] = "measured" if mode == "adaptive" and row["policy_choose_cpu_us_observed_steps"] else "pending_no_adaptive_choice_measurement"
        row["draft_gpu_waste_status"] = "unmeasured_host_call_timers_are_not_gpu_kernel_time"
        row["timing_overlap_boundary"] = "wrapper_cpu_us is included in schedule_instrumentation_cpu_us; do not add them"
        controls.append(row)
    return run, tracking, ranks, controls


def repeat_summary(rows, metrics, extra_fields=()):
    fields = (*REGIME_FIELDS, "batch", "gamma", "proposal_budget", "mode", "replay_alpha_estimator", "replay_prior_tested_count", *extra_fields)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        row = {**dict(zip(fields, key)), "n_runs": len(values), "sources": [v["source"] for v in values]}
        for metric in metrics:
            raw = [v.get(metric) for v in values]
            row[metric + "_raw"] = raw
            row[metric + "_median"] = statistics.median(raw) if raw and all(v is not None for v in raw) else None
        output.append(row)
    return output


def bin_tracking(rows, bins):
    width = max(1, math.ceil(len(rows) / bins))
    output = []
    for start in range(0, len(rows), width):
        part = rows[start:start+width]
        item = {"decode_index": statistics.mean(r["decode_index"] for r in part), "steps": len(part)}
        for name, weight_key in (("conditional_alpha", "conditional_tested"),
                                  ("availability_rank1", "availability_requests"), ("progress_tokens", "progress_requests")):
            valid = [r for r in part if r[name + "_observed"] is not None and r[name + "_before"] is not None and r[weight_key]]
            weight = sum(r[weight_key] for r in valid)
            for suffix in ("before", "observed"):
                item[name + "_" + suffix] = sum(r[name + "_" + suffix]*r[weight_key] for r in valid) / weight if weight else None
            item[name + "_weight"] = weight
        output.append(item)
    return output


def render_tracking(binned_runs, controls, output, plot_gamma):
    from plot_energy_progress import pgf_document, render_pgf
    files = []
    groups = defaultdict(list)
    for run, bins in binned_runs:
        key = tuple(run[k] for k in (*REGIME_FIELDS, "batch", "gamma", "proposal_budget", "mode"))
        groups[key].append((run, bins))
    for key, values in sorted(groups.items()):
        panels = []
        max_x = max((p["decode_index"] for _, bins in values for p in bins), default=1)
        for name, title, ylabel in (("conditional_alpha", "(a) Conditional acceptance", "Accepted / tested positions"),
                                   ("availability_rank1", "(b) Proposal availability", r"$P(P \geq 1)$"),
                                   ("progress_tokens", "(c) Decode progress", "Emitted tokens / request-step")):
            panel = {"xs": [0, round(max_x/2), math.ceil(max_x)], "xlabel": "Decode step (waves concatenated)",
                     "ylabel": ylabel, "title": title, "series": []}
            for index, (_, bins) in enumerate(values):
                for suffix, label in (("before", "forecast"), ("observed", "observed")):
                    panel["series"].append({"label": f"run {index}: {label}",
                        "medians": [(p["decode_index"], p[name + "_" + suffix]) for p in bins], "raw": []})
            panels.append(panel)
        batch = key[-4]
        tag = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:8]
        note = r"Offline EWMA replay; predictions precede each step, cold prior once per run, terminal requests censored.\\" \
               r"Display bins aggregate recorded observations and forecasts; error CSVs use every step. This is not an executed adaptive policy."
        variant = values[0][0]["replay_alpha_estimator"].replace("_", " ")
        files += render_pgf(pgf_document(panels, 2*len(values), note, f"Offline {variant}: B={batch}, gamma={plot_gamma}"),
                            output, f"F5_offline_tracking_b{batch}_g{plot_gamma}_{tag}", ["pdf", "svg", "png"])
    native = [r for r in controls if r["phase"] == "decode" and r["mode"] == "native"]
    groups = defaultdict(list)
    for row in native:
        groups[tuple(row[k] for k in REGIME_FIELDS)].append(row)
    for key, rows in sorted(groups.items()):
        batches = sorted({r["batch"] for r in rows})
        gammas = sorted({r["gamma"] for r in rows})
        panels = []
        for index, (metric, title) in enumerate((("wrapper_cpu_us_mean", "(a) Native wrapper"),
                ("acceptance_update_cpu_us_mean", "(b) Native acceptance logging/update"),
                ("schedule_instrumentation_cpu_us_mean", "(c) Schedule instrumentation"))):
            panel = {"xs": batches, "log_x": True, "xlabel": "Nominal batch size $B$", "ylabel": r"CPU time ($\mu$s)",
                     "title": title, "series": []}
            for gamma in gammas:
                raw = [(b, [r[metric] for r in rows if r["batch"] == b and r["gamma"] == gamma]) for b in batches]
                panel["series"].append({"label": r"$\gamma=%d$" % gamma,
                    "medians": [(b, statistics.median(v) if v and all(x is not None for x in v) else None) for b,v in raw], "raw": raw})
            panels.append(panel)
        tag = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:8]
        note = r"Native instrumentation: run medians of per-run step means, open dots are individual runs. Panel (a) is INCLUDED in (c).\\" \
               r"Adaptive selection CPU cost is pending. Host-call timers and discarded-token counts do not measure wasted GPU draft time."
        files += render_pgf(pgf_document(panels, len(gammas), note, "F6: measured native CPU instrumentation (adaptive pending)"),
                            output, f"F6_native_cpu_{tag}", ["pdf", "svg", "png"])
    return files


def self_test():
    def fixture():
        return {"label": "r", "batch": 2, "gamma": 2, "mode": "native", "episodes": [{"label": "w",
            "requests": [{"request_id": "a", "finished": False, "chunks": [{"elapsed_s": 1., "tokens": 1}]},
                         {"request_id": "b", "finished": True, "chunks": [{"elapsed_s": 1., "tokens": 1}]}],
            "host_steps": [{"index": 0, "elapsed_s": 1.}], "plugin": {"steps": [{"episode": "w", "step": 0,
            "phase": "decode", "decode_batch": 2, "scheduled_tokens": 2,
            "scheduled_tokens_by_request": {"a-x": 1, "b-x": 1}, "proposed_before_by_request": {"a-x": 0, "b-x": 0},
            "scheduled_drafts_by_request": {}, "accepted_by_request": {}, "scheduled_drafts": 0, "observed_drafts": 0,
            "alpha_after": .234, "control_ns": 1234, "decision": {"gamma": 2, "mode": "native"}}]}}]}
    class Tests(unittest.TestCase):
        def test_zero_proposals_and_terminal_censor(self):
            run, steps, ranks, cpu = analyze_run(fixture(), "synthetic")
            self.assertEqual(run["uncensored_request_steps"], 1)
            self.assertEqual(run["terminal_recovered_from_runner"], 1)
            self.assertEqual(steps[0]["availability_rank1_observed"], 0)
            self.assertEqual(steps[0]["availability_rank1_after_replay"], 0)
            self.assertEqual(steps[0]["conditional_alpha_before"], .7)
            self.assertIsNone(steps[0]["conditional_alpha_observed"])
            self.assertEqual(steps[0]["native_logged_alpha_after"], .234)
            self.assertIsNone(cpu[0]["policy_choose_cpu_us_mean"])
            self.assertEqual(cpu[0]["wrapper_cpu_us_mean"], 1.234)
        def test_invalid_aggregate_not_assigned_to_request(self):
            obj=fixture(); s=obj["episodes"][0]["plugin"]["steps"][0]
            s.update(scheduled_drafts=2, observed_drafts=1, scheduled_drafts_by_request={"a-x":2}, accepted_by_request={"a-x":1})
            _, steps, _, _ = analyze_run(obj, "synthetic")
            self.assertEqual(steps[0]["conditional_tested"], 0)
        def test_pre_step_error_retains_initial_prior(self):
            _, steps, _, _ = analyze_run(fixture(), "synthetic")
            self.assertAlmostEqual(steps[0]["progress_tokens_before"], 1+.7+.49)
            self.assertEqual(steps[0]["progress_tokens_observed"], 1)
        def test_missing_cpu_not_zero(self):
            obj=fixture(); del obj["episodes"][0]["plugin"]["steps"][0]["control_ns"]
            _, _, _, cpu=analyze_run(obj,"synthetic")
            self.assertIsNone(cpu[0]["wrapper_cpu_us_mean"])
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs",nargs="*")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--alpha",type=float,default=.7)
    parser.add_argument("--decay",type=float,default=.9)
    parser.add_argument("--alpha-estimator",choices=("exposure_ratio","legacy_step_ratio"),default="exposure_ratio")
    parser.add_argument("--prior-tested-count",type=float,default=1.0)
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--plot-gamma",type=int,default=7)
    parser.add_argument("--plot-batches",type=int,nargs="+",default=[1,16,64])
    parser.add_argument("--plot-bins",type=int,default=40)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.inputs or args.output is None:
        parser.error("inputs and --output are required")
    prior_manifest = args.output / "evidence_manifest.json"
    if prior_manifest.exists():
        previous = json.loads(prior_manifest.read_text()).get("replay_prior", {})
        if previous.get("alpha_estimator", "legacy_step_ratio") != args.alpha_estimator:
            parser.error("output contains another estimator's replay; choose a separate --output directory")
    paths=set()
    for path in map(Path,args.inputs):
        paths.update(p.resolve() for p in path.rglob("perf-*.json")) if path.is_dir() else paths.add(path.resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    runs,ranks,controls,excluded,binned,sources=[],[],[],[],[],[]
    with gzip.open(args.output/"tracking_steps.jsonl.gz","wt",encoding="utf-8") as stream:
        for path in sorted(paths):
            try:
                raw=path.read_bytes();obj=json.loads(raw)
                if str(obj.get("label","")).startswith("warmup"):
                    raise ValueError("warmup is not a measured performance run")
                run,steps,rank,cpu=analyze_run(obj,str(path),_context(path),args.alpha,args.decay,
                                             args.alpha_estimator,args.prior_tested_count)
                run["source_sha256"]=hashlib.sha256(raw).hexdigest()
                sources.append({"source":str(path),"sha256":run["source_sha256"]})
                runs.append(run);ranks.extend(rank);controls.extend(cpu)
                for step in steps:
                    stream.write(json.dumps(step,separators=(",",":"))+"\n")
                if run["gamma"]==args.plot_gamma and run["batch"] in args.plot_batches:
                    binned.append((run,bin_tracking(steps,args.plot_bins)))
            except (OSError,ValueError,KeyError,TypeError) as exc:
                excluded.append({"source":str(path),"reason":str(exc)})
    error_metrics=[name+"_prestep_"+suffix for name in ERRORS for suffix in ("mae","rmse","bias")]
    summaries=repeat_summary(runs,error_metrics)
    control_metrics=[m+"_mean" for m in [*CPU_FIELDS.values(),"policy_choose_cpu_us"]]+["controller_discarded_available_fraction"]
    control_metrics += [m+"_total_over_host_fraction" for m in [*CPU_FIELDS.values(),"policy_choose_cpu_us"]]
    control_summary=repeat_summary(controls,control_metrics,("phase",))
    write_csv(args.output/"tracking_runs.csv",runs)
    write_csv(args.output/"tracking_summary.csv",summaries)
    write_csv(args.output/"rank_coverage_runs.csv",ranks)
    write_csv(args.output/"control_runs.csv",controls)
    write_csv(args.output/"control_summary.csv",control_summary)
    files=render_tracking(binned,controls,args.output,args.plot_gamma) if args.plot else []
    field_occurrences=Counter()
    missing_diagnostics=Counter()
    for run in runs:
        field_occurrences.update(run["field_occurrences"])
        missing_diagnostics.update(run["diagnostics"])
    manifest={"sources":sources,"analyzed_runs":len(runs),"excluded":excluded,
        "replay_prior":{"alpha":args.alpha,"decay":args.decay,"alpha_estimator":args.alpha_estimator,"prior_tested_count":args.prior_tested_count},
        "field_occurrences":dict(field_occurrences),"observation_diagnostics":dict(missing_diagnostics),
        "analysis_code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name("cost_model.py"))},
        "reset_boundary":"once per run, continuous waves; unknown warmup estimator state not invented",
        "tracking_boundary":"offline pre-step predictions on recorded trajectories; not executed adaptive policy",
        "truth_boundary":"uncensored scheduled decode requests incl zero proposals and drain batches; current observed rates are noisy observations, not latent true probabilities",
        "conditional_alpha_boundary":"accepted / tested positions; separate from proposal availability and accepted/proposed slot ratio",
        "error_boundary":"run-level weighted one-step errors; weights are tested positions/available request-steps/progress request-steps respectively; no CI from steps",
        "native_logged_alpha_boundary":"r02 native alpha_after was fallback cumulative conditional ratio; later logs use acceptance_snapshot/class to identify scalar estimator; no pre-run state is invented",
        "prefix_weighting_boundary":"prefix_survival and availability retain equal eligible-step EWMA weighting; rank exposure counts are recorded but not used as across-step weights",
        "adaptive_choose_status":"measured" if any(r["adaptive_choose_status"]=="measured" for r in controls) else "pending_no_adaptive_choice_measurement",
        "gpu_draft_waste_status":"not measured; host draft calls are not GPU kernel duration",
        "cpu_timing_boundary":"all scheduled steps by phase, including terminal/drain; control_ns is nested inside schedule_instrumentation_ns and must not be added to it; no estimated no-instrumentation latency is produced",
        "outputs":files,"display_gamma":args.plot_gamma,"display_batches":args.plot_batches,"display_bins":args.plot_bins}
    (args.output/"evidence_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(args.output),"input_runs":len(paths),"analyzed_runs":len(runs),"excluded_runs":len(excluded),"figures":len(files),"adaptive_choose_status":manifest["adaptive_choose_status"]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

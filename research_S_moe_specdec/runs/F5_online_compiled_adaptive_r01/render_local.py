#!/usr/bin/env python3
"""Render the completed actual-online adaptive F5 analysis, without replay/fit.

From project root: python3 -B runs/F5_online_compiled_adaptive_r01/render_local.py
Reuses the existing PGF helpers. Fixed trajectory: repeat0, B64, including drain.
"""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from plot_kernel_diagnostics import coords, document
from plot_energy_progress import render_pgf
from plot_results import REGIME_FIELDS

SOURCE = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures/F5_online_compiled_adaptive_r01"
BATCHES = [1, 4, 16, 64]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = json.loads((SOURCE / "online_acceptance.json").read_text())
    assert manifest["analysis_status"] == "complete" and not manifest["campaign_partial"]
    assert not manifest["missing_inputs"] and not manifest["invalid_inputs"]
    runs, engine = manifest["runs"], manifest["engines"][0]
    assert len(runs) == 8 and len(manifest["engines"]) == 1 and engine["engine_complete"]
    assert {(r["batch"], r["repeat"]) for r in runs} == {(b, i) for b in BATCHES for i in (0, 1)}
    assert all(r["status"] == "complete" and r["mode"] == "adaptive" and r["execution_mode"] == "compiled" for r in runs)
    assert all(r["online_state_recovered_steps"] == r["host_steps"] for r in runs)
    assert len({tuple(r[k] for k in REGIME_FIELDS) for r in runs}) == 1
    ranks = {(r["source"], r["rank"]): r for r in manifest["rank_coverage"]}
    values, raw_waves, endpoints = [], set(), 0
    raw_counts = Counter()
    for run in sorted(runs, key=lambda r: (r["batch"], r["repeat"])):
        content = Path(run["source"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == run["source_sha256"]
        record = json.loads(content)
        for ep in record["episodes"]:
            steps, snapshot = ep["plugin"]["steps"], ep["plugin"]["acceptance_snapshot"]
            assert len(steps) == len(ep["host_steps"])
            assert math.isclose(steps[-1]["alpha_after"], snapshot["alpha"], abs_tol=1e-10)
            endpoints += 1
            raw_waves.add((run["source"], ep["label"]))
            raw_counts["host_steps"] += len(steps)
            raw_counts["availability_updates"] += sum(bool(s["availability_updated"]) for s in steps)
            raw_counts["logged_decision_predictions"] += sum((s.get("decision") or {}).get("expected_tokens_per_request") is not None for s in steps)
        rank = ranks[run["source"], 1]
        values.append({"batch": run["batch"], "repeat": run["repeat"],
            "source": run["source"], "source_sha256": run["source_sha256"],
            "alpha_mae": run["conditional_alpha_prestep_mae"],
            "alpha_rmse": run["conditional_alpha_prestep_rmse"],
            "alpha_bias": run["conditional_alpha_prestep_bias"],
            "alpha_tested_weight": run["conditional_alpha_prestep_weight"],
            "decision_progress_mae": run["logged_decision_progress_prestep_mae"],
            "decision_progress_rmse": run["logged_decision_progress_prestep_rmse"],
            "decision_progress_bias": run["logged_decision_progress_prestep_bias"],
            "decision_progress_request_weight": run["logged_decision_progress_prestep_weight"],
            "prefix_rank1_mae": rank["conditional_prefix_prestep_mae"],
            "availability_rank1_mae": rank["availability_prestep_mae"],
            "prefix_rank1_observed": rank["conditional_prefix_survival_observed"],
            "availability_rank1_observed": rank["availability_survival_observed"],
            "prefix_rank1_exposure": rank["verified_exposure"],
            "availability_rank1_exposure": rank["availability_exposure"],
            "display_batch": run["batch"] * 2 ** (-.045 if run["repeat"] == 0 else .045)})

    tracking_counts, tracking_waves, trajectory = Counter(), set(), []
    with gzip.open(SOURCE / "tracking_steps.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["online_state_recovered"] and row["alpha_after_checkpoint"] is not None
            assert row["full_proposal_budget_identified"]
            tracking_waves.add((row["source"], row["wave"]))
            tracking_counts["alpha_after_checkpoints"] += 1
            tracking_counts["availability_updates"] += bool(row["availability_updated_by_runtime"])
            tracking_counts["logged_decision_predictions"] += row["logged_decision_progress_before"] is not None
            tracking_counts["progress_scored_steps"] += row["logged_decision_progress_error"] is not None
            if row["repeat"] == 0 and row["nominal_batch"] == 64 and row["phase"] == "decode":
                rank = row["ranks"][0]
                trajectory.append({k: row[k] for k in ("source", "wave", "step", "actual_decode_batch", "selected_gamma",
                    "conditional_alpha_before", "conditional_alpha_observed", "conditional_tested",
                    "availability_rank1_before", "availability_rank1_observed", "availability_request_steps",
                    "logged_decision_progress_before", "progress_observed", "progress_requests", "terminal_request_steps", "state_anchor")} | {
                    "prefix_rank1_before": row["prefix_survival_before"][0],
                    "prefix_rank1_observed": rank["accepted"] / rank["verified_exposure"] if rank["verified_exposure"] else None,
                    "prefix_rank1_exposure": rank["verified_exposure"]})
    assert tracking_waves == raw_waves
    assert tracking_counts["alpha_after_checkpoints"] == raw_counts["host_steps"]
    for key in ("availability_updates", "logged_decision_predictions"):
        assert tracking_counts[key] == raw_counts[key]
    assert len({r["source"] for r in trajectory}) == len({r["wave"] for r in trajectory}) == 1
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def run_lines(series):
        lines = []
        for field, color, label in series:
            median = [(b, statistics.median(r[field] for r in values if r["batch"] == b)) for b in BATCHES]
            raw = [(r["display_batch"], r[field]) for r in values]
            lines += [r"\addplot[color=%s,mark=*,mark size=1.5pt] coordinates {%s};\addlegendentry{%s}" % (color, coords(median), label),
                r"\addplot[forget plot,only marks,color=%s,mark=o,mark size=2pt,mark options={fill=white}] coordinates {%s};" % (color, coords(raw))]
        return "\n".join(lines)
    axis = r"xmode=log,log basis x=2,xtick={1,4,16,64},xticklabels={1,4,16,64},xlabel={Nominal wave batch $B$},xmin=.88,xmax=74,ymin=0,legend pos=north east"
    axes = [r"\nextgroupplot[title={(a) Pooled conditional $\alpha$},ylabel={Tested-position-weighted MAE},%s]" % axis + run_lines([("alpha_mae", "cal", "Pre-step pooled alpha")]),
        r"\nextgroupplot[title={(b) Rank-1 state errors},ylabel={Exposure-weighted MAE},%s]" % axis + run_lines([("prefix_rank1_mae", "cal", "Conditional prefix"), ("availability_rank1_mae", "profile", "Full-proposal availability")]),
        r"\nextgroupplot[title={(c) Actual decision progress},ylabel={Tokens per request-step},%s]" % axis + run_lines([("decision_progress_mae", "cal", "MAE"), ("decision_progress_rmse", "test", "RMSE")])]
    note = (r"Actual compiled OLMoE / n-gram adaptive run; held-out records 64--127. Eight original run values, with continuous warm EWMA state.\\"
            r"Open dots retain both repetitions; lines are two-value medians, not confidence intervals. No reset across batches or waves.\\"
            r"Errors compare pre-step predictions to noisy realized outcomes, not latent truth; batch changes observation noise. Terminal requests are censored.\\"
            r"Rank-1 prefix errors use verified exposure; availability includes zero full-budget proposals; decision progress is the prediction actually logged online.")
    tex = document(axes, "F5: actual online adaptive prediction errors", note).replace("width=8.0cm,height=6.0cm", "width=6.2cm,height=5.6cm")
    files = render_pgf(tex, OUTPUT, "F5_online_prediction_errors", ["pdf", "svg", "png"])

    panels = [("(a) Pooled conditional alpha", "conditional_alpha_before", "conditional_alpha_observed", "Conditional accepted / tested", 0, 1),
        ("(b) Rank-1 prefix survival", "prefix_rank1_before", "prefix_rank1_observed", "Conditional prefix probability", 0, 1),
        ("(c) Rank-1 full-proposal availability", "availability_rank1_before", "availability_rank1_observed", "Proposal-positive fraction", 0, .4),
        ("(d) Actual chosen-cap progress", "logged_decision_progress_before", "progress_observed", "Tokens per request-step", .98, 1.5)]
    axes = []
    for title, before, observed, ylabel, lo, hi in panels:
        axes.append(r"\nextgroupplot[title={%s},xlabel={Original host step},ylabel={%s},xmin=0,xmax=127,ymin=%s,ymax=%s,legend pos=north west]" % (title, ylabel, lo, hi) +
            r"\addplot[color=cal,line width=1pt] coordinates {%s};\addlegendentry{Actual pre-step state/prediction}" % coords([(r["step"], r[before]) for r in trajectory]) +
            r"\addplot[only marks,color=test,mark=o,mark size=1.4pt] coordinates {%s};\addlegendentry{Realized eligible outcome}" % coords([(r["step"], r[observed]) for r in trajectory]))
    note = (r"Fixed illustration: repeat 0, nominal $B=64$, includes drain to $B=1$; not an independent repeat or whole-run summary.\\"
            r"State continues from the preceding measured run; all pre-step values were checked against executed updates and recorded snapshots.\\"
            r"Pooled alpha and rank-1 prefix survival are different statistics. No-exposure/terminal outcomes remain missing, not zero.\\"
            r"Availability is learned from complete budget-7 proposals before truncation. Panel (d) uses the actual online decision's selected cap.")
    tex = document(axes, "F5: actual adaptive state and decision trajectory", note)
    tex = tex.replace("group size=4 by 1,horizontal sep=1.9cm", "group size=2 by 2,horizontal sep=2.4cm,vertical sep=2.0cm")
    tex = tex.replace("(group c4r1.north east)", "(group c2r1.north east)")
    tex = tex.replace("(group c1r1.south west)", "(group c1r2.south west)").replace("(group c4r1.south east)", "(group c2r2.south east)")
    files += render_pgf(tex, OUTPUT, "F5_online_state_example", ["pdf", "svg", "png"])
    write_csv("plot_run_values.csv", values)
    write_csv("plot_trajectory_values.csv", trajectory)
    rank_counts = [{"rank": j, **{k: sum(r[k] for r in manifest["rank_coverage"] if r["rank"] == j)
        for k in ("available", "availability_exposure", "verified_exposure", "accepted")}} for j in range(1, 8)]
    result = {"status": "actual_online_adaptive_diagnostic", "analysis_manifest": str(SOURCE / "online_acceptance.json"),
        "analysis_sha256": sha(SOURCE / "online_acceptance.json"), "script_sha256": sha(Path(__file__)),
        "tracking_sha256": sha(SOURCE / "tracking_steps.jsonl.gz"), "original_run_count": len(runs),
        "original_runs_sha_verified": len(runs), "wave_end_full_state_snapshots_matched_by_analyzer": endpoints,
        "tracking_counts": dict(tracking_counts), "initial_state_metadata": engine["initial_state_metadata"],
        "initial_alpha": engine["initial_acceptance_snapshot"]["alpha"],
        "initial_alpha_counts": engine["initial_acceptance_snapshot"]["alpha_counts"],
        "terminal_request_steps": sum(r["terminal_request_steps"] for r in runs),
        "unknown_terminal_request_steps": sum(r["terminal_status_unknown_request_steps"] for r in runs),
        "terminal_recovered_from_runner": sum(r["terminal_recovered_from_runner"] for r in runs),
        "proxy_emission_mismatches": sum(r["proxy_emission_mismatches"] for r in runs),
        "eligible_request_steps": sum(r["eligible_request_steps"] for r in runs), "rank_coverage_totals": rank_counts,
        "trajectory": {"repeat":0, "batch":64, "decode_steps":len(trajectory), "actual_batch_min":min(r["actual_decode_batch"] for r in trajectory)},
        "figures": files, "boundaries": manifest["boundaries"] + [
            "Rendering reuses completed analyzer outputs, verifies raw source hashes and checkpoint counts, and performs no new state replay, fit or prediction.",
            "Prefix observations are conditional on ranks actually exposed by the chosen caps; availability observes the full budget before truncation.",
            "Snapshot/decision agreement validates reconstruction, not latent prediction accuracy or end-to-end policy benefit.",
            "The illustration is fixed at repeat0/B64 and includes draining; it is not selected for low error."]}
    (OUTPUT / "plot_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "tracking_counts":dict(tracking_counts), "wave_end_snapshots":endpoints, "figures":files}))


if __name__ == "__main__":
    main()

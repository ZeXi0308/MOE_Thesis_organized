#!/usr/bin/env python3
"""Render the recorded compiled-gamma1 F5 diagnostic; no fitting or replay.

Run from the project root: python3 -B runs/F5_online_compiled_gamma01_r01/render_local.py
The input is the completed selected-engine analysis, not a completed campaign.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from plot_kernel_diagnostics import coords, document
from plot_energy_progress import render_pgf

SOURCE = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures/F5_online_compiled_gamma1_partial_r01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((SOURCE / "online_acceptance.json").read_text())
    assert manifest["analysis_status"] == "complete" and manifest["campaign_partial"] is True
    runs = [r for r in manifest["runs"] if r["gamma"] == 1]
    ranks = {(r["batch"], r["repeat"]): r for r in manifest["rank_coverage"] if r["gamma"] == r["rank"] == 1}
    assert len(runs) == 14 and all(r["status"] == "complete" and r["execution_mode"] == "compiled" and r["mode"] == "native" for r in runs)
    assert len({(r["model"], r["draft_source"], r["dtype"], r["quantization"], r["workload_sha256"]) for r in runs}) == 1
    assert all(r["recorded_decision_prediction_steps"] == 0 for r in runs)
    values = []
    for run in runs:
        rank = ranks[run["batch"], run["repeat"]]
        values.append({"batch": run["batch"], "repeat": run["repeat"], "source": run["source"],
            "alpha_prestep_mae": run["conditional_alpha_prestep_mae"],
            "alpha_prestep_rmse": run["conditional_alpha_prestep_rmse"],
            "alpha_prestep_bias": run["conditional_alpha_prestep_bias"],
            "alpha_tested_weight": run["conditional_alpha_prestep_weight"],
            "conditional_accepted_over_tested": run["conditional_alpha_pooled"],
            "proposal_positive_fraction": rank["availability_survival_observed"],
            "proposal_request_step_exposure": rank["availability_exposure"]})
    batches = sorted({r["batch"] for r in values})
    assert all(sorted(r["repeat"] for r in values if r["batch"] == b) == [0, 1] for b in batches)
    def lines(fields):
        result = []
        for field, color, label in fields:
            medians = [(b, statistics.median(r[field] for r in values if r["batch"] == b)) for b in batches]
            raw = [(r["batch"] * (0.987 if r["repeat"] == 0 else 1.013), r[field]) for r in values]
            result += [r"\addplot[color=%s,mark=*,mark size=1.5pt] coordinates {%s};" % (color, coords(medians)),
                       r"\addlegendentry{%s}" % label,
                       r"\addplot[forget plot,only marks,color=%s,mark=o,mark size=2.0pt,mark options={fill=white}] coordinates {%s};" % (color, coords(raw))]
        return "\n".join(result)
    axis = r"xmode=log,log basis x=2,xtick={1,2,4,8,16,32,64},xticklabels={1,2,4,8,16,32,64},xlabel={Nominal wave batch B},ymin=0,legend pos=north east"
    axes = [r"\nextgroupplot[title={(a) Actual pre-step scalar alpha error},ylabel={Exposure-weighted error},%s]" % axis
        + lines([("alpha_prestep_mae", "cal", "MAE"), ("alpha_prestep_rmse", "test", "RMSE")]),
        r"\nextgroupplot[title={(b) Conditional acceptance and proposal availability},ylabel={Observed fraction},%s,legend pos=south east]" % axis
        + lines([("conditional_accepted_over_tested", "cal", "Accepted / tested (conditional)"),
                 ("proposal_positive_fraction", "profile", "Proposal-positive request-steps")])]
    note = (r"Compiled OLMoE / n-gram, native static $\gamma=1$ only; surrounding matrix is partial.\\"
            r"Lines are medians of the two recorded run values (open dots), not confidence intervals; EWMA state is continuous across runs.\\"
            r"Errors compare pre-step state to noisy realized step rates, weighted by tested positions; batch changes observation noise. Terminal steps are censored.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = render_pgf(document(axes, "F5: actual online observer, compiled gamma1 (partial matrix)", note),
                       OUTPUT, "F5_online_errors_coverage", ["pdf", "svg", "png"])
    trajectory, checkpoints = [], {0: 0, 1: 0}
    with gzip.open(SOURCE / "tracking_steps.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["online_state_recovered"] and row.get("alpha_after_checkpoint") is not None
            assert row["availability_updated_by_runtime"] is False
            assert not any(row["availability_exposures_before"])
            checkpoints[row["proposal_budget"]] += 1
            if row["proposal_budget"] == 1 and row["repeat"] == 0 and row["nominal_batch"] == 64 and row["phase"] == "decode":
                trajectory.append({key: row[key] for key in ("source", "wave", "step", "actual_decode_batch",
                    "conditional_alpha_before", "conditional_alpha_observed", "conditional_tested", "state_anchor")})
    trace = [r"\nextgroupplot[title={Illustrative first run at B cap 64; includes draining},xlabel={Original host step},ylabel={Conditional acceptance},xmin=0,xmax=127,ymin=0,ymax=1,legend pos=south west]",
        r"\addplot[color=cal,line width=1pt] coordinates {%s};" % coords([(r["step"], r["conditional_alpha_before"]) for r in trajectory]),
        r"\addlegendentry{Actual EWMA state before step}",
        r"\addplot[only marks,color=test,mark=o,mark size=1.4pt] coordinates {%s};" % coords([(r["step"], r["conditional_alpha_observed"]) for r in trajectory]),
        r"\addlegendentry{Realized accepted / tested}"]
    note = (r"Native static $\gamma=1$ observer: no adaptive decision or progress prediction was logged.\\"
            r"State starts from the preceding run's recorded checkpoint; no reset or future observations. This one wave is illustrative.\\"
            r"Unexposed/terminal outcomes remain missing. Native availability EWMA was never updated, so its prior is not plotted as learned tracking.")
    tex = document(["\n".join(trace)], "F5: actual pre-step alpha trajectory (compiled gamma1 only)", note).replace("width=8.0cm", "width=15.5cm")
    files += render_pgf(tex, OUTPUT, "F5_online_alpha_example", ["pdf", "svg", "png"])
    for name, rows in (("plot_run_values.csv", values), ("plot_trajectory_values.csv", trajectory)):
        with (OUTPUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"status": "measured_selected_engine_diagnostic", "campaign_partial": True,
        "gamma": 1, "analysis_manifest": str(SOURCE / "online_acceptance.json"),
        "analysis_sha256": sha(SOURCE / "online_acceptance.json"), "script_sha256": sha(Path(__file__)),
        "alpha_after_checkpoints_matched_by_analyzer": checkpoints, "raw_run_count": len(values),
        "logged_policy_progress_predictions": 0, "availability_runtime_updates": 0,
        "figures": files, "boundaries": [
            "Actual warm continuous observer state; not a cold-prior offline estimator.",
            "Native static gamma1 does not use this state to choose actions; no policy-prediction error is claimed.",
            "Errors are versus noisy realized per-step conditional rates, not latent ground-truth probabilities.",
            "The two perf records per batch retain shared continuous EWMA state; no independent-system confidence interval.",
            "The alpha trajectory is first repeat at highest configured batch, including draining; not a whole-matrix trajectory.",
            "Gamma0 has zero tested exposure and null alpha errors; native availability state is an unupdated prior and is not plotted as learned."]}
    (OUTPUT / "plot_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "figures": files, "alpha_after_checks": checkpoints}))


if __name__ == "__main__":
    main()

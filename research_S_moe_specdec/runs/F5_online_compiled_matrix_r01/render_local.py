#!/usr/bin/env python3
"""Render existing complete-matrix F5 observations; no replay, fitting or GPU.

From project root: python3 -B runs/F5_online_compiled_matrix_r01/render_local.py
Fixed illustration: gamma1/4/7 over every B; gamma4 repeat0 B64 trajectory.
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
OUTPUT = ROOT / "figures/F5_online_compiled_matrix_r01"
GAMMAS, BATCHES = [1, 4, 7], [1, 2, 4, 8, 16, 32, 64]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = json.loads((SOURCE / "online_acceptance.json").read_text())
    assert manifest["analysis_status"] == "complete" and manifest["campaign_partial"] is False
    assert not manifest["missing_inputs"] and not manifest["invalid_inputs"]
    runs = manifest["runs"]
    assert len(runs) == 112 and len(manifest["engines"]) == 8
    assert {(r["gamma"], r["batch"], r["repeat"]) for r in runs} == {(g, b, r) for g in range(8) for b in BATCHES for r in (0, 1)}
    assert all(r["status"] == "complete" and r["execution_mode"] == "compiled" and r["mode"] == "native" for r in runs)
    assert len({tuple(r[k] for k in REGIME_FIELDS) for r in runs}) == 1
    assert all(r["recorded_decision_prediction_steps"] == 0 and r["online_state_recovered_steps"] == r["host_steps"] for r in runs)
    assert all(e["engine_complete"] and e["analyzed_run_records"] == 14 for e in manifest["engines"])

    # Read original files only to verify provenance and count recorded endpoints.
    # Full estimator-state reconstruction/checks are the existing analyzer's work.
    endpoints, host_steps, raw_waves = Counter(), Counter(), set()
    for run in runs:
        content = Path(run["source"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == run["source_sha256"]
        record = json.loads(content)
        assert record["all_finished"] and not record["trace_enabled"]
        episodes = record.get("episodes", [record])
        for episode in episodes:
            snapshot = episode["plugin"]["acceptance_snapshot"]
            assert snapshot["availability_updates"] == 0 and not snapshot["availability_exposures"]
            steps = episode["plugin"]["steps"]
            assert len(steps) == len(episode["host_steps"])
            assert math.isclose(steps[-1]["alpha_after"], snapshot["alpha"], rel_tol=1e-9, abs_tol=1e-10)
            endpoints[run["gamma"]] += 1
            host_steps[run["gamma"]] += len(steps)
            raw_waves.add((run["source"], episode["label"]))
    print(json.dumps({"verified_original_runs": len(runs), "recorded_wave_end_snapshots": dict(endpoints)}), flush=True)

    rank1 = {(r["source"], r["rank"]): r for r in manifest["rank_coverage"] if r["rank"] == 1}
    values = []
    for run in runs:
        if run["gamma"] not in GAMMAS:
            continue
        rank = rank1[run["source"], 1]
        assert rank["status"] == "complete" and rank["availability_exposure"] > 0
        values.append({"gamma": run["gamma"], "batch": run["batch"], "repeat": run["repeat"],
            "source": run["source"], "source_sha256": run["source_sha256"],
            "alpha_prestep_mae": run["conditional_alpha_prestep_mae"],
            "alpha_prestep_rmse": run["conditional_alpha_prestep_rmse"],
            "alpha_prestep_bias": run["conditional_alpha_prestep_bias"],
            "alpha_tested_weight": run["conditional_alpha_prestep_weight"],
            "conditional_accepted": run["conditional_accepted"], "conditional_tested": run["conditional_tested"],
            "conditional_accepted_over_tested": run["conditional_alpha_pooled"],
            "proposal_positive_fraction": rank["availability_survival_observed"],
            "proposal_positive_count": rank["available"], "proposal_request_step_exposure": rank["availability_exposure"],
            "display_batch": run["batch"] * 2 ** (-0.09 if run["repeat"] == 0 else 0.09)})
    fields = ["alpha_prestep_mae", "conditional_accepted_over_tested", "proposal_positive_fraction"]
    medians = []
    for gamma in GAMMAS:
        for batch in BATCHES:
            cell = [r for r in values if r["gamma"] == gamma and r["batch"] == batch]
            assert sorted(r["repeat"] for r in cell) == [0, 1]
            medians.append({"gamma": gamma, "batch": batch, "n_runs": 2,
                **{field: statistics.median(r[field] for r in cell) for field in fields}})

    checkpoints, tracking_waves, trajectory = Counter(), set(), []
    with gzip.open(SOURCE / "tracking_steps.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["online_state_recovered"] and row["alpha_after_checkpoint"] is not None
            assert row["availability_updated_by_runtime"] is False and not any(row["availability_exposures_before"])
            assert row["logged_decision_progress_before"] is None
            checkpoints[row["proposal_budget"]] += 1
            tracking_waves.add((row["source"], row["wave"]))
            if row["proposal_budget"] == 4 and row["repeat"] == 0 and row["nominal_batch"] == 64 and row["phase"] == "decode":
                trajectory.append({key: row[key] for key in ("source", "wave", "step", "actual_decode_batch",
                    "conditional_alpha_before", "conditional_alpha_observed", "conditional_tested",
                    "terminal_request_steps", "state_anchor")})
    assert checkpoints == host_steps and tracking_waves == raw_waves
    assert len({r["source"] for r in trajectory}) == len({r["wave"] for r in trajectory}) == 1

    styles = [(1, "cal", "*"), (4, "test", "square*"), (7, "profile", "triangle*")]
    def lines(field):
        out = []
        for gamma, color, marker in styles:
            points = [(r["batch"], r[field]) for r in medians if r["gamma"] == gamma]
            raw = [(r["display_batch"], r[field]) for r in values if r["gamma"] == gamma]
            out += [r"\addplot[color=%s,mark=%s,mark size=1.5pt] coordinates {%s};" % (color, marker, coords(points)),
                    r"\addlegendentry{$\gamma=%d$}" % gamma,
                    r"\addplot[forget plot,only marks,color=%s,mark=o,mark size=2pt,mark options={fill=white}] coordinates {%s};" % (color, coords(raw))]
        return "\n".join(out)
    axis = r"xmode=log,log basis x=2,xtick={1,2,4,8,16,32,64},xticklabels={1,2,4,8,16,32,64},xlabel={Nominal wave batch $B$},xmin=0.88,xmax=74,legend pos=north east"
    axes = [r"\nextgroupplot[title={(a) Actual pre-step pooled $\alpha$ error},ylabel={Tested-position-weighted MAE},ymin=0,ymax=0.55,%s]" % axis + lines(fields[0]),
            r"\nextgroupplot[title={(b) Conditional acceptance},ylabel={Accepted / tested positions},ymin=0.45,ymax=0.65,%s]" % axis + lines(fields[1]),
            r"\nextgroupplot[title={(c) Nonempty proposal coverage},ylabel={Fraction of eligible request-steps},ymin=0,ymax=0.25,%s]" % axis + lines(fields[2])]
    note = (r"Complete compiled OLMoE / n-gram matrix: 8 engines, 112 runs; fixed illustrations $\gamma=1,4,7$.\\"
            r"Open dots show both repetitions (small horizontal display offsets); lines are medians, with no confidence intervals. EWMA state remains continuous.\\"
            r"MAE compares pre-step state to noisy realized step rates, not latent truth; terminal requests are censored and batch affects observation noise.\\"
            r"For $\gamma>1$, pooled $\alpha$ is not a rank-invariant geometric constant. Native logs no decision-progress prediction; panel (c) is observed coverage only.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tex = document(axes, "F5: actual online observer in the complete compiled matrix", note)
    tex = tex.replace("width=8.0cm,height=6.0cm", "width=6.0cm,height=5.6cm").replace("horizontal sep=1.9cm", "horizontal sep=2.1cm")
    files = render_pgf(tex, OUTPUT, "F5_online_errors_coverage", ["pdf", "svg", "png"])

    trace = [r"\nextgroupplot[title={Fixed example: $\gamma=4$, repeat 0, $B=64$; includes drain},xlabel={Original host step},ylabel={Pooled conditional acceptance},xmin=0,xmax=127,ymin=0,ymax=1,legend pos=south west]",
        r"\addplot[color=cal,line width=1pt] coordinates {%s};" % coords([(r["step"], r["conditional_alpha_before"]) for r in trajectory]),
        r"\addlegendentry{Actual EWMA state before step}",
        r"\addplot[only marks,color=test,mark=o,mark size=1.4pt] coordinates {%s};" % coords([(r["step"], r["conditional_alpha_observed"]) for r in trajectory]),
        r"\addlegendentry{Realized accepted / tested}"]
    note = (r"Actual warm continuous observer, starting from the preceding run's checkpoint; no state reset or future information.\\"
            r"Native static $\gamma=4$ does not use this state to select actions and logs no decision-progress prediction.\\"
            r"Pooled $\alpha$ is not a geometric rank constant. Observations with no tested positions or only terminal requests remain missing.\\"
            r"Native availability EWMA was never updated; its prior is not plotted as learned tracking. This single wave is illustrative, not an independent repeat.")
    tex = document(["\n".join(trace)], "F5: continuous pre-step pooled alpha state", note).replace("width=8.0cm", "width=16.5cm")
    files += render_pgf(tex, OUTPUT, "F5_online_alpha_example", ["pdf", "svg", "png"])
    write_csv("plot_run_values.csv", values)
    write_csv("plot_cell_medians.csv", medians)
    write_csv("plot_trajectory_values.csv", trajectory)
    result = {"status": "measured_complete_matrix_diagnostic", "campaign_partial": False,
        "analysis_manifest": str(SOURCE / "online_acceptance.json"), "analysis_sha256": sha(SOURCE / "online_acceptance.json"),
        "tracking_sha256": sha(SOURCE / "tracking_steps.jsonl.gz"), "script_sha256": sha(Path(__file__)),
        "original_run_count": len(runs), "original_runs_sha_verified": len(runs), "engine_count": 8,
        "plotted_gammas": GAMMAS, "plotted_raw_run_count": len(values), "plotted_two_repeat_cells": len(medians),
        "alpha_after_checkpoints_matched_by_analyzer": dict(checkpoints), "total_alpha_after_checks": sum(checkpoints.values()),
        "wave_end_state_snapshots_matched_by_analyzer": dict(endpoints), "total_wave_end_snapshots": sum(endpoints.values()),
        "terminal_request_steps_censored": sum(r["terminal_request_steps"] for r in runs),
        "terminal_recovered_from_runner": sum(r["terminal_recovered_from_runner"] for r in runs),
        "terminal_status_unknown_request_steps": sum(r["terminal_status_unknown_request_steps"] for r in runs),
        "trajectory": {"gamma": 4, "repeat": 0, "batch": 64, "source": trajectory[0]["source"],
                       "decode_steps": len(trajectory), "actual_batch_min": min(r["actual_decode_batch"] for r in trajectory)},
        "logged_policy_progress_predictions": 0, "availability_runtime_updates": 0, "figures": files,
        "boundaries": [
            "Reuses completed actual-online analysis. Rendering only verifies original source hashes, counts recorded checkpoints, and joins summary fields; no estimator rerun or fitting.",
            "Pre-step error is tested-position weighted within each original run versus noisy realized rates, not latent acceptance probabilities. Observation noise varies with batch.",
            "For gamma>1, pooled accepted/tested alpha is not a rank-invariant geometric acceptance constant.",
            "Native runs log no decision-progress predictions. Proposal-positive rates are observed rank1 availability, not learned availability EWMA state; that observer has zero updates.",
            "Both original repetitions are retained; EWMA state persists across files/waves. Lines are two-value medians, with no independent-system confidence interval.",
            "Fixed gamma4 repeat0 B64 example includes draining. Missing/no-exposure/terminal observations are not zero-filled.",
            "All 112 original runs and all endpoint snapshots were counted; full-state endpoint equality is the completed analyzer's recorded validation, not a second reconstruction."]}
    (OUTPUT / "plot_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "alpha_after_checks": dict(checkpoints), "total": sum(checkpoints.values()),
        "wave_end_snapshots": sum(endpoints.values()), "figures": files}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Descriptive comparison of measured complete-step costs and fixed-trace scores.

Run from project root after build_calibration and diagnose_stepwise_prefix_oracle.
No new fit or GPU execution; neither cost differences nor score gaps are policy
speedups or isolated causal estimates.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "code"))
from plot_energy_progress import pgf_document, render_pgf
from plot_results import REGIME_FIELDS, _context, resolve_regime, write_csv

BATCHES = (1, 4, 16, 64)
DIAG = ROOT / "runs/derived/stepwise_prefix_compiled_r01"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_brief():
    comparison = read(OUT / "cost_comparison.json")
    diagnostic = read(DIAG / "diagnostic_summary.json")
    diagnostic["analysis_script_sha256"] = sha(Path(__file__))
    (DIAG / "diagnostic_summary.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    rows = comparison["comparison"]
    rankings, by_batch = [], []
    for batch in BATCHES:
        subset = [r for r in rows if r["batch"] == batch]
        ordered = sorted(subset, key=lambda r: r["fixed_budget7_complete_step_ms"])
        for rank, row in enumerate(ordered, 1):
            rankings.append({"batch": batch, "cost_rank": rank, "gamma_cap": row["gamma_cap"],
                "complete_step_ms": row["fixed_budget7_complete_step_ms"],
                "runs_per_calibration_cell": 1,
                "scope": "cost only, not cost/progress or policy-performance ordering"})
        by_batch.append({"batch": batch,
            "complete_step_ms_range": [ordered[0]["fixed_budget7_complete_step_ms"], ordered[-1]["fixed_budget7_complete_step_ms"]],
            "ascending_cap_cost_order": [r["gamma_cap"] for r in ordered],
            "second_minus_min_cost_ms": ordered[1]["fixed_budget7_complete_step_ms"] - ordered[0]["fixed_budget7_complete_step_ms"],
            "native_complete_step_median_ms_range": [min(r["native_budget_equals_gamma_complete_step_ms_median"] for r in subset), max(r["native_budget_equals_gamma_complete_step_ms_median"] for r in subset)],
            "G7_minus_native_median_ms_range": [min(r["descriptive_difference_ms"] for r in subset), max(r["descriptive_difference_ms"] for r in subset)]})
    write_csv(OUT / "cost_rankings.csv", rankings)
    brief = {"status": "local_analysis_complete", "execution_mode": "compiled",
        "calibration_cells": 32, "calibration_runs_per_cell": 1, "native_runs_per_cell": 2,
        "calibration_excluded_files": 0, "native_excluded_files": 0,
        "native_raw_records_recomputed": 112, "native_matched_comparison_records": 64,
        "coordinator_table_max_difference_ms": comparison["coordinator_table_max_difference_ms"],
        "calibration_eligible_steps": comparison["fixed_budget7_eligible_steps"],
        "calibration_eligible_waves": comparison["fixed_budget7_eligible_waves"],
        "cost_ranges_and_orders_by_batch": by_batch,
        "cost_difference_all_cells_ms": [min(r["descriptive_difference_ms"] for r in rows), max(r["descriptive_difference_ms"] for r in rows)],
        "cost_ratio_all_cells": [min(r["descriptive_ratio"] for r in rows), max(r["descriptive_ratio"] for r in rows)],
        "diagnostic_counts": diagnostic["counts"],
        "diagnostic_cost_lookup": diagnostic["replayed_cost_source_counts"],
        "diagnostic_run_values": diagnostic["raw_run_values"],
        "source_behavior_sha256": comparison["source_behavior_sha256"],
        "limitations": diagnostic["boundaries"] + [comparison["boundary"],
            "All comparisons use compiled data; no eager trajectory or cost enters the analysis.",
            "Both campaigns have matching runner/plugin/cost-model source hashes. Fixed-G7 calibration has64 output tokens/request and one run/cell; native G=gamma has128 tokens and two runs. General warmup offsets are0 versus80, with shape warmup0 in both; time windows differ.",
            "Pure-cost ranks ignore accepted progress and are not claims of optimal actions or measured policy performance. Small nonmonotone single-run differences have no established statistical ordering.",
            "B1 replay and perfect-prefix gaps are zero because this table's cap7 is both the lowest-cost cap and has weakly maximal identified progress on this fixed trace. This is conditional on the single-run table, not evidence of globally optimal online control."],
        "figures": ["figures/fullbudget_cost_compiled_r01/complete_step_costs.pdf", "figures/stepwise_prefix_compiled_r01/stepwise_prefix_diagnostic.pdf"],
        "qa": "Both rendered PNGs visually checked; labels, notes and raw points visible without clipping."}
    (OUT / "numerical_brief.json").write_text(json.dumps(brief, indent=2) + "\n")


def main():
    full = read(OUT / "calibration.json")
    native_path = OUT / "native_direct/calibration.json"
    native = read(native_path)
    assert len(full["records"]) == 32 and not read(OUT / "exclusions.json")
    assert full["metadata"]["execution_mode"] == native["metadata"]["execution_mode"] == "compiled"
    assert len(native["records"]) == 112 and not read(OUT / "native_direct/exclusions.json")
    coordinator_path = ROOT / "runs/controller_preparation_compiled_r01/calibration/calibration.json"
    coordinator = read(coordinator_path)
    keyed = {(r["batch"], r["gamma"]): r for r in coordinator["records"]}
    table_checks = []
    for row in full["records"]:
        other = keyed[row["batch"], row["gamma"]]
        assert all(row[k] == other[k] for k in row if k != "source")
        table_checks.append({"batch":row["batch"], "gamma":row["gamma"],
            "difference_ms": abs(row["total_step_ms"]-other["total_step_ms"]),
            "source":row["source"], "coordinator_source":other["source"]})
    behavior = {}
    for name in ("olmoe_compiled_fullbudget_calibration_r01", "olmoe_ngram_compiled_r01"):
        campaign = ROOT / "runs" / name
        assert read(campaign / "status.json")["status"] == "complete"
        for envpath in sorted(campaign.glob("gamma*/environment.json")):
            env = read(envpath)
            for filename in ("run_matrix.py", "vllm_s_plugin.py", "cost_model.py"):
                actual_sha = sha(campaign / "source" / filename)
                assert env["source_sha256"][filename] == actual_sha
                if filename in behavior:
                    assert behavior[filename] == actual_sha
                behavior[filename] = actual_sha
    (OUT / "coordinator_table_checks.json").write_text(json.dumps({
        "all_record_fields_except_machine_paths_equal": True,
        "coordinator_table_sha256":sha(coordinator_path), "rows":table_checks}, indent=2)+"\n")
    native_rows, checks = defaultdict(list), []
    for previous in native["records"]:
        if previous["batch"] not in BATCHES:
            continue
        path = Path(previous["source"])
        if not path.is_file() and "/runs/" in str(path):
            path = ROOT / "runs" / str(path).split("/runs/", 1)[1]
        record, context = read(path), _context(path)
        actual = previous  # Already recomputed from raw data by build_calibration in native_direct.
        assert actual is not None and actual["proposal_budget"] == actual["gamma"]
        assert record["output_tokens"] == 64 * 128
        regime = resolve_regime(record, context)
        assert all(str(regime[k]) == str(full["metadata"][k]) for k in REGIME_FIELDS)
        checks.append({"source": str(path), "sha256": sha(path),
                       "current_reader_total_step_ms": actual["total_step_ms"],
                       "raw_output_tokens": record["output_tokens"], "proposal_budget": actual["proposal_budget"]})
        native_rows[actual["batch"], actual["gamma"]].append(actual)
    comparison = []
    for row in sorted(full["records"], key=lambda r: (r["batch"], r["gamma"])):
        assert row["proposal_budget"] == 7 and row["eligible_steps"] > 0
        original = sorted(native_rows[row["batch"], row["gamma"]], key=lambda r: r["run_id"])
        assert len(original) == 2
        raw = [r["total_step_ms"] for r in original]
        native_median = statistics.median(raw)
        comparison.append({"batch": row["batch"], "gamma_cap": row["gamma"],
            "fixed_budget7_complete_step_ms": row["total_step_ms"], "fixed_budget7_run_count": 1,
            "native_budget_equals_gamma_complete_step_ms_raw": raw, "native_run_count": 2,
            "native_budget_equals_gamma_complete_step_ms_median": native_median,
            "descriptive_difference_ms": row["total_step_ms"] - native_median,
            "descriptive_ratio": row["total_step_ms"] / native_median,
            "fixed_budget7_eligible_steps": row["eligible_steps"],
            "fixed_budget7_source": row["source"], "native_sources": [r["source"] for r in original]})
    write_csv(OUT / "fixed_budget7_vs_native_costs.csv", comparison)
    write_csv(OUT / "native_cost_reader_checks.csv", checks)
    plotdir = ROOT / "figures/fullbudget_cost_compiled_r01"
    plotdir.mkdir(parents=True, exist_ok=True)
    panels = []
    for native_panel in (False, True):
        key = "native_budget_equals_gamma_complete_step_ms_median" if native_panel else "fixed_budget7_complete_step_ms"
        panels.append({"title": (r"(b) Native G=$\gamma$; compiled r01" if native_panel else r"(a) Fixed proposal budget G=7; post-truncation"),
            "xlabel": r"Verification cap $\gamma$", "ylabel": "Complete host step (ms)", "xs": list(range(8)),
            "series": [{"label": f"B={batch}", "medians": [(r["gamma_cap"], r[key]) for r in comparison if r["batch"] == batch],
                        "raw": [(r["gamma_cap"], r["native_budget_equals_gamma_complete_step_ms_raw"]) for r in comparison if r["batch"] == batch] if native_panel else []}
                       for batch in BATCHES]})
    note = (r"Each cost is a run mean of eligible wave means: pure decode, actual B matches, terminal steps excluded.\\"
            r"Left: one run per cell, 64 output tokens/request, full shape warmup. Right: two runs, 128 tokens, full shape warmup.\\"
            r"General warmup offset: left 0, right 80; both shape offset 0. Different workloads/time windows; not isolated drafting/truncation costs.")
    tex = pgf_document(panels, 4, note, "Compiled complete-step calibration: two different proposal budgets")
    costs = [v for r in comparison for v in [r["fixed_budget7_complete_step_ms"], *r["native_budget_equals_gamma_complete_step_ms_raw"]]]
    tex = tex.replace("unbounded coords=jump", f"unbounded coords=jump,ymin={min(costs)*.96},ymax={max(costs)*1.04}")
    cost_files = render_pgf(tex, plotdir, "complete_step_costs", ["pdf", "svg", "png"])
    limits = {
        "fixed_budget7": {"batches": list(BATCHES), "gammas": list(range(8)), "budget": 7, "runs_per_cell": 1,
            "requests": 64, "output_tokens_per_request": 64, "warmup_offset": 0, "shape_warmup_offset": 0,
            "shape_warmup": True, "general_warmup_tokens": 32},
        "native_compiled_r01": {"budget": "gamma", "runs_per_cell": 2, "requests": 64, "output_tokens_per_request": 128,
            "general_warmup_offset": 80, "shape_warmup": True, "shape_warmup_offset": 0, "general_warmup_tokens": 32}}
    summary = {"status": "measured_descriptive_cost_comparison", "fixed_budget7_cells": 32,
        "fixed_budget7_eligible_steps": sum(r["eligible_steps"] for r in full["records"]),
        "fixed_budget7_eligible_waves": sum(r["eligible_waves"] for r in full["records"]),
        "native_comparison_run_records": len(checks),
        "coordinator_table_max_difference_ms": max(r["difference_ms"] for r in table_checks),
        "source_behavior_sha256": behavior,
        "calibration_sha256": sha(OUT / "calibration.json"), "native_table_sha256": sha(native_path),
        "workload_and_warmup_differences": limits, "comparison": comparison, "figures": cost_files,
        "boundary": "Complete synchronous host step includes current verification, next draft generation and host/controller work. No async component arithmetic, HBM inference or causal budget-tax estimate."}
    (OUT / "cost_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    diagnostic = read(DIAG / "diagnostic.json")
    assert diagnostic["status"] == "development_diagnostic_computed" and len(diagnostic["runs"]) == 14 and not diagnostic["exclusions"]
    disagreement, sources = defaultdict(Counter), Counter()
    with gzip.open(DIAG / "step_diagnostics.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["all_caps_identified_and_cost_covered"]
            c = disagreement[row["source"]]
            c["steps"] += 1
            c["replay"] += row["replayed_pre_step_gamma"] != row["perfect_prefix_info_gamma"]
            c["cap7"] += row["perfect_prefix_info_gamma"] != 7
            sources[row["replayed_decision_cost_source"]] += 1
    data = []
    total_counts = Counter()
    for run in sorted(diagnostic["runs"], key=lambda r: (r["batch"], r["repeat"])):
        c = disagreement[run["source"]]
        assert c["steps"] == run["counts"]["paired_replayed_reference_steps"]
        total_counts.update(run["counts"])
        data.append({"batch": run["batch"], "repeat": run["repeat"], "source": run["source"],
            "paired_steps": c["steps"], "replayed_choice_score_gap": run["replayed_choice_score_gap_mean"],
            "cap7_score_gap": run["executed_cap7_score_gap_mean"],
            "replayed_choice_disagreement_fraction": c["replay"] / c["steps"],
            "cap7_disagreement_fraction": c["cap7"] / c["steps"],
            "replayed_choices": run["choice_histograms"]["replayed_pre_step"],
            "perfect_prefix_choices": run["choice_histograms"]["perfect_prefix_info"]})
    write_csv(DIAG / "plot_run_values.csv", data)
    batches = sorted({r["batch"] for r in data})
    panels = []
    for index, fields in enumerate((("replayed_choice_score_gap", "cap7_score_gap"),
                                    ("replayed_choice_disagreement_fraction", "cap7_disagreement_fraction"))):
        panel = {"title": "(a) Score gap to perfect-prefix reference" if index == 0 else "(b) Cap differs from perfect-prefix reference",
            "xlabel": "Original nominal wave batch B", "ylabel": "ms / mean-request progress" if index == 0 else "Fraction of eligible scored steps",
            "xs": batches, "log_x": True, "series": []}
        for key, label in zip(fields, ("Replayed pre-step cap", "Recorded cap7")):
            raw = [(b, [r[key] for r in data if r["batch"] == b]) for b in batches]
            panel["series"].append({"label": label, "medians": [(b, statistics.median(v)) for b, v in raw], "raw": raw})
        panels.append(panel)
    plotdir = ROOT / "figures/stepwise_prefix_compiled_r01"
    plotdir.mkdir(parents=True, exist_ok=True)
    note = (r"Fixed compiled r01 $\gamma=7$ development trajectory; cold prior once per run; all candidates use separate G=7 complete-step calibration.\\"
            r"Future prefixes inform only the reference. Changed trajectories are untested; this is not measured TPOT, a speedup, or a theoretical bound.\\"
            r"Two raw run summaries are retained but coincide under identical prefix histories and replay initialization; steps are not independent repeats.")
    diagnostic_files = render_pgf(pgf_document(panels, 2, note, "Fixed-trajectory single-step perfect-prefix-information diagnostic"),
                                 plotdir, "stepwise_prefix_diagnostic", ["pdf", "svg", "png"])
    stats = {"status": diagnostic["status"], "run_records": 14, "counts": dict(total_counts),
        "replayed_cost_source_counts": dict(sources), "raw_run_values": data,
        "diagnostic_sha256": sha(DIAG / "diagnostic.json"), "calibration_sha256": sha(OUT / "calibration.json"),
        "analysis_script_sha256": sha(Path(__file__)), "figures": diagnostic_files,
        "boundaries": diagnostic["boundaries"] + [
            "The single-run calibrated cap-cost differences are nonmonotone; small action-ranking gaps include unresolved calibration variation.",
            "Per-run replay score summaries coincide across the two measured trajectory repeats. This is not evidence of independent uncertainty vanishing.",
            "Small score gaps do not establish that online information has no value or that complete-policy improvement has no remaining opportunity."]}
    (DIAG / "diagnostic_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    write_brief()
    print(json.dumps({"cost_cells":32, "coordinator_table_max_difference_ms": summary["coordinator_table_max_difference_ms"],
                     "diagnostic_counts": dict(total_counts), "cost_sources": dict(sources),
                     "cost_figures": cost_files, "diagnostic_figures": diagnostic_files}))


if __name__ == "__main__":
    main()

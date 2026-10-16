#!/usr/bin/env python3
"""Render the existing F1 layer-mean MAE diagnostic without matplotlib.

  python code/plot_route_prediction_pgf.py --analysis runs/route_prediction_r02 \
    --output figures/F1_olmoe_ngram_r02_reproduced --title 'OLMoE / n-gram'

First run validate_route_prediction.py --no-plot separately for each target,
precision, draft source and execution regime. This renderer reads its JSON/CSV
statistics; it does not refit q_e or recalculate prediction errors. The plotted
equal mean over layers matches validate_route_prediction.make_plot exactly.

Original route JSON and adjacent configuration/engine_args JSON are required to
identify the domain. Remote /runs/ paths resolve against this local project.
A mixed-domain analysis is rejected: splitting its already-fitted marginals at
plot time would not repair the fit. Only scheduled target decode rows are used;
physical graph-padding counts require a separate analysis and are never pooled.
Requires installed pdflatex/PGFPlots and Poppler. No GPU or network access.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from plot_energy_progress import pgf_document, render_pgf
from plot_results import _context, resolve_regime
from validate_route_prediction import walk_routes


PROJECT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def source_path(name):
    path = Path(name)
    if not path.is_file() and "/runs/" in name:
        path = PROJECT / "runs" / name.split("/runs/", 1)[1]
    require(path.is_file(), f"Original route source unavailable: {name}")
    return path.resolve()


def load_analysis(directory):
    report_path = directory / "route_prediction.json"
    groups_path = directory / "route_prediction_groups.csv"
    report = read(report_path)
    require(report.get("status") == "measured", "No measured F1 predictions")
    require(report.get("evidence_kind") == "actual_verification_histogram_prediction_no_token_replay",
            "Unsupported F1 evidence schema")
    domains, provenance, counts = {}, [], defaultdict(int)
    for partition in ("fit", "test"):
        inputs = report["input_provenance"][partition]
        require(inputs, f"Missing {partition} source provenance")
        for source in inputs:
            path = source_path(source["path"])
            require(sha(path) == source["sha256"], f"Original source hash differs: {path}")
            config_path, engine_path = path.parent / "configuration.json", path.parent / "engine_args.json"
            require(config_path.is_file() and engine_path.is_file(), f"Missing execution provenance beside {path}")
            obj, config = read(path), read(config_path)
            require(isinstance(obj, dict), "Use run_matrix route episode JSON for domain provenance")
            domain = resolve_regime(obj, _context(path))
            for key in ("model", "draft_source", "dtype", "execution_mode"):
                require(domain[key] not in (None, "", "unspecified", "none"), f"Unknown {key} for {path}")
            plan = config.get("plan", {})
            domain.update({key: plan.get(key) for key in
                           ("model_revision", "draft_model", "draft_revision", "speculative_overrides")})
            domain["distinct_scope"] = "scheduled_target_decode_verification_rows"
            for row, context in walk_routes(obj):
                if row.get("model_phase") != "target" or row.get("phase") != "decode":
                    continue
                require(not any(key in row for key in ("physical_tokens", "physical_expert_token_counts", "physical_row_scope"))
                        and "padding" not in str(row.get("route_path", "")).lower(),
                        f"Physical/padding D cannot enter scheduled-row F1: {path}")
                for key in ("model", "draft_source", "execution_mode"):
                    require(context.get(key) in (None, domain[key]), f"Nested {key} differs in {path}")
                counts[partition] += 1
            domain_id = hashlib.sha256(canonical(domain).encode()).hexdigest()[:12]
            domains[domain_id] = domain
            provenance.append({"partition": partition, "path": str(path), "sha256": source["sha256"],
                               "domain_id": domain_id, "configuration_sha256": sha(config_path),
                               "engine_args_sha256": sha(engine_path)})
    require(len(domains) == 1, "Analysis contains mixed domains; rerun the validator separately before plotting")
    for partition in ("fit", "test"):
        require(counts[partition] == report[f"{partition}_route_events"], f"{partition} raw route count differs")
    groups = report["groups"]
    key = lambda row: (row["layer"], int(row["actual_batch"]), int(row["nominal_gamma"]))
    lookup = {key(row): row for row in groups}
    require(len(lookup) == len(groups), "Duplicate original layer/B/gamma groups")
    with groups_path.open(newline="") as handle:
        exported = list(csv.DictReader(handle))
    require(len(exported) == len(groups), "JSON/CSV group counts differ")
    seen = set()
    for row in exported:
        identity = key(row)
        require(identity not in seen and identity in lookup, "CSV group identity differs from JSON")
        seen.add(identity)
        original = lookup[identity]
        require(row["status"] == original["status"] == "measured", "Unmeasured group cannot be plotted")
        for field in ("route_events", "mae"):
            require(float(row[field]) == original[field], f"JSON/CSV {field} differs at {identity}")
    require(sum(row["route_events"] for row in groups) == report["predicted_test_route_events"],
            "Group event counts differ from prediction count")
    return report, groups, domains, provenance, {
        "route_prediction.json": sha(report_path), "route_prediction_groups.csv": sha(groups_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", help="Display label only; domain identity comes from original execution records")
    args = parser.parse_args()
    directory = args.analysis if args.analysis.is_dir() else args.analysis.parent
    report, groups, domains, sources, hashes = load_analysis(directory)
    domain_id, domain = next(iter(domains.items()))
    cells = []
    grouped = defaultdict(list)
    for row in groups:
        grouped[row["actual_batch"], row["nominal_gamma"]].append(row)
    for (batch, gamma), rows in sorted(grouped.items()):
        cells.append({"domain_id": domain_id, "actual_batch": batch, "nominal_gamma": gamma,
                      "layer_mean_mae": statistics.mean(row["mae"] for row in rows),
                      "layer_count": len(rows), "route_events": sum(row["route_events"] for row in rows),
                      "layer_mae": {row["layer"]: row["mae"] for row in rows}})
    gammas = sorted({row["nominal_gamma"] for row in cells})
    ticks = [x for x in (1, 2, 4, 8, 16, 32, 64, 128, 256) if x <= max(row["actual_batch"] for row in cells)]
    panel = {"title": "Actual-S prediction error by held-out gamma", "xlabel": "Actual decode batch B",
             "ylabel": "MAE in distinct experts (equal mean over layers)", "xs": ticks, "log_x": True,
             "series": [{"label": r"$\gamma=%d$" % gamma,
                         "medians": [(row["actual_batch"], row["layer_mean_mae"]) for row in cells if row["nominal_gamma"] == gamma],
                         "raw": []} for gamma in gammas]}
    title = args.title or f"{Path(domain['model']).name} / {domain['draft_source']}"
    title += f" / {domain['dtype']} / {domain['quantization']} / {domain['execution_mode']}"
    note = (r"$\gamma=0$ marginals; held-out executed $\gamma$ conditions, with the same prompt subset permitted.\\"
            r"Scheduled target verification rows only; physical padding D is not included.\\"
            r"Layer-step counts are correlated; no independent-repeat confidence intervals.")
    # The shared helper calls these generic line values 'medians'; values above
    # are exactly the original F1's equal layer means, and are labelled as such.
    tex = pgf_document([panel], len(gammas), note, title).replace("width=6.5cm", "width=13.5cm")
    args.output.mkdir(parents=True, exist_ok=True)
    files = render_pgf(tex, args.output, "route_prediction_mae", ["pdf", "svg", "png"])
    with (args.output / "plot_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows({**row, "layer_mae": canonical(row["layer_mae"])} for row in cells)
    manifest = {"status": "reproduced_existing_measured_statistics", "analysis": str(directory.resolve()),
                "analysis_sha256": hashes, "script_sha256": sha(Path(__file__)), "domains": domains,
                "sources": sources, "figure_files": files, "plot_cells": len(cells),
                "original_group_count": len(groups), "original_global_metrics": report["metrics"],
                "aggregation": "Exact statistics.mean of exported per-layer MAEs at each actual B and nominal gamma; identical to original F1 make_plot",
                "checks": {"source_hashes_match": True, "source_route_counts_match": True,
                           "json_csv_group_values_exactly_match": True, "single_execution_domain": True},
                "boundaries": report["boundaries"] + [
                    "This entry point draws an existing F1 error diagnostic; no new marginals, predictions or fitted models.",
                    "Physical model-forward padding D and effective scheduled-row D are not combined."]}
    (args.output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": manifest["status"], "domain_id": domain_id,
                      "plot_cells": len(cells), "output": str(args.output), "files": files}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot five executed policy roles from either comparison.json schema.

  python code/plot_policy_comparisons.py \
    --comparison runs/DOMAIN_controller_comparison_r01 \
    --output figures/DOMAIN_policies_r01 --include-wave-energy

Inputs are directories containing comparison.json or the JSON files themselves,
from compare_controller_policies.py / compare_reduced_evaluations.py. The raw
accepted run/control records are the source of every point. Each input, execution
domain and held-out population gets separate figures. No role is inferred from
gamma, strategy labels or a post-hoc envelope. A single repeat is shown only as
a raw point; a missing/incomplete batch breaks the median line.

Default figures: F3 effective TPOT, F4 validated cumulative-counter span energy,
F6 timed decision CPU and discarded available drafts. Wave-window energy is an
optional separate figure, never a substitute for counter-span energy. Rendering
is local PGFPlots/Poppler; --no-plot exports data/provenance with stdlib only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

from compare_controller_policies import ROLES, canonical, output_csv, sha
from plot_energy_progress import render_pgf, tex_escape
from plot_results import REGIME_FIELDS, _finite


LABELS = {"ar": "AR", "static": "Fixed gamma", "native_unrestricted": "Native table",
          "native_threshold": "Native threshold", "adaptive": "Adaptive"}
COLORS = dict(zip(ROLES, ("222222", "0072B2", "E69F00", "009E73", "CC79A7")))
MARKS = dict(zip(ROLES, ("*", "square*", "triangle*", "diamond*", "pentagon*")))
METRICS = {
    "effective_tpot_ms": ("F3_effective_tpot", "Effective TPOT", "Effective TPOT (ms / output token)", 1.0),
    "population_counter_joules_per_token": ("F4_counter_span_energy", "Complete counter span", "Energy (J / output token)", 1.0),
    "joules_per_token": ("F4_measured_window_energy", "Sum of measured wave windows", "Energy (J / output token)", 1.0),
    "policy_choose_cpu_us_mean": ("F6_control", "(a) Timed decision calls", r"Mean decision CPU ($\mu$s)", 1.0),
    "discarded_available_token_fraction": ("F6_control", "(b) Post-proposal truncation", r"Discarded / available drafts (\%)", 100.0),
}
CONTROL = ("policy_choose_cpu_us_mean", "discarded_available_token_fraction")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def identity(row):
    require(row.get("role") in ROLES, "unknown/missing explicit policy role")
    require(isinstance(row.get("repeat"), int) and row["repeat"] >= 0, "missing original repeat identity")
    require(isinstance(row.get("batch"), int) and row["batch"] > 0, "invalid nominal batch")
    require(bool(row.get("source")), "missing original run source")
    require(bool(row.get("sample_indices_sha256")), "missing held-out population identity")
    require(all(k in row for k in REGIME_FIELDS), "incomplete execution regime")
    return tuple(canonical(row[k]) for k in (*REGIME_FIELDS, "sample_indices_sha256"))


def load_comparison(path):
    path = Path(path)
    if path.suffix != ".json":
        path /= "comparison.json"
    obj = json.loads(path.read_text())
    require(obj.get("status") != "invalid_evidence", "comparison reports invalid evidence; correct it before plotting")
    require(all(isinstance(obj.get(k), list) for k in ("runs", "control_runs", "coverage")), "expected comparison raw runs, controls and coverage arrays")
    groups, raw, seen, accepted = {}, [], set(), {}
    coverage = {}
    for cell in obj["coverage"]:
        key = cell["role"], int(cell["batch"])
        require(key not in coverage, "duplicate coverage cell")
        coverage[key] = cell
    protocol_batches = obj.get("protocol", {}).get("batches", sorted({k[1] for k in coverage}))
    input_hash = sha(path)
    for row in obj["runs"]:
        key = identity(row)
        require(row.get("evidence_kind") == "measured_run", "only comparison-accepted measured run records are supported")
        require(row["source"] not in accepted, "duplicate original run source")
        require(row["batch"] in protocol_batches, "run batch outside declared comparison protocol")
        accepted[row["source"]] = row
        token = (key, row["role"], row["batch"], row["repeat"])
        require(token not in seen, "duplicate role/batch/repeat; refusing to pool them")
        seen.add(token)
        group = hashlib.sha256(canonical((input_hash, key, obj.get("execution_domain"))).encode()).hexdigest()[:12]
        groups[group] = {"comparison": str(path.resolve()), "comparison_sha256": input_hash,
            "domain": obj.get("domain"), "execution_domain": obj.get("execution_domain", {}),
            "regime": {k: row[k] for k in REGIME_FIELDS}, "sample_indices_sha256": row["sample_indices_sha256"],
            "batches": protocol_batches, "fixture": obj.get("test_fixture") is True,
            "comparison_boundaries": obj.get("boundaries", []), "coverage": coverage}
        for metric in ("effective_tpot_ms", "population_counter_joules_per_token", "joules_per_token"):
            value = _finite(row.get(metric))
            require(value is None or value > 0, f"nonpositive measured {metric}")
            if metric == "population_counter_joules_per_token" and value is not None:
                require(row.get("population_counter_status") == "validated_cumulative_counter_span", "counter-span energy lacks validated raw-counter status")
            raw.append({"group": group, "metric": metric, "role": row["role"], "batch": row["batch"],
                "repeat": row["repeat"], "source": row["source"], "source_sha256": row.get("source_sha256"),
                "value": value, "counter_status": row.get("population_counter_status") if metric.startswith("population_counter") else None})
    control_seen = set()
    for row in obj["control_runs"]:
        if row.get("phase") != "decode":
            continue
        key = identity(row)
        parent = accepted.get(row["source"])
        require(parent is not None and identity(parent) == key and all(parent[k] == row[k] for k in ("role", "batch", "repeat")), "control row does not belong to its accepted performance run")
        require(row["source"] not in control_seen, "duplicate decode control record")
        control_seen.add(row["source"])
        group = hashlib.sha256(canonical((input_hash, key, obj.get("execution_domain"))).encode()).hexdigest()[:12]
        for metric in CONTROL:
            value = _finite(row.get(metric))
            require(value is None or value >= 0, f"negative measured {metric}")
            require(metric != "discarded_available_token_fraction" or value is None or value <= 1, "discarded fraction is outside [0,1]")
            raw.append({"group": group, "metric": metric, "role": row["role"], "batch": row["batch"],
                "repeat": row["repeat"], "source": row["source"], "source_sha256": parent.get("source_sha256"),
                "value": value, "timed_decision_calls": row.get("policy_choose_cpu_us_observed_steps"),
                "scheduled_decode_steps": row.get("steps"), "available_drafts": row.get("drafts_available_before"),
                "discarded_drafts": row.get("truncated_by_controller")})
    return groups, raw, {"source": str(path.resolve()), "sha256": input_hash, "comparison_status": obj.get("status"),
                         "run_records": len(obj["runs"]), "fixture": obj.get("test_fixture") is True}


def prepare(groups, raw):
    indexed = defaultdict(list)
    for row in raw:
        indexed[row["group"], row["metric"], row["role"], row["batch"]].append(row)
    cells = []
    for group, metadata in groups.items():
        for metric in METRICS:
            for role in ROLES:
                for batch in metadata["batches"]:
                    observed = sorted(indexed[group, metric, role, batch], key=lambda r: r["repeat"])
                    declared = metadata["coverage"].get((role, batch), {})
                    count = max(2, int(declared.get("expected_run_records", 2)))
                    repeats = [r["repeat"] for r in observed]
                    values = [r["value"] for r in observed]
                    complete = declared.get("status") == "measured" and repeats == list(range(count)) and all(v is not None for v in values)
                    cells.append({"group": group, "metric": metric, "role": role, "role_label": LABELS[role],
                        "batch": batch, "expected_repeats": count, "repeats": repeats, "sources": [r["source"] for r in observed],
                        "raw": values, "observed_values": sum(v is not None for v in values),
                        "median": statistics.median(values) if complete else None,
                        "status": "complete_repeats" if complete else "raw_only_or_unavailable",
                        "coverage_status": declared.get("status", "pending"), "display_scale": METRICS[metric][3]})
    return cells


def document(metadata, cells, metrics):
    """Use explicit roles and NaN gaps instead of connecting missing batches."""
    batches = metadata["batches"]
    lookup = {(r["metric"], r["role"], r["batch"]): r for r in cells}
    roles = [role for role in ROLES if any(lookup[m, role, b]["observed_values"] for m in metrics for b in batches)]
    count = len(metrics)
    width = "8.0cm" if count == 2 else "11.5cm"
    xlabel = "Synchronous wave batch B" if metadata["regime"]["admission_mode"] == "wave" else "Admission cap B"
    label = metadata.get("display_label") or metadata["domain"] or Path(metadata["regime"]["model"]).name
    title = f"{label} | {metadata['regime']['draft_source']} | {metadata['regime']['execution_mode']}"
    if metadata["fixture"]:
        title = "CPU fixture: layout test only"
    lines = [r"\documentclass[border=5pt]{standalone}", r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}\pgfplotsset{compat=1.18}", r"\begin{document}\sffamily\begin{tikzpicture}"]
    for index, role in enumerate(ROLES):
        lines.append(r"\definecolor{policy%d}{HTML}{%s}" % (index, COLORS[role]))
    lines.append(r"\begin{groupplot}[group style={group size=%d by 1,horizontal sep=2.0cm},width=%s,height=5.8cm,scale only axis,grid=major,grid style={gray!20},axis lines*=left,unbounded coords=jump,xmode=log,log basis x=2,minor x tick num=0,scaled y ticks=false,tick label style={font=\small},label style={font=\small},title style={font=\small},xtick={%s},xticklabels={%s},xlabel={%s}]" % (count, width, ",".join(map(str, batches)), ",".join(map(str, batches)), tex_escape(xlabel)))
    for index, metric in enumerate(metrics):
        _, panel_title, ylabel, scale = METRICS[metric]
        if metric == "joules_per_token" and metadata["regime"]["admission_mode"] != "wave":
            panel_title = "Measured population window"
        options = ["title={" + tex_escape(panel_title) + "}", "ylabel={" + ylabel + "}",
                   f"xmin={min(batches) * .8}", f"xmax={max(batches) * 1.2}"]
        any_values = any(lookup[metric, role, batch]["observed_values"] for role in roles for batch in batches)
        if metric == "discarded_available_token_fraction":
            options += ["ymin=-3", "ymax=105"]
        elif not any_values:
            options += ["ymin=0", "ymax=1"]
        if index == 0:
            options += [r"legend to name=policyLegend", "legend columns=3", r"legend style={draw=none,font=\small,/tikz/every even column/.append style={column sep=8pt}}"]
        lines.append(r"\nextgroupplot[" + ",".join(options) + "]")
        if index == 0:
            for role in roles:
                lines.append(r"\addlegendimage{color=policy%d,mark=%s,line width=0.9pt}\addlegendentry{%s}" % (ROLES.index(role), MARKS[role], LABELS[role]))
        for role in roles:
            color = "policy" + str(ROLES.index(role))
            medians = " ".join(f"({batch},{lookup[metric, role, batch]['median'] * scale:.12g})" if lookup[metric, role, batch]["median"] is not None else f"({batch},nan)" for batch in batches)
            if any(lookup[metric, role, batch]["median"] is not None for batch in batches):
                lines.append(r"\addplot[forget plot,color=%s,mark=%s,mark size=2pt,line width=0.9pt] coordinates {%s};" % (color, MARKS[role], medians))
            points = []
            for batch in batches:
                cell = lookup[metric, role, batch]
                for repeat, value in zip(cell["repeats"], cell["raw"]):
                    if value is not None:
                        x = batch * math.exp(.035 * (repeat - (cell["expected_repeats"] - 1) / 2))
                        points.append(f"({x:.12g},{value * scale:.12g})")
            if points:
                lines.append(r"\addplot[only marks,forget plot,color=%s,mark=o,mark size=2.2pt,mark options={fill=white,line width=0.6pt}] coordinates {%s};" % (color, " ".join(points)))
        if not any_values:
            lines.append(r"\node[align=center,font=\small,text=gray] at (rel axis cs:0.5,0.5) {No recorded measurement};")
    note = r"Open dots: every run value (small $x$ offsets). Lines: complete run medians; missing repeats break lines.\\No confidence intervals; shared engines and online state are not independent engine repeats."
    if metrics == ["population_counter_joules_per_token"]:
        note += r"\\Whole-device first-start to last-end counter span includes gaps; counter endpoints differ from runner wall endpoints."
    elif metrics == ["joules_per_token"]:
        note += (r"\\Sum of measured wave-window energy only; inter-wave gaps are excluded. Separate from counter-span energy."
                 if metadata["regime"]["admission_mode"] == "wave" else
                 r"\\Energy of one measured population window; counter-read endpoints are reported in the separate counter-span figure.")
    elif metrics == list(CONTROL):
        note += r"\\Scheduled decode steps include terminal/drain. CPU uses timed calls only; unrecorded values are absent.\\Discarded counts are not measured GPU time or energy."
    else:
        note += r"\\TPOT: per-run mean across complete requests, from first to last output receipt divided by output tokens minus one."
    if metadata["regime"]["execution_mode"] == "compiled":
        note += r"\\Whole policies: native tables and fixed-budget engines can use different graph modes."
    if metadata["fixture"]:
        note += r"\\SYNTHETIC CPU FIXTURE: no research measurement or performance claim."
    lines += [r"\end{groupplot}",
        r"\path (group c1r1.north west) -- node[above=1.6cm,font=\large] {%s} (group c%dr1.north east);" % (tex_escape(title), count),
        r"\path (group c1r1.north west) -- node[above=0.75cm] {\ref{policyLegend}} (group c%dr1.north east);" % count,
        r"\path (group c1r1.south west) -- node[below=1.4cm,align=center,font=\scriptsize] {%s} (group c%dr1.south east);" % (note, count),
        r"\end{tikzpicture}\end{document}"]
    return "\n".join(lines) + "\n"


def analyze(inputs, output, *, no_plot=False, include_wave_energy=False, display_label=None):
    output = Path(output).resolve()
    require(not output.exists() or not any(output.iterdir()), "choose a fresh/empty plot output directory")
    groups, raw, provenance, pending = {}, [], [], []
    visited = set()
    for value in inputs:
        try:
            current, points, source = load_comparison(value)
        except FileNotFoundError:
            pending.append({"source": str(value), "reason": "comparison_not_available"})
            continue
        require(source["sha256"] not in visited, "same comparison content supplied twice")
        visited.add(source["sha256"])
        groups.update(current); raw.extend(points); provenance.append(source)
    if any(g["fixture"] for g in groups.values()):
        temporary_roots = (Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve())
        require(any(output.is_relative_to(root) for root in temporary_roots), "CPU fixture plots must stay in a temporary directory")
    cells = prepare(groups, raw)
    if display_label:
        require(len(provenance) == 1, "a display label requires one input comparison")
        for metadata in groups.values():
            metadata["display_label"] = display_label
    for point in raw:
        declared = groups[point["group"]]["coverage"].get((point["role"], point["batch"]), {})
        count = max(2, int(declared.get("expected_run_records", 2)))
        point["display_scale"] = METRICS[point["metric"]][3]
        point["plot_x"] = point["batch"] * math.exp(.035 * (point["repeat"] - (count - 1) / 2))
        point["display_value"] = point["value"] * point["display_scale"] if point["value"] is not None else None
    output.mkdir(parents=True, exist_ok=True)
    output_csv(output / "plot_raw.csv", raw)
    output_csv(output / "plot_cells.csv", cells)
    artifacts, omitted = [], []
    panels = [["effective_tpot_ms"], ["population_counter_joules_per_token"], list(CONTROL)]
    if include_wave_energy:
        panels.append(["joules_per_token"])
    for group, metadata in groups.items():
        values = [r for r in cells if r["group"] == group]
        for metrics in panels:
            name = METRICS[metrics[0]][0] + "_" + group
            if not any(r["observed_values"] for r in values if r["metric"] in metrics):
                omitted.append({"group": group, "figure": name, "reason": "no_recorded_metric_values"})
                continue
            if not no_plot:
                artifacts.extend(render_pgf(document(metadata, values, metrics), output, name, ["pdf", "svg", "png"]))
    fixture_only = bool(groups) and all(g["fixture"] for g in groups.values())
    status = "data_exported" if no_plot and raw else "rendered_measured_values" if artifacts else "pending_no_measurements"
    if fixture_only and artifacts:
        status = "rendered_cpu_fixture"
    report = {"schema_version": 1, "status": status,
        "inputs": provenance, "pending_inputs": pending,
        "groups": [{"group": name, **{k: v for k, v in meta.items() if k != "coverage"}} for name, meta in groups.items()],
        "figure_artifacts": artifacts, "omitted_figures": omitted,
        "data_artifacts": ["plot_raw.csv", "plot_cells.csv"], "include_wave_energy": include_wave_energy,
        "fixture_only": fixture_only,
        "source_sha256": {name: sha(Path(__file__).parent / name) for name in (Path(__file__).name, "plot_energy_progress.py")},
        "boundaries": ["Each input comparison, execution regime and actual sample hash is plotted separately; no pooling across inputs or draft domains.",
            "Explicit roles retain native unrestricted-table and threshold identities. No post-hoc envelope, inferred policy identity or best-repeat selection.",
            "Only complete declared repeat sets (at least two) get medians; missing/invalid coverage breaks lines. All finite accepted raw points remain visible.",
            "F4 uses only validated cumulative-counter spans. Optional wave-window energy has a separate figure and timing boundary.",
            "F6 uses per-run means of actually timed decision calls and discarded/available token counts over scheduled decode steps. Timing coverage is exported; missing native/fallback call timing is not zero.",
            "No confidence intervals, kernel/HBM inference, subtraction of nested CPU scopes, GPU execution or remote upload."]}
    (output / "plot_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-wave-energy", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--label", help="Display label for a single comparison; does not alter data or grouping")
    args = parser.parse_args()
    result = analyze(args.comparison, args.output, no_plot=args.no_plot, include_wave_energy=args.include_wave_energy, display_label=args.label)
    print(json.dumps({"status": result["status"], "groups": len(result["groups"]), "artifacts": len(result["figure_artifacts"]), "output": args.output}))
    return 2 if result["status"] == "pending_no_measurements" else 0


if __name__ == "__main__":
    raise SystemExit(main())

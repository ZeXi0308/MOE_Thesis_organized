#!/usr/bin/env python3
"""Summarize retained scan/bracket measurements; no fitting or independent-step inference."""
import argparse
import json
from pathlib import Path


def span(values):
    values = list(values)
    return dict(n=len(values), min=min(values) if values else None, max=max(values) if values else None)


def load(path):
    return json.loads(path.read_text())


def summarize(root):
    result = dict(evidence_type="CUSTOM_CONTINUOUS_RUNTIME", scope="finite-episode static cap measurements",
        claim_ceiling="No U/C incremental signal, online selector, hardware congestion, or native-serving claim.",
        sampling_note="The same 16 source requests are replayed across cells. Repeats and adjacent decode steps do not create independent request/document samples.",
        scans={}, bracket_cap6_comparisons=[], same_cap_trajectories=[], off_on_pairs=[])
    all_cells = []
    for name in ("scan", "bracket"):
        analysis, config = load(root / f"{name}_analysis/analysis.json"), load(root / name / "config.json")
        if not analysis["complete_scan"] or any(c["state"] != "COMPLETE" for c in analysis["cells"]):
            raise ValueError(f"{name}: complete valid scan required for this summary")
        cells = analysis["cells"]
        for c in cells:
            c.update(bundle=name, label=f"{name}/cell-{c['cell']:03d}", raw=load(root / name / f"cell-{c['cell']:03d}.json"))
        all_cells.extend(cells)
        off = [c for c in cells if not c["plan"]["telemetry"]]
        result["scans"][name] = dict(source_analysis=f"{name}_analysis/analysis.json",
            requests_per_cell=config["requests"], ttft_slo_s=config["ttft_slo_s"], tpot_slo_s=config["tpot_slo_s"],
            off_by_cap={str(cap): dict(goodput_rps=span(c["metrics"]["goodput_rps"] for c in off if c["plan"]["cap"] == cap),
                joint_pass_count=span(c["metrics"]["n_slo_pass"] for c in off if c["plan"]["cap"] == cap))
                for cap in sorted({c["plan"]["cap"] for c in off})})
        if name == "bracket":
            lookup = {(c["plan"]["repeat"], c["plan"]["regime"], c["plan"]["cap"]): c for c in off}
            for (repeat, regime, cap), six in sorted(lookup.items()):
                if cap != 6:
                    continue
                for base_cap in (4, 8):
                    base = lookup[repeat, regime, base_cap]
                    a, b = six["metrics"], base["metrics"]
                    result["bracket_cap6_comparisons"].append(dict(repeat=repeat, regime=regime, baseline_cap=base_cap,
                        cap6_cell=six["label"], baseline_cell=base["label"],
                        goodput_delta_rps=a["goodput_rps"] - b["goodput_rps"],
                        goodput_relative_pct=100 * (a["goodput_rps"] / b["goodput_rps"] - 1),
                        joint_pass_delta=a["n_slo_pass"] - b["n_slo_pass"]))
        by_id = {c["cell"]: c for c in cells}
        for pair in analysis["telemetry_pairs"]:
            a, b = (by_id[pair[key]]["metrics"] for key in ("off_cell", "on_cell"))
            result["off_on_pairs"].append(dict(bundle=name, repeat=pair["repeat"], regime=pair["regime"], cap=pair["cap"],
                on_minus_off_wall_s=b["observation_duration_s"] - a["observation_duration_s"],
                on_minus_off_wall_relative_pct=100 * (b["observation_duration_s"] / a["observation_duration_s"] - 1)))
    def membership(c):
        return [(s["request_ids"], s["decode_steps"]) for s in c["raw"]["steps"]]
    def outputs(c):
        return sorted((r["request_id"], r["output_token_ids"]) for r in c["raw"]["requests"])
    def pressure(c):
        return [(s["request_ids"], s["decode_steps"], s["pressure"]) for s in c["raw"]["steps"]]
    for cap in sorted({c["plan"]["cap"] for c in all_cells}):
        cells = [c for c in all_cells if c["plan"]["cap"] == cap]
        on = [c for c in cells if c["plan"]["telemetry"]]
        checks = dict(cap=cap, n_cells=len(cells), n_on_cells=len(on), reference_cell=cells[0]["label"], on_reference_cell=on[0]["label"])
        for key, group, signature in (("membership", cells, membership), ("output", cells, outputs), ("on_pressure", on, pressure)):
            differing = [c["label"] for c in group[1:] if signature(c) != signature(group[0])]
            checks.update({f"{key}_all_equal": not differing, f"{key}_differing_cells": differing})
        result["same_cap_trajectories"].append(checks)
    result["off_on_wall_relative_pct"] = {name: span(p["on_minus_off_wall_relative_pct"] for p in result["off_on_pairs"]
        if name == "all" or p["bundle"] == name) for name in ("scan", "bracket", "all")}
    result["telemetry_helper"] = {"interpretation": "Helper interval already included in episode wall time; excludes router-logit return/materialization. OFF/ON differences are rerun diagnostics, not isolated causal overhead."}
    for on in (False, True):
        cells = [c for c in all_cells if c["plan"]["telemetry"] == on]
        costs = [c["execution_exposure"]["telemetry_helper"] for c in cells]
        result["telemetry_helper"]["on" if on else "off"] = dict(total_s=span(x["total_s"] for x in costs),
            episode_wall_pct=span(100 * x["episode_wall_fraction"] for x in costs))
    result["unique_request_ids"] = len({r["request_id"] for c in all_cells for r in c["raw"]["requests"]})
    result["unique_document_ids"] = len({r["document_id"] for c in all_cells for r in c["raw"]["requests"]})
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "measurements.json")
    args = parser.parse_args()
    with args.output.open("x") as output:
        json.dump(summarize(root), output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(args.output)

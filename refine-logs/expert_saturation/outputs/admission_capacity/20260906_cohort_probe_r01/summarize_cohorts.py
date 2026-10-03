"""Recompute the three-cohort comparison; preserve per-run outcomes and limits."""
import json
from pathlib import Path
from statistics import mean, median
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "refine-logs/expert_saturation/experiments/admission_capacity"))
from analyze_capacity import analyze


def pressure_window(raw, cap):
    first_completion = min(r["completion_s"] for r in raw["requests"]
                           if r["completion_s"] is not None)
    window = []
    for step in raw["steps"]:
        ordinary = step["ordinary"]
        eligible = (ordinary["decode_requests"] == cap
                    and ordinary["prefill_requests_this_iteration"] == 0
                    and step["token_emission_s"] < first_completion)
        if eligible:
            window.append(step)
        elif window:
            break
    if not window:
        return dict(status="NO_ELIGIBLE_WINDOW")
    pressure = [p for s in window for p in s["pressure"]]
    first = window[0]
    return dict(status="DESCRIPTIVE", steps=[s["step"] for s in window],
                U_mean=mean(p["U"] for p in pressure), C_mean=mean(p["C"] for p in pressure),
                layer_U_means=[mean(s["pressure"][i]["U"] for s in window) for i in range(16)],
                layer_C_means=[mean(s["pressure"][i]["C"] for s in window) for i in range(16)],
                model_call_median_s=median(s["model_call_s"] for s in window),
                entry_ordinary=first["ordinary"],
                ordinary_shapes=[dict(width=s["ordinary"]["decode_requests"],
                                      kv_lengths=s["ordinary"]["kv_lengths"],
                                      waiting=s["ordinary"]["waiting_requests"],
                                      future=s["ordinary"]["future_requests"]) for s in window])


def main():
    runs = {0: HERE.parent / "20260906_gpu_pilot_r01/bracket",
            16: HERE / "gpu_results/offset16", 32: HERE / "gpu_results/offset32"}
    output = dict(evidence_type="CUSTOM_CONTINUOUS_RUNTIME_EXPLORATORY",
                  interpretation="Static response only; no pre-action residual, dynamic Oracle or controller result.",
                  cohorts={}, comparisons=[])
    for offset, directory in runs.items():
        a = analyze(directory)
        if not a["complete_scan"] or any(c["issues"] for c in a["cells"]):
            raise RuntimeError(f"cohort {offset} not complete/valid: {a['verdict']}")
        cells = []
        for cell in a["cells"]:
            if cell["plan"]["cap"] not in (6, 8):
                continue
            m = cell["metrics"]
            row = dict(cell=cell["cell"], **cell["plan"],
                       **{k: m[k] for k in ("goodput_rps", "throughput_rps", "slo_attainment",
                                          "n_completed", "n_failed", "n_unfinished", "n_slo_pass")},
                       ttft_p50_s=m["latency_s"]["ttft"]["p50"],
                       tpot_p50_s=m["latency_s"]["tpot"]["p50"],
                       wall_s=m["observation_duration_s"])
            if row["telemetry"]:
                raw = json.loads((directory / f"cell-{cell['cell']:03d}.json").read_text())
                row["first_full_no_prefill_window"] = pressure_window(raw, row["cap"])
            cells.append(row)
        output["cohorts"][offset] = dict(run=str(directory.relative_to(ROOT)), cells=cells,
                                        complete_scan=True, analyzed_cells=len(a["cells"]))
        for telemetry in (False, True):
            for regime in ("steady", "bursty"):
                for repeat in (0, 1):
                    pair = {c["cap"]: c for c in cells if
                            (c["telemetry"], c["regime"], c["repeat"]) == (telemetry, regime, repeat)}
                    delta = pair[8]["goodput_rps"] - pair[6]["goodput_rps"]
                    output["comparisons"].append(dict(offset=offset, telemetry=telemetry,
                        regime=regime, repeat=repeat, cap8_minus_cap6_goodput=delta,
                        preferred_static_cap=8 if delta > 0 else 6,
                        cap6_goodput=pair[6]["goodput_rps"], cap8_goodput=pair[8]["goodput_rps"]))
    with (HERE / "summary.json").open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")
    for offset, cohort in output["cohorts"].items():
        for cap in (6, 8):
            rows = [r for r in cohort["cells"] if r["cap"] == cap and not r["telemetry"]]
            print(offset, cap, "goodput", [round(r["goodput_rps"], 4) for r in rows],
                  "slo_pass", [r["n_slo_pass"] for r in rows])
    print("OFF comparisons", [r["preferred_static_cap"] for r in output["comparisons"] if not r["telemetry"]])


if __name__ == "__main__":
    main()

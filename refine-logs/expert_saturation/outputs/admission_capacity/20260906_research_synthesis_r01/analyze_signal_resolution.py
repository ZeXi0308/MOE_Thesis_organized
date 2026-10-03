#!/usr/bin/env python3
"""Describe retained U/C resolution, not a new runtime or performance experiment."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "20260906_step_action_r01/analysis.json"


def main():
    saved = json.loads(SOURCE.read_text())
    experts, top_k, width, layers = 64, 8, 6, 16
    representatives, summaries = {}, {}
    for cohort in ("c0", "c32"):
        cells = [c for c in saved["cells"] if c["cohort"] == cohort and c["telemetry"]]
        assert len(cells) == 8
        pressure = cells[0]["pressure_layers"]
        assert len(pressure) == layers
        assert all(c["pressure_layers"] == pressure for c in cells)
        assert all(c["ordinary"]["decode_batch_width"] == width for c in cells)
        active_counts, maxima = [], []
        for row in pressure:
            assert row["routed_tokens"] == width * top_k
            active = round(row["U"] * experts)
            assert abs(row["U"] * experts - active) < 1e-6
            assert abs(row["C"] - experts * row["max_expert_tokens"] / row["routed_tokens"]) < 1e-5
            active_counts.append(active)
            maxima.append(row["max_expert_tokens"])
        representatives[cohort] = (active_counts, maxima)
        summaries[cohort] = dict(ON_cells=8, identical_latest_pressure=True,
            active_experts_per_layer=active_counts, max_expert_tokens_per_layer=maxima,
            sum_active_experts=sum(active_counts), sum_max_expert_tokens=sum(maxima),
            mean_U=sum(active_counts) / (experts * layers),
            mean_C=experts * sum(maxima) / (width * top_k * layers))
    u_delta = [a-b for a, b in zip(representatives["c0"][0], representatives["c32"][0])]
    max_delta = [a-b for a, b in zip(representatives["c0"][1], representatives["c32"][1])]
    output = dict(source=str(SOURCE.relative_to(HERE.parents[5])),
        evidence_type="STRUCTURAL_REANALYSIS_OF_RETAINED_GPU_DATA",
        new_GPU_runs=0, fitted_predictors=0,
        model_assumptions=dict(experts=experts, top_k=top_k, tokens_per_layer=width, layers=layers,
                               distinct_experts_per_token=True),
        cohorts=summaries,
        c0_minus_c32=dict(active_count_delta_per_layer=u_delta, max_count_delta_per_layer=max_delta,
            net_active_count_delta=sum(u_delta), absolute_active_count_delta=sum(map(abs, u_delta)),
            net_max_count_delta=sum(max_delta), absolute_max_count_delta=sum(map(abs, max_delta)),
            layers_with_different_max=sum(x != 0 for x in max_delta)),
        arithmetic=dict(single_layer_C_increment=experts / (width * top_k),
                        layer_mean_C_increment=experts / (width * top_k * layers)),
        interpretation="Net aggregate differences can cancel layerwise differences; two cohorts do not establish or refute conditional action value. No inference about HBM or congestion.")
    destination = HERE / "signal_resolution.json"
    with destination.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(output["c0_minus_c32"], indent=2))


if __name__ == "__main__":
    main()

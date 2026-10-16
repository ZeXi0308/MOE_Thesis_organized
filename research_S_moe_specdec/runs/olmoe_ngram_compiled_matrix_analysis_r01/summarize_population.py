#!/usr/bin/env python3
"""Add full-population boundaries and descriptive numbers to measured F2 output.

Run from project root. Reuses existing cumulative-counter validation and keeps
all two original perf values. No fit, resampling, GPU execution or uploads.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "code"))
from compare_controller_policies import population_counter_energy
from plot_results import write_csv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    campaign = ROOT / "runs/olmoe_ngram_compiled_r01"
    status = json.loads((campaign / "status.json").read_text())
    assert status["status"] == "complete"
    assert sorted(c["gamma"] for c in status["cells"] if c.get("returncode") == 0) == list(range(8))
    metadata = json.loads((OUT / "analysis_metadata.json").read_text())
    assert metadata["run_count"] == 112 and metadata["excluded_count"] == 0
    raw, original = read_csv(OUT / "raw_runs.csv"), read_csv(OUT / "summary.csv")
    assert len(raw) == 112 and len(original) == 56
    values, identities = [], set()
    for row in raw:
        source = Path(row["source"])
        record = json.loads(source.read_text())
        assert record["all_finished"] and record["output_tokens"] == 8192
        population, wave = record["population_wall_s"], record["wall_s"]
        assert population >= wave > 0
        assert math.isclose(record["population_tokens_per_second"], 8192 / population, rel_tol=1e-12)
        assert math.isclose(record["population_inter_episode_gap_s"], population - wave, abs_tol=1e-9)
        requests = [q for episode in record["episodes"] for q in episode["requests"]]
        assert len(requests) == 64 and all(len(q["output_token_ids"]) == 128 for q in requests)
        identities.add(tuple(sorted(q["source_index"] for q in requests)))
        value = {"batch": int(row["batch"]), "gamma": int(row["gamma"]), "repeat": int(row["repeat"]),
            "source": str(source), "source_sha256": sha(source), "execution_mode": row["execution_mode"],
            "output_tokens": 8192, "request_count": 64,
            "wave_window_wall_s": wave, "wave_window_tokens_per_second": record["tokens_per_second"],
            "effective_tpot_ms": float(row["effective_tpot_ms"]),
            "wave_window_energy_j": record["energy_j"], "wave_window_joules_per_token": record["joules_per_token"],
            "population_wall_s": population, "population_tokens_per_second": record["population_tokens_per_second"],
            "population_inter_episode_gap_s": population - wave,
            "population_gap_fraction": (population - wave) / population,
            "population_gap_boundary": record["population_inter_episode_gap_boundary"],
            **population_counter_energy(record)}
        values.append(value)
    assert len(identities) == 1 and len(next(iter(identities))) == 64
    values.sort(key=lambda r: (r["batch"], r["gamma"], r["repeat"]))
    grouped = defaultdict(list)
    for row in values:
        grouped[row["batch"], row["gamma"]].append(row)
    assert set(grouped) == {(b, g) for b in (1, 2, 4, 8, 16, 32, 64) for g in range(8)}
    assert all([r["repeat"] for r in rows] == [0, 1] for rows in grouped.values())
    metrics = ("wave_window_tokens_per_second", "effective_tpot_ms", "wave_window_joules_per_token",
        "population_wall_s", "population_tokens_per_second", "population_inter_episode_gap_s", "population_gap_fraction",
        "population_counter_energy_j", "population_counter_joules_per_token", "population_counter_elapsed_s")
    cells = []
    for (batch, gamma), rows in sorted(grouped.items()):
        cell = {"batch": batch, "gamma": gamma, "n_runs": 2, "sources": [r["source"] for r in rows]}
        for metric in metrics:
            observed = [r.get(metric) for r in rows]
            cell[metric + "_raw"] = observed
            cell[metric] = statistics.median(observed) if all(v is not None for v in observed) else None
        original_cell = next(r for r in original if int(r["batch"]) == batch and int(r["gamma"]) == gamma)
        cell["wave_window_speedup"] = float(original_cell["speedup"])
        cell["tpot_speedup"] = float(original_cell["tpot_speedup"])
        cells.append(cell)
    for cell in cells:
        base = next(r for r in cells if r["batch"] == cell["batch"] and r["gamma"] == 0)
        for name, metric in (("population_speedup", "population_tokens_per_second"),
                ("wave_window_energy_ratio_to_ar", "wave_window_joules_per_token"),
                ("population_counter_energy_ratio_to_ar", "population_counter_joules_per_token")):
            cell[name] = cell[metric] / base[metric] if cell[metric] is not None and base[metric] else None
        for metric in ("wave_window_tokens_per_second", "effective_tpot_ms", "wave_window_joules_per_token",
                       "population_tokens_per_second", "population_counter_joules_per_token"):
            cell["ar_" + metric + "_raw"] = base[metric + "_raw"]
    ranges = []
    for batch in (1, 2, 4, 8, 16, 32, 64):
        selected = [r for r in cells if r["batch"] == batch and r["gamma"] > 0]
        row = {"batch": batch}
        for metric in ("wave_window_speedup", "tpot_speedup", "wave_window_energy_ratio_to_ar",
                       "population_speedup", "population_counter_energy_ratio_to_ar"):
            pairs = [(r[metric], r["gamma"]) for r in selected if r[metric] is not None]
            row[metric + "_min"], row[metric + "_min_gamma"] = min(pairs) if pairs else (None, None)
            row[metric + "_max"], row[metric + "_max_gamma"] = max(pairs) if pairs else (None, None)
        row["population_gap_fraction_raw_range_all_gamma"] = [min(r["population_gap_fraction"] for r in values if r["batch"] == batch),
                                                              max(r["population_gap_fraction"] for r in values if r["batch"] == batch)]
        ranges.append(row)
    gains = {metric: [r for r in cells if r["gamma"] > 0 and r[metric] is not None and r[metric] > 1]
             for metric in ("wave_window_speedup", "tpot_speedup", "population_speedup")}
    saving = {metric: [r for r in cells if r["gamma"] > 0 and r[metric] is not None and r[metric] < 1]
              for metric in ("wave_window_energy_ratio_to_ar", "population_counter_energy_ratio_to_ar")}
    for name, rows in (("full_population_raw.csv", values), ("cell_summary_with_population.csv", cells),
                       ("per_batch_ranges.csv", ranges), ("above_ar_tpot_cells_with_raw.csv", gains["tpot_speedup"])):
        write_csv(OUT / name, rows)
    brief = {"status": "complete_measured_matrix", "run_count": 112, "cell_count": 56, "excluded_count": 0,
        "campaign_status_sha256": sha(campaign / "status.json"), "script_sha256": sha(Path(__file__)),
        "gpu_lock_seconds": status["gpu_lock_seconds"], "per_batch_gamma1to7_ranges": ranges,
        "counter_status_counts": dict(Counter(r["population_counter_status"] for r in values)),
        "above_ar_cells": {k: [{"batch": r["batch"], "gamma": r["gamma"], "ratio": r[k]} for r in v] for k, v in gains.items()},
        "energy_below_ar_cells": {k: [{"batch": r["batch"], "gamma": r["gamma"], "ratio": r[k]} for r in v] for k, v in saving.items()},
        "boundaries": [
            "F2 throughput and energy sum wave measurement windows, including their prefill/setup/drain, but exclude between-window handling gaps.",
            "Population wall/throughput are runner full-population measurements; their difference from summed windows includes metering and Python handling around/between episodes, not solely GPU idle time.",
            "Counter-span energy uses original first-start to final-end NVML cumulative reads, with validated UUID/monotonicity. It includes gaps but its counter-read endpoints are not exactly runner population-wall endpoints.",
            "Effective TPOT is the per-run request mean of first-to-last output receipt intervals per generated-token interval; excludes each request's prefill.",
            "Ratios use medians of two original run records. Raw repeat0/1 values and same-B AR values are retained; no confidence intervals or statistical significance are inferred.",
            "Prompt IDs match across all 112 runs. Initial full-batch warmup and all shape/prompt warmups precede measurement; historical general warmup offset80 is retained. No cold-start or prompt-unseen claim.",
            "enforce_eager=False establishes the compiled configuration, not per-step FULL/PIECEWISE/eager dispatch fractions. Performance has routed-expert return and route capture disabled.",
            "Direct native gamma0..7 engines execute in campaign order, with within-engine forward/reverse B repeats. Differences across execution windows are not isolated causal effects of graph mode."]}
    (OUT / "numerical_brief.json").write_text(json.dumps(brief, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"run_count": 112, "cell_count": 56, "counter_status_counts": brief["counter_status_counts"],
        "above_ar_counts": {k:len(v) for k,v in gains.items()}, "energy_below_ar_counts": {k:len(v) for k,v in saving.items()},
        "ranges": ranges}, allow_nan=False))


if __name__ == "__main__":
    main()

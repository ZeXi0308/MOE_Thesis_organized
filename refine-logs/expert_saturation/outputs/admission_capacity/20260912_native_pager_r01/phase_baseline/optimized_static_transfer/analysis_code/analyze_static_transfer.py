"""Export a safe check of frozen static-cost transfer to new documents."""
import argparse, json, statistics
from pathlib import Path
from functools import partial
from analyze_static_calibration import analyze as analyze_static, error, sha
PREDICTED = {"first_step_s": "first_action_step_s",
             "first_old_itl_s": "max_old_cross_action_itl_s"}
METRICS = ("wall_s", "old_done_s", "old_max_itl_s", "new_ttft_s",
           "copy_payload_bytes", "engine_calls", *PREDICTED.values())
def delta(first, second):
    value = second - first
    return dict(signed=value, absolute=abs(value),
                relative=value / first if first else None)
def transfer_row(base, config, execution, raws, cells, source_analysis_path=None, source_summary_path=None):
    path = base / "frozen_transfer_prediction.json"
    complete = len(raws) == 6
    if not path.exists():
        return dict(status="INVALID_PROTOCOL" if complete else "UNRUN_OR_MISSING"), complete
    data, frozen = path.read_bytes(), json.loads(path.read_text())
    source = (base / frozen["source_reference"]).resolve()
    analysis_path = source_analysis_path or source / "safe_readback/analysis.json"
    summary_path = source_summary_path or source / "summary.json"
    source_analysis, source_summary = json.loads(analysis_path.read_text()), json.loads(summary_path.read_text())
    prior = source_analysis["midpoint"]
    expected = {"8": prior["endpoint_means"]["8"], "16": prior["prediction"],
                "32": prior["endpoint_means"]["32"]}
    starts = {row["cell"]: row.get("start_unix") for row in execution.get("cells", [])}
    actual, by_cap = [], {}
    for cell in config["cells"]:
        name, cap, raw = cell["cell"], cell["chunk"], raws.get(cell["cell"])
        if raw is None:
            actual.append(dict(cell=name, cap=cap, status="UNRUN_OR_MISSING"))
            continue
        observed = {key: cells[name]["metrics"][metric] for key, metric in PREDICTED.items()}
        predicted = frozen["predictions"][str(cap)]
        action_start = raw["measurement_origin_unix_s"] + raw["action"]["start_s"]
        row = dict(cell=name, cap=cap, repeat=cell["repeat"],
            frozen_before_cell_start=frozen["created_unix"] < starts.get(name, float("-inf")),
            frozen_before_action_start=frozen["created_unix"] < action_start,
            observed=observed, predicted=predicted,
            errors={key: error(observed[key], predicted[key]) for key in PREDICTED})
        actual.append(row)
        by_cap.setdefault(cap, []).append(name)
    repeats, means = {}, {key: {} for key in PREDICTED}
    for cap, names in by_cap.items():
        ordered = sorted(names, key=lambda name: cells[name]["cell_config"]["repeat"])
        if len(ordered) != 2:
            continue
        first, second = (cells[name]["metrics"] for name in ordered)
        repeats[str(cap)] = dict(cells=ordered,
            r1_minus_r0={key: delta(first[key], second[key]) for key in METRICS})
        for key, metric in PREDICTED.items():
            means[key][str(cap)] = statistics.mean(cells[name]["metrics"][metric] for name in ordered)
    rank_order = {}
    for key in PREDICTED:
        predicted_order = sorted((8, 16, 32), key=lambda cap: frozen["predictions"][str(cap)][key])
        actual_order = (sorted((8, 16, 32), key=lambda cap: means[key][str(cap)])
                        if len(means[key]) == 3 else None)
        rank_order[key] = dict(predicted_low_to_high=predicted_order,
            actual_repeat_mean_low_to_high=actual_order,
            matches_frozen_order=actual_order == predicted_order if actual_order else None,
            semantics="Ordering of two-repeat means; descriptive only.")
    checks = dict(frozen_status=frozen.get("status") == "FROZEN_BEFORE_TRANSFER_GPU",
        config_hash_matches=config.get("frozen_prediction_sha256") == sha(data),
        source_analysis_hash=sha(analysis_path.read_bytes()) == frozen.get("source_analysis_sha256"),
        source_summary_hash=sha(summary_path.read_bytes()) == frozen.get("source_summary_sha256"),
        source_reports_agree=prior == source_summary.get("midpoint"),
        source_prestate_matches=source_analysis["execution"]["reference_prestate_sha256"] == frozen.get("source_prestate_sha256"),
        predictions_match_frozen_source=frozen.get("predictions") == expected,
        embedded_copy_matches=execution.get("transfer_prediction") == frozen,
        frozen_before_execution=frozen["created_unix"] < execution.get("start_unix", float("-inf")),
        frozen_before_all_cells=len(actual) == 6 and all(row.get("frozen_before_cell_start") for row in actual),
        frozen_before_all_actions=len(actual) == 6 and all(row.get("frozen_before_action_start") for row in actual))
    valid = complete and all(checks.values()) and len(by_cap) == 3 and all(len(rows) == 2 for rows in by_cap.values())
    return dict(status="VALIDATED" if valid else ("INVALID_PROTOCOL" if complete else "INCOMPLETE"),
        file_sha256=sha(data), created_unix=frozen["created_unix"], source_reference=frozen["source_reference"],
        source_hashes={key: frozen[key] for key in ("source_analysis_sha256", "source_summary_sha256", "source_prestate_sha256")},
        predictions=frozen["predictions"], scope=frozen.get("scope"), checks=checks,
        cells=actual, repeat_changes=repeats, actual_repeat_means=means, rank_order=rank_order), complete and not valid
def analyze(base, source_analysis=None, source_summary=None):
    out = analyze_static(base, partial(transfer_row, source_analysis_path=source_analysis, source_summary_path=source_summary))
    out["schema"] = "optimized-static-transfer-safe-v1"
    out["artifact_sha256"].pop("midpoint_prediction.json", None)
    out["artifact_sha256"]["frozen_transfer_prediction.json"] = sha((base / "frozen_transfer_prediction.json").read_bytes())
    out["transfer_prediction"] = out.pop("midpoint")
    return out
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-analysis", type=Path)
    parser.add_argument("--source-summary", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(analyze(args.results, args.source_analysis, args.source_summary), indent=2, allow_nan=False) + "\n"
    if any(f'"{key}"' in encoded for key in
           ("prompt_token_ids", "output_token_ids", "token_received_s", "requests", "steps")):
        raise RuntimeError("unsafe raw field in safe export")
    with args.out.open("x") as handle:
        handle.write(encoded)
if __name__ == "__main__":
    main()

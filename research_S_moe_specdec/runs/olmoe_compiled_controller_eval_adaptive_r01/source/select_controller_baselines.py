#!/usr/bin/env python3
"""Select a fixed gamma and two executable native batch tables from calibration.

  python select_controller_baselines.py --performance runs/CALIBRATION \
      --cost-calibration runs/CALIBRATION/calibration/calibration.json \
      --output runs/controller_selection_01
  python select_controller_baselines.py --output /tmp/no_data_selection
  python select_controller_baselines.py --self-test

Each execution/resource/workload regime gets separate artifacts. The selection
objective is measured request-mean effective TPOT, with explicit batch weights.
CalibrationTable's complete-step time alone is insufficient for that objective.
An objective-enriched CalibrationTable may be used only when its metadata has
objective_metric='measured_request_mean_effective_tpot_ms' and every objective
row identifies a source/run and request_count. Plain timing tables are retained
as cost-coverage provenance; missing objective data produces a waiting status.

Native-table and threshold scores combine static calibration cell estimates for
CHOICE ONLY. They are not actual performance of those policies. The emitted
tables must be run, unchanged, on separate evaluation runs. No future-informed
step splicing or post-hoc evaluation-set choice is performed by this script.

Writes a fresh output directory; existing outputs are not overwritten.
Standard library only; --self-test uses synthetic fixtures, never GPU execution.
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
import unittest

from cost_model import CalibrationTable
from plot_results import REGIME_FIELDS, _finite, load_runs, resolve_regime


METRIC = "measured_request_mean_effective_tpot_ms"
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "controller_protocol.json"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def validate_protocol(obj: dict) -> dict:
    batches = [int(b) for b in obj["batches"]]
    gammas = [int(g) for g in obj["candidate_gammas"]]
    if batches != sorted(set(batches)) or not batches or batches[0] != 1:
        raise ValueError("protocol batches must be sorted, unique, positive and start at 1")
    if gammas != sorted(set(gammas)) or not gammas or gammas[0] != 0:
        raise ValueError("candidate_gammas must be sorted, unique, nonnegative and include 0")
    objective = obj["objective"]
    if objective.get("metric") != METRIC or objective.get("direction") != "minimize":
        raise ValueError("only the explicit measured effective-TPOT minimization objective is supported")
    if objective.get("run_aggregation") not in ("median", "mean"):
        raise ValueError("run_aggregation must be median or mean")
    weights = {int(k): float(v) for k, v in objective["batch_weights"].items()}
    if set(weights) != set(batches) or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("positive finite explicit weights are required for every batch")
    maximum = int(obj["selection"]["native_engine_max_gamma"])
    if maximum < max(gammas) or maximum <= 0:
        raise ValueError("native engine maximum must be positive and cover every candidate gamma")
    thresholds = [int(v) for v in obj["selection"]["threshold_candidates"]]
    if thresholds != sorted(set(thresholds)) or not thresholds or thresholds[0] != 0:
        raise ValueError("threshold candidates must be sorted/unique and include always-disabled t=0")
    if any(t < 0 or t > max(batches) for t in thresholds):
        raise ValueError("threshold outside declared batch domain")
    if not obj["selection"].get("require_complete_candidate_grid", False):
        raise ValueError("this selector requires the complete declared candidate grid; narrow the protocol explicitly if needed")
    total = sum(weights.values())
    return {"batches": batches, "gammas": gammas,
            "weights": {b: w / total for b, w in weights.items()},
            "thresholds": thresholds, "engine_max_gamma": maximum,
            "aggregate": statistics.median if objective["run_aggregation"] == "median" else statistics.mean}


def native_ranges(mapping: dict[int, int], maximum_batch: int) -> list[list[int]]:
    """Explicit contiguous lower-neighbor ranges, allowing 0 and nonmonotonic K."""
    batches = sorted(mapping)
    if not batches or batches[0] != 1 or maximum_batch < batches[-1]:
        raise ValueError("native table must start at B=1 and cover the declared domain")
    result = []
    for i, batch in enumerate(batches):
        end = batches[i + 1] - 1 if i + 1 < len(batches) else maximum_batch
        gamma = int(mapping[batch])
        if gamma < 0:
            raise ValueError("native gamma cannot be negative")
        if result and result[-1][2] == gamma and result[-1][1] + 1 == batch:
            result[-1][1] = end
        else:
            result.append([batch, end, gamma])
    return result


def threshold_ranges(threshold: int, gamma: int, maximum_batch: int) -> list[list[int]]:
    if not 0 <= threshold <= maximum_batch or gamma < 0:
        raise ValueError("invalid threshold policy")
    if threshold == 0:
        return [[1, maximum_batch, 0]]
    if threshold == maximum_batch:
        return [[1, maximum_batch, gamma]]
    return [[1, threshold, gamma], [threshold + 1, maximum_batch, 0]]


def select_regime(rows: list[dict], protocol: dict) -> dict:
    config = validate_protocol(protocol)
    cells = defaultdict(list)
    exclusions = []
    for row in rows:
        if row.get("strategy", "static") != "static" or int(row.get("proposal_budget", row["gamma"])) != int(row["gamma"]):
            exclusions.append({"source": row.get("source"), "reason": "not static direct-draft calibration"})
            continue
        b, g = int(row["batch"]), int(row["gamma"])
        value = _finite(row.get("effective_tpot_ms"))
        if b not in config["batches"] or g not in config["gammas"]:
            continue
        if value is None or value <= 0:
            exclusions.append({"source": row.get("source"), "reason": "no positive measured effective TPOT"})
            continue
        cells[b, g].append(row)
    missing = [{"batch": b, "gamma": g, "proposal_budget": g}
               for b in config["batches"] for g in config["gammas"] if (b, g) not in cells]
    summaries = [{"batch": b, "gamma": g, "proposal_budget": g,
                  "effective_tpot_ms": config["aggregate"](r["effective_tpot_ms"] for r in obs),
                  "n_runs": len(obs), "raw_run_values_ms": [r["effective_tpot_ms"] for r in obs],
                  "source_files": [r["source"] for r in obs]}
                 for (b, g), obs in sorted(cells.items())]
    base = {"status": "waiting_for_calibration" if missing else "selected_pending_evaluation",
            "objective_metric": METRIC, "normalized_batch_weights": config["weights"],
            "missing_cells": missing, "measured_cells": summaries, "exclusions": exclusions,
            "selection_source_files": sorted({r["source"] for obs in cells.values() for r in obs}),
            "score_boundary": "weighted calibration-cell estimate for selection; not an executed-policy measurement"}
    if missing:
        return base
    score = {(r["batch"], r["gamma"]): r["effective_tpot_ms"] for r in summaries}
    weighted = lambda mapping: sum(config["weights"][b] * score[b, mapping[b]] for b in config["batches"])
    static_candidates = [{"gamma": g, "weighted_effective_tpot_ms": weighted({b: g for b in config["batches"]})}
                         for g in config["gammas"]]
    static = min(static_candidates, key=lambda r: (r["weighted_effective_tpot_ms"], r["gamma"]))
    per_batch = {b: min(config["gammas"], key=lambda g: (score[b, g], g)) for b in config["batches"]}
    threshold_candidates = []
    for g in [g for g in config["gammas"] if g > 0] or [0]:
        for threshold in config["thresholds"]:
            mapping = {b: g if b <= threshold else 0 for b in config["batches"]}
            threshold_candidates.append({"active_gamma": g, "disable_above_batch": threshold,
                                         "weighted_effective_tpot_ms": weighted(mapping)})
    threshold = min(threshold_candidates, key=lambda r: (r["weighted_effective_tpot_ms"],
                                                         r["active_gamma"], r["disable_above_batch"]))
    base.update({
        "static": {"selected_gamma": static["gamma"],
                   "selection_score_ms": static["weighted_effective_tpot_ms"],
                   "scope": "one fixed gamma shared by all declared batches",
                   "runner": {"mode": "native", "gamma": static["gamma"], "proposal_budget": static["gamma"]}},
        "native_unrestricted": {"calibrated_batch_to_gamma": per_batch,
            "selection_score_ms": weighted(per_batch),
            "table": native_ranges(per_batch, max(config["batches"])),
            "runner": {"mode": "native", "gamma": config["engine_max_gamma"],
                       "proposal_budget": config["engine_max_gamma"]},
            "semantics": "offline calibrated, frozen deployable table; not an evaluation-set oracle",
            "unmeasured_intermediate_batches": "lower-neighbor gamma assignment, not performance interpolation"},
        "native_threshold": {**threshold,
            "table": threshold_ranges(threshold["disable_above_batch"], threshold["active_gamma"], max(config["batches"])),
            "runner": {"mode": "native", "gamma": config["engine_max_gamma"],
                       "proposal_budget": config["engine_max_gamma"]},
            "semantics": "primary dense-style baseline: native scheduler changes draft production budget"},
        "postproposal_threshold_ablation": {
            "role": "optional execution-cost ablation; does not replace native threshold baseline",
            "runner": {"mode": "dense_heuristic", "gamma": threshold["active_gamma"],
                       "proposal_budget": config["engine_max_gamma"]},
            "plan_override": {"disable_above_batch": threshold["disable_above_batch"]},
            "selection_score_ms": None,
            "reason": "full-budget proposal then truncation has different cost; direct-draft scores are inapplicable"},
        "static_candidate_scores": static_candidates,
        "threshold_candidate_scores": threshold_candidates,
        "evaluation_status": "not_run_by_this_selector",
        "required_evaluation": "execute each frozen policy on separate runs; do not splice steps from other policies"})
    return base


def source_provenance(source: str) -> dict:
    path = Path(source)
    record = {"source": source, "exists_locally": path.is_file()}
    if not path.is_file():
        return record
    record["sha256"] = _hash_file(path)
    if path.suffix == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            requests = obj.get("requests", [r for e in obj.get("episodes", []) for r in e.get("requests", [])])
            indices = sorted([r["source_index"] for r in requests if "source_index" in r])
            record.update({"label": obj.get("label"), "request_count": len(requests),
                           "sample_source_indices": indices,
                           "sample_indices_sha256": hashlib.sha256(_canonical(indices).encode()).hexdigest(),
                           "output_tokens": obj.get("output_tokens"), "measurement_kind": obj.get("measurement_kind")})
    record["context_files"] = [{"path": str(path.parent / name), "sha256": _hash_file(path.parent / name)}
        for name in ("configuration.json", "environment.json", "engine_args.json") if (path.parent / name).is_file()]
    return record


def table_inputs(paths: list[str]) -> tuple[list[dict], list[dict]]:
    """Load optional timing coverage and only explicitly measured objective rows."""
    rows, provenance = [], []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            provenance.append({"path": str(path), "status": "missing"})
            continue
        obj = json.loads(path.read_text())
        records = obj if isinstance(obj, list) else obj["records"]
        metadata = {} if isinstance(obj, list) else obj.get("metadata", {})
        table = CalibrationTable(records, metadata)
        info = {"path": str(path.resolve()), "sha256": _hash_file(path), "record_count": len(records),
                "cost_cells": [{"batch": b, "gamma": g, "proposal_budget": budget}
                               for b, g, budget in sorted(table.cells, key=lambda k: (k[0], k[1], -1 if k[2] is None else k[2]))],
                "metadata": metadata}
        if metadata.get("objective_metric") != METRIC:
            info["status"] = "timing_coverage_only_missing_measured_effective_tpot_objective"
        else:
            accepted = 0
            for r in records:
                tpot = _finite(r.get("effective_tpot_ms"))
                source = r.get("source")
                if tpot is None or tpot <= 0 or not source or int(r.get("request_count", 0)) <= 0:
                    continue
                rows.append({**resolve_regime(r, metadata), "batch": int(r["batch"]), "gamma": int(r["gamma"]),
                             "proposal_budget": int(r.get("proposal_budget", r["gamma"])),
                             "strategy": "static", "effective_tpot_ms": tpot, "source": str(source),
                             "label": r.get("run_id"), "request_count": int(r["request_count"]),
                             "objective_evidence_source": str(path.resolve())})
                accepted += 1
            info["status"] = "measured_objective_rows_loaded" if accepted else "no_valid_measured_objective_rows"
            info["objective_rows"] = accepted
        provenance.append(info)
    return rows, provenance


def write_selection(output: Path, protocol: dict, protocol_path: Path, rows: list[dict],
                    cost_provenance: list[dict], exclusions: list[dict]) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row["source"], row.get("label"), row["batch"], row["gamma"], row.get("proposal_budget"))
        if identity in seen:
            continue
        seen.add(identity)
        groups[tuple(row.get(k, "unspecified") for k in REGIME_FIELDS)].append(row)
    manifest = {"schema_version": 1, "protocol_source": str(protocol_path.resolve()),
                "protocol_sha256": _hash_file(protocol_path), "objective_metric": METRIC,
                "status": "waiting_for_calibration" if not groups else "see_regime_selections",
                "cost_calibrations": cost_provenance, "input_exclusions": exclusions,
                "regimes": [], "evaluation_status": "not_run_by_this_selector"}
    for key, values in sorted(groups.items()):
        regime = dict(zip(REGIME_FIELDS, key))
        name = "regime_" + hashlib.sha256(_canonical(key).encode()).hexdigest()[:10]
        directory = output / name
        directory.mkdir()
        selection = select_regime(values, protocol)
        selection["regime"] = regime
        selection["protocol_sha256"] = manifest["protocol_sha256"]
        selection["source_provenance"] = [source_provenance(p) for p in selection["selection_source_files"]]
        if selection["status"] == "selected_pending_evaluation":
            static = {**selection["static"], "selection_source_files": selection["selection_source_files"],
                      "regime": regime, "objective_metric": METRIC,
                      "batch_weights": selection["normalized_batch_weights"],
                      "status": "selected_on_calibration_pending_separate_evaluation"}
            (directory / "static_selection.json").write_text(json.dumps(static, indent=2))
            for kind in ("native_unrestricted", "native_threshold"):
                filename = directory / (kind + "_table.json")
                filename.write_text(json.dumps(selection[kind]["table"], indent=2))
                selection[kind]["runner"]["native_table"] = str(filename.resolve())
            selection["static"]["plot_controller_static_selection"] = str((directory / "static_selection.json").resolve())
        path = directory / "selection.json"
        path.write_text(json.dumps(selection, indent=2))
        manifest["regimes"].append({"regime": regime, "status": selection["status"],
                                    "selection": str(path), "missing_cells": len(selection["missing_cells"])})
    (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def self_test() -> None:
    class SelectorTests(unittest.TestCase):
        def protocol(self):
            p = json.loads(DEFAULT_PROTOCOL.read_text())
            p["batches"] = [1, 4]
            p["candidate_gammas"] = [0, 1, 3]
            p["objective"]["batch_weights"] = {"1": .25, "4": .75}
            p["selection"]["threshold_candidates"] = [0, 1, 4]
            return p

        def rows(self):
            return [{"batch": b, "gamma": g, "proposal_budget": g, "strategy": "static",
                     "effective_tpot_ms": value, "source": f"synthetic-{b}-{g}"}
                    for b, costs in [(1, [1, 2, 4]), (4, [9, 8, 2])]
                    for g, value in zip([0, 1, 3], costs)]

        def test_global_static_is_one_gamma(self):
            out = select_regime(self.rows(), self.protocol())
            self.assertEqual(out["static"]["selected_gamma"], 3)
            self.assertEqual(out["static"]["selection_score_ms"], 2.5)
            self.assertEqual(out["native_unrestricted"]["calibrated_batch_to_gamma"], {1: 0, 4: 3})

        def test_threshold_uses_native_draft_budget(self):
            out = select_regime(self.rows(), self.protocol())
            self.assertEqual(out["native_threshold"]["runner"]["mode"], "native")
            self.assertEqual(out["native_threshold"]["disable_above_batch"], 4)
            self.assertIsNone(out["postproposal_threshold_ablation"]["selection_score_ms"])

        def test_nonmonotonic_and_zero_table(self):
            table = native_ranges({1: 0, 4: 3, 16: 0, 64: 7}, 64)
            self.assertEqual(table, [[1, 3, 0], [4, 15, 3], [16, 63, 0], [64, 64, 7]])
            self.assertEqual(threshold_ranges(4, 3, 64), [[1, 4, 3], [5, 64, 0]])

        def test_missing_data_does_not_select_partial_winner(self):
            self.assertEqual(select_regime([], self.protocol())["status"], "waiting_for_calibration")
            self.assertNotIn("static", select_regime(self.rows()[:-1], self.protocol()))

        def test_postproposal_cells_do_not_replace_direct_cells(self):
            rows = self.rows()
            rows[-1]["proposal_budget"] = 7
            self.assertEqual(select_regime(rows, self.protocol())["missing_cells"],
                             [{"batch": 4, "gamma": 3, "proposal_budget": 3}])

        def test_plain_cost_table_does_not_supply_objective(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "table.json"
                path.write_text(json.dumps({"records": [{"batch": 1, "gamma": 0, "total_step_ms": 10}]}))
                rows, provenance = table_inputs([str(path)])
                self.assertEqual(rows, [])
                self.assertIn("coverage_only", provenance[0]["status"])

        def test_objective_table_requires_measured_boundary_and_population(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "table.json"
                path.write_text(json.dumps({"metadata": {"objective_metric": METRIC, "execution_mode": "eager"},
                    "records": [{"batch": 1, "gamma": 0, "total_step_ms": 10, "effective_tpot_ms": 11,
                                 "source": "synthetic.json", "request_count": 64}]}))
                rows, _ = table_inputs([str(path)])
                self.assertEqual(rows[0]["effective_tpot_ms"], 11)
                self.assertEqual(rows[0]["execution_mode"], "eager")

        def test_static_file_compatible_with_plot_controller(self):
            from plot_controller import selected_static
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "static.json"
                path.write_text(json.dumps({"selected_gamma": 3, "selection_source_files": ["calibration.json"]}))
                self.assertEqual(selected_static(str(path), [{"source": "evaluation.json"}])["selected_gamma"], 3)
                with self.assertRaises(ValueError):
                    selected_static(str(path), [{"source": "calibration.json"}])
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelectorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", nargs="*", default=[])
    parser.add_argument("--cost-calibration", nargs="*", default=[])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output:
        parser.error("--output is required unless --self-test")
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    missing = [{"source": str(p), "reason": "input does not exist"}
               for p in args.performance if not Path(p).exists()]
    available = [p for p in args.performance if Path(p).exists()]
    runs, excluded = load_runs(available) if available else ([], [])
    objective_rows, costs = table_inputs(args.cost_calibration)
    # A real performance file is authoritative if an enriched timing table
    # references the same source. Do not count that source as another repeat.
    perf_sources = {r["source"] for r in runs}
    objective_rows = [r for r in objective_rows if r["source"] not in perf_sources]
    manifest = write_selection(Path(args.output), protocol, protocol_path,
                               runs + objective_rows, costs, missing + excluded)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bounded OLMoE step-0 serial-vs-natural-batch conformance prevalence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import run_exact_event_conformance as EXACT

TARGETS = 4

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steady-capture", type=Path, required=True)
    parser.add_argument("--bursty-capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-width8", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()

def ordered_request_ids(capture: Mapping[str, Any]) -> list[str]:
    values = [str(row["request_id"]) for row in capture["manifest"]["requests"]]
    if len(values) != len(set(values)) or set(values) != set(capture["ledger"]):
        raise EXACT.ExperimentError("manifest/ledger request identity does not close")
    for request_id in values:
        ledger = capture["ledger"][request_id]
        if not ledger.get("steps") or int(ledger["steps"][0]["decode_step"]) != 0:
            raise EXACT.ExperimentError(f"missing captured step 0 for {request_id}")
    return values

def build_cases(regime: str, capture: Mapping[str, Any], include_width8: bool) -> list[dict[str, Any]]:
    ordered = ordered_request_ids(capture)
    natural = []
    for batch in sorted(capture["batches"], key=lambda row: int(row["batch_index"])):
        ids = [str(value) for value in batch["request_ids"]]
        steps = [int(value) for value in batch["decode_steps"]]
        if len(ids) != TARGETS or steps != [0] * TARGETS:
            continue
        quartet = ids[:TARGETS]
        lengths = [int(capture["ledger"][value]["prompt_tokens"]) for value in quartet]
        documents = [str(capture["ledger"][value]["document_id"]) for value in quartet]
        if len(set(lengths)) >= 2 and len(set(documents)) == TARGETS:
            natural = quartet
            natural_batch_index = int(batch["batch_index"])
            break
    if len(natural) != TARGETS:
        raise EXACT.ExperimentError("no captured heterogeneous four-row step-0 batch")
    target_ids = natural
    width2 = {}
    for target in target_ids:
        target_length = int(capture["ledger"][target]["prompt_tokens"])
        companion = next((value for value in target_ids if value != target
                          and int(capture["ledger"][value]["prompt_tokens"]) != target_length), None)
        if companion is None:
            raise EXACT.ExperimentError("cannot form captured-batch heterogeneous width-2 control")
        width2[target] = [target, companion]
    memberships: dict[int, dict[str, list[str]]] = {
        2: width2,
        4: {target: natural for target in target_ids},
    }
    if include_width8:
        used_documents = {str(capture["ledger"][value]["document_id"]) for value in target_ids}
        octet = list(natural)
        for request_id in ordered:
            document = str(capture["ledger"][request_id]["document_id"])
            if request_id not in octet and document not in used_documents:
                octet.append(request_id); used_documents.add(document)
            if len(octet) == 8: break
        octet = [value for value in ordered if value in set(octet)]
        if len(octet) == 8:
            memberships[8] = {target: octet for target in target_ids}
        else:
            raise EXACT.ExperimentError("--include-width8 requires eight distinct-document states")
    cases: list[dict[str, Any]] = []
    for width, rows in memberships.items():
        for target in target_ids:
            members = list(rows[target])
            lengths = [int(capture["ledger"][value]["prompt_tokens"]) for value in members]
            if len(members) != width or len(set(members)) != width or len(set(lengths)) < 2:
                raise EXACT.ExperimentError("natural heterogeneous membership did not close")
            cases.append(
                {
                    "case_id": f"{regime}:w{width}:{target}",
                    "regime": regime,
                    "width": width,
                    "target_request_id": target,
                    "target_row": members.index(target),
                    "request_ids": members,
                    "document_ids": [str(capture["ledger"][value]["document_id"]) for value in members],
                    "prompt_lengths": lengths,
                    "step": 0,
                    "historical_source_batch": width == 4,
                    "historical_source_batch_index": (
                        natural_batch_index if width == 4 else None
                    ),
                }
            )
    return cases

def model_contract(capture: Mapping[str, Any]) -> dict[str, Any]:
    manifest = capture["manifest"]
    return {
        "model": {key: manifest["model"].get(key) for key in ("id", "revision", "tokenizer_revision", "dtype")},
        "generation": {key: manifest["generation"].get(key) for key in ("mode", "do_sample", "max_decode_steps")},
    }

def build_states(model: Any, prepared: Mapping[str, Any], capture: Mapping[str, Any], request_ids: Sequence[str]) -> list[Any]:
    return [
        EXACT.prefill_state(model, prepared[request_id],
                            int(capture["ledger"][request_id]["steps"][0]["input_token_id"]))
        for request_id in request_ids
    ]

def input_equality(a: Any, batch: Any) -> dict[str, Any]:
    target_row = int(batch.target_row)
    result = {
        "target_logical_cache_sha256": a.target_logical_cache_sha256,
        "batch_target_logical_cache_sha256": batch.target_logical_cache_sha256,
        "target_token_id": int(a.input_ids[0, 0].item()),
        "batch_target_token_id": int(batch.input_ids[target_row, 0].item()),
        "target_position_id": int(a.position_ids[0, 0].item()),
        "batch_target_position_id": int(batch.position_ids[target_row, 0].item()),
        "cache_storage_non_alias": not bool(a.cache_storage_ptrs.intersection(batch.cache_storage_ptrs)),
    }
    result["passed"] = bool(
        result["target_logical_cache_sha256"]
        == result["batch_target_logical_cache_sha256"]
        and result["target_token_id"] == result["batch_target_token_id"]
        and result["target_position_id"] == result["batch_target_position_id"]
        and result["cache_storage_non_alias"]
    )
    if not result["passed"]:
        raise EXACT.ExperimentError(f"target causal input equality failed: {result}")
    return result

def summarize_case(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hashes = {
        arm: [EXACT.arm_stability_fingerprint(row["arms"][arm]) for row in repeats]
        for arm in ("A", "C")
    }
    stable = {arm: len(set(values)) == 1 for arm, values in hashes.items()}
    comparisons = [row["A_vs_C"] for row in repeats]
    flips = [int(row["route_flip_layers"]) for row in comparisons]
    near = [int(row["near_boundary_associated_route_flip_layers"]) for row in comparisons]
    total_flips = sum(flips)
    return {
        "within_arm_repeat_stable": all(stable.values()),
        "stable_by_arm": stable,
        "arm_fingerprints": hashes,
        "route_flip_layers_by_repeat": flips,
        "route_divergence_in_every_repeat": bool(flips and min(flips) > 0),
        "near_boundary_route_flip_fraction_weighted": sum(near) / max(1, total_flips),
        "first_exact_causal_stage_by_repeat": [row["first_exact_causal_stage"] for row in comparisons],
        "first_material_causal_stage_by_repeat": [row["first_non_allclose_or_route_causal_stage"] for row in comparisons],
        "final_logits_by_repeat": [row["final_logits"] for row in comparisons],
        "predicted_token_changed_by_repeat": [bool(row["predicted_token_changed"]) for row in comparisons],
    }

def sealed_files(output_dir: Path) -> dict[str, dict[str, Any]]:
    excluded = {"RUN_COMPLETE.json", "RUN_FAILED.json", "run.log"}
    return {
        str(path.relative_to(output_dir)): {"sha256": EXACT.sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }

def main() -> None:
    args = parse_args()
    if not args.offline:
        raise EXACT.ExperimentError("pass --offline; model downloads are forbidden")
    if args.repeats < 3:
        raise EXACT.ExperimentError("at least three repeats are required")
    if sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV"):
        raise EXACT.ExperimentError("run with system Python, not a virtual environment")
    captures = {
        "steady": EXACT.load_capture(args.steady_capture.resolve(), "steady"),
        "bursty": EXACT.load_capture(args.bursty_capture.resolve(), "bursty"),
    }
    contracts = {key: model_contract(value) for key, value in captures.items()}
    if EXACT.canonical_sha256(contracts["steady"]) != EXACT.canonical_sha256(
        contracts["bursty"]
    ):
        raise EXACT.ExperimentError("steady/bursty model-generation contracts differ")
    for key, expected in EXACT.BASE.EXPECTED_MODEL.items():
        if contracts["steady"]["model"].get(key) != expected:
            raise EXACT.ExperimentError(f"capture model contract differs at {key}")
    cases = [
        case
        for regime in ("steady", "bursty")
        for case in build_cases(regime, captures[regime], args.include_width8)
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in output_dir.iterdir() if path.name != "run.log"]
    if unexpected:
        raise EXACT.ExperimentError(
            f"output directory contains non-wrapper files: {sorted(unexpected)}"
        )
    EXACT.write_json_exclusive(
        output_dir / "RUN_STARTED.json",
        {"status": "RUN_STARTED", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    )
    EXACT.write_text_exclusive(
        output_dir / "commands.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join([sys.executable, *sys.argv]) + "\n",
        mode=0o755,
    )
    if EXACT.BASE.query_gpu_compute_processes():
        raise EXACT.ExperimentError("GPU is not idle before model load")
    torch, transformers, tokenizer, model, load_seconds = EXACT.BASE.load_exact_model(
        captures["steady"]["manifest"]
    )
    own_process = EXACT.BASE.query_gpu_compute_processes()
    if len(own_process) != 1:
        raise EXACT.ExperimentError("model load did not create one isolated GPU process")
    monitor = EXACT.BASE.GpuIsolationMonitor(own_process)
    selected_ids = {
        regime: sorted(
            {value for case in cases if case["regime"] == regime for value in case["request_ids"]}
        )
        for regime in captures
    }
    prepared_by_regime: dict[str, dict[str, Any]] = {}
    prompt_identity = {}
    for regime, capture in captures.items():
        prepared = EXACT.CAPTURE._prepare_requests(capture["manifest"], tokenizer, model.device)
        prepared_by_regime[regime] = {state.request_id: state for state in prepared}
        prompt_identity[regime] = EXACT.BASE.validate_tokenizer_and_prompt_identity(
            capture["manifest"], tokenizer, prepared, selected_ids[regime]
        )
    config = {
        "schema": "longrun-a-prevalence-config-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": EXACT.git_value("rev-parse", "HEAD"),
        "git_branch": EXACT.git_value("branch", "--show-current"),
        "git_status_short": EXACT.git_value("status", "--short"),
        "source_sha256": {
            str(Path(path).resolve().relative_to(EXACT.ROOT)): EXACT.sha256_file(Path(path))
            for path in (__file__, EXACT.__file__, EXACT.CAPTURE.__file__, EXACT.BASE.__file__)
        },
        "model": EXACT.BASE.EXPECTED_MODEL,
        "runtime": EXACT.BASE._environment(torch, transformers),
        "model_load_seconds": load_seconds,
        "offline": True,
        "repeats": args.repeats,
        "targets_per_regime_width": TARGETS,
        "selection_rule": "first captured distinct-document heterogeneous step-0 width-4 batch; width-2 uses in-batch unequal-length companion; outcomes unused",
        "cases": cases,
        "prompt_identity": prompt_identity,
        "capture_hashes": {key: value["hashes"] for key, value in captures.items()},
        "cross_capture_contract": contracts,
        "near_tie_margin_inherited_pre_run": EXACT.NEAR_TIE_MARGIN,
        "claim_ceiling": "single-OLMoE single-RTX5090 custom-runtime step-0 prevalence diagnostic only",
    }
    EXACT.write_json_exclusive(output_dir / "config.json", config)
    repeat_rows: dict[str, list[dict[str, Any]]] = {case["case_id"]: [] for case in cases}
    monitor.start()
    try:
        for case in cases:
            regime = str(case["regime"])
            capture = captures[regime]
            for repeat in range(args.repeats):
                monitor.check(f"before_{case['case_id']}_{repeat}")
                monitor.require_clean()
                seed = int(capture["manifest"]["seed"])
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                states = build_states(
                    model, prepared_by_regime[regime], capture, case["request_ids"]
                )
                target = states[int(case["target_row"])]
                arms = {
                    "A": EXACT.build_arm("A", [target], 0, [case["target_request_id"]]),
                    "C": EXACT.build_arm(
                        "C", states, int(case["target_row"]), case["request_ids"]
                    ),
                }
                equality = input_equality(arms["A"], arms["C"])
                order = ["A", "C"] if repeat % 2 == 0 else ["C", "A"]
                records = {name: EXACT.run_arm(torch, model, arms[name]) for name in order}
                records = {name: records[name] for name in ("A", "C")}
                comparison = EXACT.compare_records(records["A"], records["C"])
                source_reproduction = {
                    "applicable": bool(case["historical_source_batch"]),
                    "all_rows_input_token_match": None,
                    "all_rows_route_match": None,
                    "all_rows_predicted_token_match": None,
                }
                if case["historical_source_batch"]:
                    expected_inputs = [
                        int(capture["ledger"][request_id]["steps"][0]["input_token_id"])
                        for request_id in case["request_ids"]
                    ]
                    expected_tokens = [
                        int(
                            capture["ledger"][request_id]["steps"][0][
                                "predicted_next_token_id"
                            ]
                        )
                        for request_id in case["request_ids"]
                    ]
                    expected_routes = [
                        [
                            [int(value) for value in layer["experts"]]
                            for layer in capture["ledger"][request_id]["steps"][0][
                                "route_signature"
                            ]
                        ]
                        for request_id in case["request_ids"]
                    ]
                    source_reproduction.update(
                        {
                            "all_rows_input_token_match": [
                                int(value) for value in arms["C"].input_ids[:, 0].tolist()
                            ]
                            == expected_inputs,
                            "all_rows_route_match": all(
                                records["C"]["all_routes"][layer][row]
                                == expected_routes[row][layer]
                                for layer in range(len(records["C"]["all_routes"]))
                                for row in range(len(case["request_ids"]))
                            ),
                            "all_rows_predicted_token_match": records["C"][
                                "all_predicted_tokens"
                            ]
                            == expected_tokens,
                        }
                    )
                    if not all(
                        source_reproduction[key]
                        for key in (
                            "all_rows_input_token_match",
                            "all_rows_route_match",
                            "all_rows_predicted_token_match",
                        )
                    ):
                        raise EXACT.ExperimentError(
                            f"width-4 historical source reproduction failed: {source_reproduction}"
                        )
                public = {
                    "case_id": case["case_id"],
                    "repeat": repeat,
                    "arm_order": order,
                    "causal_input_equality": equality,
                    "source_reproduction": source_reproduction,
                    "arms": {name: record["public"] for name, record in records.items()},
                    "A_vs_C": comparison,
                }
                repeat_path = output_dir / "repeats" / case["case_id"].replace(":", "_") / f"repeat_{repeat}.json"
                EXACT.write_json_exclusive(repeat_path, public)
                repeat_rows[case["case_id"]].append(public)
                monitor.check(f"after_{case['case_id']}_{repeat}")
                monitor.require_clean()
                print(json.dumps({"case": case["case_id"], "repeat": repeat, "route_flip_layers": comparison["route_flip_layers"]}, sort_keys=True), flush=True)
                del states, arms, records
                torch.cuda.empty_cache()
    finally:
        monitor.stop()
    monitor.require_clean()
    case_summaries = {
        case_id: summarize_case(rows) for case_id, rows in repeat_rows.items()
    }
    by_config = {}
    for regime in ("steady", "bursty"):
        for width in sorted({int(case["width"]) for case in cases if case["regime"] == regime}):
            keys = [case["case_id"] for case in cases if case["regime"] == regime and int(case["width"]) == width]
            summaries = [case_summaries[key] for key in keys]
            by_config[f"{regime}:w{width}"] = {
                "target_cases": len(keys),
                "all_cases_within_arm_repeat_stable": all(row["within_arm_repeat_stable"] for row in summaries),
                "stable_route_divergence_cases": sum(row["within_arm_repeat_stable"] and row["route_divergence_in_every_repeat"] for row in summaries),
                "stable_route_divergence_case_fraction": sum(row["within_arm_repeat_stable"] and row["route_divergence_in_every_repeat"] for row in summaries) / max(1, len(keys)),
            }
    results = {
        "schema": "longrun-a-prevalence-results-v1",
        "case_summaries": case_summaries,
        "by_regime_width": by_config,
        "gpu_isolation": monitor.summary(),
        "latency_claim_allowed": False,
    }
    EXACT.write_json_exclusive(output_dir / "results.json", results)
    EXACT.write_json_exclusive(
        output_dir / "RUN_COMPLETE.json",
        {
            "status": "PREVALENCE_CHECK_COMPLETE",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": sealed_files(output_dir),
            "unsealed_stream_files": {"run.log": "wrapper-owned tee stream; excluded from seal"},
        },
    )
    print(json.dumps({"status": "PREVALENCE_CHECK_COMPLETE", "output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        try:
            if "--output-dir" in sys.argv:
                output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
                if (output_dir / "RUN_STARTED.json").exists() and not (output_dir / "RUN_COMPLETE.json").exists() and not (output_dir / "RUN_FAILED.json").exists():
                    EXACT.write_json_exclusive(
                        output_dir / "RUN_FAILED.json",
                        {
                            "status": "RUN_FAILED",
                            "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "failure_category": EXACT.classify_failure(error),
                            "files": sealed_files(output_dir),
                            "unsealed_stream_files": {"run.log": "wrapper-owned tee stream; excluded from seal"},
                        },
                    )
        except BaseException:
            pass
        raise

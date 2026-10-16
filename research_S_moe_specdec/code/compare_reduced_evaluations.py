#!/usr/bin/env python3
"""CPU-only F3/F4/F6 tables for one frozen reduced-domain evaluation.

  python code/compare_reduced_evaluations.py \
    --preparation runs/DOMAIN_controller_preparation_r01 \
    --output runs/DOMAIN_controller_comparison_r01

Read evaluation_commands.json, its frozen artifacts and original run JSON.
Missing evidence is pending, incompatible evidence is invalid. Neither produces
an invented measurement. Only complete two-repeat cells enter comparisons;
all available valid raw repeat values remain in the output. No GPU or plotting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from compare_controller_policies import (
    CONTROL_METRICS, METRICS, ROLES, canonical, controls, output_csv,
    population_counter_energy, sample_identity, selection_checks, sha,
    summarize, validate_policy,
)
from plot_results import REGIME_FIELDS, _finite, _requests, load_runs
from prepare_reduced_evaluation import (
    RUNTIME_SOURCE, behavior_sha, check_config, domain_signature, expected_spec,
    validate_cost_grid,
)
from select_controller_baselines import validate_protocol


PROJECT = Path(__file__).resolve().parents[1]
ENV_FIELDS = ("vllm", "torch", "cuda", "gpu", "model_runner_version", "VLLM_USE_V2_MODEL_RUNNER")
F3_METRICS = ("effective_tpot_ms", "population_wall_s", "population_tokens_per_second",
              "population_inter_episode_gap_s", "wall_s", "tokens_per_second")
F4_METRICS = ("energy_j", "joules_per_token", "population_counter_energy_j",
              "population_counter_joules_per_token", "population_counter_elapsed_s")
F6_METRICS = (*CONTROL_METRICS, "policy_choose_cpu_us_sum", "wrapper_cpu_us_sum",
              "drafts_available_before", "truncated_by_controller", "scheduled_drafts")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def relocate(value, base=None):
    """Resolve original remote paths after copying the complete S project."""
    path = Path(value)
    if path.exists():
        return path.resolve()
    candidates = [PROJECT / path] if not path.is_absolute() else []
    if base is not None and not path.is_absolute():
        candidates.insert(0, Path(base) / path)
    for name in ("runs", "configs"):
        marker = f"/{name}/"
        if marker in str(path):
            candidates.append(PROJECT / name / str(path).split(marker, 1)[1])
    return next((p.resolve() for p in candidates if p.exists()),
                (candidates[0] if candidates else path).resolve())


def artifact(manifest, name, hash_key=None):
    path = relocate(manifest[name])
    obj = read(path)
    require(sha(path) == manifest[hash_key or name + "_sha256"], f"frozen {name} SHA changed")
    return path, obj


def preparation(path):
    manifest_path = Path(path)
    if manifest_path.suffix != ".json":
        manifest_path /= "evaluation_commands.json"
    obj = read(manifest_path)
    require(obj.get("status") == "ready_for_serialized_evaluation", "preparation is not ready for evaluation")
    _, plan = artifact(obj, "evaluation_config")
    _, protocol = artifact(obj, "protocol")
    selection_path, selection = artifact(obj, "selection")
    table_path, table = artifact(obj, "calibration_table", "calibration_sha256")
    check_config(plan, calibration=False)
    parsed = validate_protocol(protocol)
    candidates = obj["candidate_gammas"]
    require(candidates == parsed["gammas"] == plan["candidate_gammas"], "candidate grids differ")
    require(parsed["batches"] == plan["batches"], "protocol and evaluation batches differ")
    validate_cost_grid(table["records"])
    require(len(table["records"]) == obj["calibration_cells"], "calibration cell count differs")
    require(selection.get("status") == "selected_pending_evaluation", "frozen selection is incomplete")
    require(selection["static"]["selected_gamma"] in candidates, "selected static gamma outside candidates")
    commands = obj["commands"]
    require(Counter(c["role"] for c in commands) == Counter(ROLES), "manifest needs exactly the five policy roles")
    require(len({c["run"] for c in commands}) == len(ROLES), "policy roles share a run directory")
    for command in commands:
        require(Path(command["run"]).name == command["run"], "manifest run must be one directory-name component")
        require(sha(relocate(command["config"])) == obj["evaluation_config_sha256"], "command config differs from frozen evaluation config")
        require(command["expected_performance_run_records"] == len(plan["batches"]) * plan["repeats"], "manifest record count differs")
    workload = relocate(plan["workload"])
    records = read(workload)["records"]
    development = sorted(r["source_index"] for r in records[:64])
    heldout = sorted(r["source_index"] for r in records[64:128])
    require(len(set(development + heldout)) == 128, "development/held-out populations are incomplete or overlap")
    evidence = obj["calibration_evidence"]
    native_evidence = obj["selection_evidence"]
    for key in ("domain", "regime", "environment", "runtime_behavior", "workload_sha256"):
        require(evidence[key] == native_evidence[key], f"selection/calibration evidence differs in {key}")
    require(domain_signature(plan) == evidence["domain"], "evaluation domain differs from preparation")
    require(plan.get("preparation_domain") == obj["domain"], "evaluation domain label differs")
    require(evidence["workload_sha256"] == sha(workload), "workload differs from preparation")
    require(all(table["metadata"].get(k) == v for k, v in evidence["regime"].items()), "calibration regime differs from preparation")
    require(selection["regime"] == evidence["regime"], "selection regime differs from preparation")
    sources = {}
    for group in (evidence, native_evidence):
        require(group["development_source_indices"] == development and group["heldout_source_indices"] == heldout,
                "preparation request identities differ from the actual workload")
        require(bool(group["source_files"]), "missing development source provenance")
        for item in group["source_files"]:
            source = relocate(item["source"])
            require(sha(source) == item["sha256"], "development source SHA changed")
            require(sorted(r["source_index"] for r in _requests(read(source))) == development,
                    "development source population differs from the declared slice")
            sources[item["source"]] = item["sha256"]
    require(bool(selection.get("source_provenance")), "selection lacks source provenance")
    for item in selection["source_provenance"]:
        require(sources.get(item["source"]) == item.get("sha256"), "selection source is outside the verified development evidence")
        require(item.get("sample_source_indices") == development, "selection source identities differ")
    for cell in table["records"]:
        require(cell.get("source") in sources, "calibration source is outside the verified development evidence")
        require((_finite(cell.get("total_step_ms")) or 0) > 0, "calibration lacks a positive measured complete-step cost")
    return {"manifest": obj, "manifest_path": manifest_path.resolve(), "plan": plan,
            "selection": selection, "selection_path": selection_path, "table_path": table_path,
            "workload": workload, "heldout": heldout, "evidence": evidence}


def campaign(context, command):
    manifest, plan, selection = (context[k] for k in ("manifest", "plan", "selection"))
    directory = PROJECT / "runs" / command["run"]
    status = read(directory / "status.json")
    if status.get("status") != "complete":
        if status.get("status") in ("waiting_for_lock", "running"):
            raise FileNotFoundError(f"campaign not complete yet: {directory} ({status.get('status')})")
        raise ValueError(f"campaign did not complete: {status.get('status')}")
    role = command["role"]
    gamma = 0 if role == "ar" else selection["static"]["selected_gamma"] if role == "static" else max(manifest["candidate_gammas"])
    cells = status.get("cells", [])
    require(len(cells) == 1 and cells[0].get("gamma") == gamma and cells[0].get("returncode") == 0,
            "campaign must contain exactly its one successful frozen policy cell")
    require(read(directory / "plan.json") == plan, "executed campaign plan differs from frozen evaluation config")
    folder = directory / f"gamma{gamma}"
    require(read(folder / "status.json").get("status") == "complete", "runner did not complete")
    config, engine, env = (read(folder / name) for name in ("configuration.json", "engine_args.json", "environment.json"))
    require(config["plan"] == plan, "runner plan differs from frozen evaluation config")
    args = config["args"]
    mode = "adaptive" if role == "adaptive" else "native"
    require(args.get("mode") == mode and args.get("gamma") == gamma and not args.get("natural_stop"), "runner mode/gamma/termination differs from frozen role")
    require(args.get("proposal_budget", gamma) == gamma, "runner proposal budget differs from frozen role")
    for key in ("batches", "repeats", "output_tokens", "admission_mode"):
        require(args.get(key) is None or args[key] == plan[key], f"unexpected runtime {key} override")
    spec = expected_spec(plan, gamma)
    if role.startswith("native_"):
        spec["num_speculative_tokens_per_batch_size"] = selection[role]["table"]
        require(read(directory / "native_table.json") == selection[role]["table"], "executed native table differs from frozen selection")
        require(bool(args.get("native_table")), "native table was not supplied to runner")
    else:
        require(not args.get("native_table"), "unexpected native table")
    require(engine.get("speculative_config") == spec, "actual engine draft method/model/budget/quantization/table differs from frozen role")
    if role == "adaptive":
        require(bool(args.get("calibration")) and sha(directory / "calibration.json") == manifest["calibration_sha256"], "executed adaptive calibration differs from frozen table")
    sig = domain_signature(plan)
    for key in ("model", "dtype", "quantization", "kv_cache_memory_bytes", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "enforce_eager", "kernel_config"):
        require(engine.get(key) == sig[key], f"actual engine domain mismatch: {key}")
    require(engine.get("async_scheduling") is False and not engine.get("enable_return_routed_experts", False), "instrumented/asynchronous performance engine")
    require({k: env.get(k) for k in ENV_FIELDS} == context["evidence"]["environment"], "evaluation and calibration environments differ")
    source_hashes = {}
    for name in RUNTIME_SOURCE:
        source = directory / "source" / name
        source_hashes[name] = sha(source)
        require(env.get("source_sha256", {}).get(name) == source_hashes[name], f"captured/actual source SHA differs: {name}")
        require(behavior_sha(source) == context["evidence"]["runtime_behavior"][name], f"evaluation/calibration runtime behavior differs: {name}")
    initial = read(folder / "evaluation_initial_state.json")
    require(all(initial.get(k) == plan[k] for k in ("input_offset", "warmup_offset", "shape_warmup_offset")), "actual evaluation initial-state offsets differ")
    return folder, config, engine, initial, source_hashes


def observation(path, role, context, config, engine, initial, source_hashes):
    rows, rejected = load_runs([path])
    require(len(rows) == 1 and not rejected, f"not one eligible measured run: {rejected}")
    row, record, plan = rows[0], read(path), context["plan"]
    require(record.get("all_finished") is True, "population is not complete")
    require(row["measurement_kind"] == "fixed_length_generation", "not a fixed-length generation record")
    require(all(row[k] == context["evidence"]["regime"][k] for k in REGIME_FIELDS), "performance regime differs from calibration/selection")
    identity = sample_identity(record, config, context["workload"], plan["input_offset"], plan["measured_requests"], plan["output_tokens"])
    require(identity["sample_source_indices"] == context["heldout"], "actual held-out IDs differ from preparation")
    validate_policy(role, row, engine, context["selection"])
    independence = selection_checks(context["selection"], row, identity, sha(path))
    control, snapshots = controls(record)
    if role == "adaptive":
        for phase in control:
            require(set(map(int, phase["selected_gamma_histogram"])) <= set(context["manifest"]["candidate_gammas"]), "executed adaptive gamma outside frozen candidates")
    population_wall = _finite(record.get("population_wall_s"))
    require(population_wall is not None and population_wall > 0 and population_wall >= row["wall_s"] - 1e-8, "missing/invalid complete-population wall time")
    gap = _finite(record.get("population_inter_episode_gap_s"))
    require(gap is not None and abs(gap - (population_wall - row["wall_s"])) < 1e-6, "population inter-wave gap disagrees with timing windows")
    boundary = plan.get("semantic_token_boundary")
    if boundary is not None:
        require(config.get("all_workload_prompt_tokens_at_or_above_boundary") == 0, "semantic prompt token boundary exceeded or unrecorded")
        require(all(r.get("output_tokens_at_or_above_semantic_boundary") == 0 and not any(t >= boundary for t in r["output_token_ids"]) for r in _requests(record)), "semantic output token boundary exceeded or unrecorded")
    complete = {**row, **identity, "role": role, "domain": context["manifest"]["domain"],
        "source_sha256": sha(path), "configuration_sha256": sha(path.parent / "configuration.json"),
        "engine_args_sha256": sha(path.parent / "engine_args.json"), "runtime_source_sha256": source_hashes,
        "selection_independence": independence, "population_wall_s": population_wall,
        "population_tokens_per_second": row["output_tokens"] / population_wall,
        "population_inter_episode_gap_s": gap, "energy_j": _finite(record.get("energy_j")),
        "native_table": (engine.get("speculative_config") or {}).get("num_speculative_tokens_per_batch_size"),
        "engine_evaluation_initial_state": initial, "acceptance_snapshots": snapshots,
        **population_counter_energy(record)}
    controls_out = [{**{k: complete[k] for k in (*REGIME_FIELDS, "sample_indices_sha256", "role", "batch", "source", "repeat")}, **c} for c in control]
    return complete, controls_out


def table_rows(summary, coverage, metrics, *, phase=False):
    """Leave unpaired cells blank; never choose an available/better repeat."""
    indexed = {(r["role"], r["batch"], r.get("phase")): r for r in summary}
    output = []
    for cell in coverage:
        phases = sorted({key[2] for key in indexed if key[:2] == (cell["role"], cell["batch"])}) if phase else [None]
        for name in phases or ["decode"]:
            source = indexed.get((cell["role"], cell["batch"], name), {})
            row = {**cell, **({"phase": name} if phase else {})}
            for metric in metrics:
                for suffix in ("raw", "observed_runs", "median", "min", "max"):
                    value = source.get(metric + "_" + suffix, [] if suffix == "raw" else 0 if suffix == "observed_runs" else None)
                    if suffix in ("median", "min", "max") and (cell["status"] != "measured" or source.get(metric + "_observed_runs") != cell["expected_run_records"]):
                        value = None
                    row[metric + "_" + suffix] = value
            output.append(row)
    return output


def analyze(preparation_path, output):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "choose a fresh/empty derived output directory")
    runs, control_runs, coverage, issues = [], [], [], []
    context = None
    try:
        context = preparation(preparation_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append({"status": "pending" if isinstance(exc, FileNotFoundError) else "invalid", "source": str(preparation_path), "reason": str(exc)})
    if context:
        plan = context["plan"]
        for command in context["manifest"]["commands"]:
            role, accepted, role_issues = command["role"], [], []
            try:
                folder, config, engine, initial, sources = campaign(context, command)
                expected = {folder / f"perf-r{r}-b{b}.json": (r, b) for r in range(plan["repeats"]) for b in plan["batches"]}
                extra = set(folder.glob("perf-*.json")) - set(expected)
                require(not extra, f"extra performance records outside frozen evaluation grid: {sorted(map(str, extra))}")
                for path, (repeat, batch) in sorted(expected.items()):
                    try:
                        record, control = observation(path, role, context, config, engine, initial, sources)
                        require((record["repeat"], record["batch"]) == (repeat, batch), "record label/B differs from its prescribed file")
                        runs.append(record); accepted.append(record); control_runs.extend(control)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        role_issues.append({"role": role, "batch": batch, "repeat": repeat, "source": str(path), "status": "pending" if isinstance(exc, FileNotFoundError) else "invalid", "reason": str(exc)})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                role_issues.append({"role": role, "source": command["run"], "status": "pending" if isinstance(exc, FileNotFoundError) else "invalid", "reason": str(exc)})
            issues.extend(role_issues)
            for batch in plan["batches"]:
                selected = [r for r in accepted if r["batch"] == batch]
                reasons = [e for e in role_issues if e.get("batch", batch) == batch]
                status = "invalid" if any(e["status"] == "invalid" for e in reasons) else "pending" if reasons or sorted(r["repeat"] for r in selected) != list(range(plan["repeats"])) else "measured"
                coverage.append({"domain": context["manifest"]["domain"], "role": role, "batch": batch,
                    "expected_run_records": plan["repeats"], "observed_run_records": len(selected),
                    "repeats": [r["repeat"] for r in selected], "sources": [r["source"] for r in selected], "status": status})
    summaries = summarize(runs, (*METRICS, "population_inter_episode_gap_s"))
    control_summary = summarize(control_runs, F6_METRICS, ("phase",))
    f3, f4, f6 = table_rows(summaries, coverage, F3_METRICS), table_rows(summaries, coverage, F4_METRICS), table_rows(control_summary, coverage, F6_METRICS, phase=True)
    ar = {r["batch"]: r for r in f3 if r["role"] == "ar"}
    for row in f3:
        base = ar.get(row["batch"], {})
        for name, metric, inverse in (("effective_tpot_speedup_vs_ar", "effective_tpot_ms", True), ("population_throughput_ratio_vs_ar", "population_tokens_per_second", False)):
            left, right = row[metric + "_median"], base.get(metric + "_median")
            row[name] = (right / left if inverse else left / right) if left and right else None
    objectives = []
    if context:
        weights = {int(k): v for k, v in context["selection"]["normalized_batch_weights"].items()}
        for role in ROLES:
            cells = {r["batch"]: r for r in f3 if r["role"] == role}
            full = set(cells) == set(weights) and all(r["effective_tpot_ms_median"] is not None for r in cells.values())
            objectives.append({"role": role, "batch_weights": weights, "status": "measured" if full else "pending",
                "weighted_run_median_effective_tpot_ms": sum(weights[b] * cells[b]["effective_tpot_ms_median"] for b in weights) if full else None})
    status = "invalid_evidence" if any(e["status"] == "invalid" for e in issues) else "measured_comparison_complete" if coverage and all(c["status"] == "measured" for c in coverage) else "pending"
    result = {"schema_version": 1, "status": status, "preparation": str(Path(preparation_path).resolve()),
        "preparation_sha256": sha(context["manifest_path"]) if context else None,
        "domain": context["manifest"]["domain"] if context else None,
        "candidate_gammas": context["manifest"]["candidate_gammas"] if context else None,
        "protocol": {k: context["plan"][k] for k in ("batches", "repeats", "measured_requests", "output_tokens", "input_offset", "warmup_offset", "shape_warmup_offset", "admission_mode", "max_num_seqs")} if context else {},
        "execution_domain": context["evidence"]["domain"] if context else {},
        "environment": context["evidence"]["environment"] if context else {},
        "frozen_policies": {k: context["selection"][k] for k in ("static", "native_unrestricted", "native_threshold")} if context else {},
        "frozen_artifacts": {k: context["manifest"][k] for k in ("evaluation_config_sha256", "protocol_sha256", "calibration_sha256", "selection_sha256")} if context else {},
        "preparation_boundaries": context["manifest"].get("boundaries", []) if context else [],
        "selection_warmup_offset": context["manifest"]["selection_evidence"].get("warmup_offset") if context else None,
        "selection_shape_warmup_offset": context["manifest"]["selection_evidence"].get("shape_warmup_offset") if context else None,
        "coverage": coverage, "issues": issues, "runs": runs, "control_runs": control_runs,
        "F3": f3, "F4": f4, "F6": f6, "weighted_objective": objectives,
        "analysis_source_sha256": {name: sha(Path(__file__).parent / name) for name in (Path(__file__).name, "compare_controller_policies.py", "prepare_reduced_evaluation.py", "plot_results.py")},
        "boundaries": [
            "Measured workload records[64:128] are held out from measured development records[0:64]; source IDs need not equal record offsets. Actual selection warmup offsets are recorded separately; no overlap is inferred solely from a historical runner default.",
            "Each comparison cell requires both original repeat records. Raw values are retained; requests, steps, shared engines and continuous EWMA states are not independent system repeats. No confidence interval or best-repeat selection.",
            "F3 effective TPOT uses first-to-last output elapsed time per request and excludes its prefill. Complete-population throughput includes prefill, drain and inter-wave CPU/meter gaps.",
            "F4 wave-window energy and whole-device validated cumulative-counter span energy are separate. Counter-read endpoints include gaps but do not exactly coincide with population wall endpoints. Missing counter data remains unavailable.",
            "F6 reports all scheduled steps by phase, including terminal/drain. Native choose timing may be unrecorded, not zero. Wrapper CPU is nested in schedule instrumentation; do not add/subtract nested timings to infer GPU/host causal costs.",
            "Post-proposal truncation counts describe discarded available draft tokens, not measured wasted GPU time or energy. Frozen policy choices/selection scores are not executed observations.",
            "No synthetic measurements, figures, quality claim or new GPU execution. Incompatible evidence is excluded explicitly; missing observations remain pending."]}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("F3_latency_throughput", f3), ("F4_energy", f4), ("F6_control", f6), ("run_records", runs), ("control_records", control_runs), ("coverage", coverage), ("weighted_objective", objectives)):
        output_csv(output / (name + ".csv"), rows)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.preparation, args.output)
    print(json.dumps({"status": result["status"], "accepted_run_records": len(result["runs"]), "issues": len(result["issues"]), "output": args.output}))
    return 1 if result["status"] == "invalid_evidence" else 2 if result["status"] == "pending" else 0


if __name__ == "__main__":
    raise SystemExit(main())

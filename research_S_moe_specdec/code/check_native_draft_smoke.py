#!/usr/bin/env python3
"""CPU-only promotion gate for the existing gamma=0/3 native smoke runs.

  python code/check_native_draft_smoke.py --run runs/qwen3_awq_eagle3_smoke_r01 \
      --draft-source eagle3

Reads existing outputs only; writes an independent S/runs/.../report.json.
Exit 0 passes, exit 1 rejects. Zero accepted drafts is allowed. This checks
execution/record integrity, not performance, useful acceptance, or quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CRITICAL_SOURCE = ("run_matrix.py", "vllm_s_plugin.py", "cost_model.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_counts(record):
    """Extract existing raw counters; also usable on historical native records."""
    totals = Counter()
    errors = []
    for episode in record.get("episodes", [record]):
        plugin = episode["plugin"]
        steps = plugin["steps"]
        if len(episode["host_steps"]) != len(steps):
            errors.append("host/plugin step count differs")
        for step in steps:
            values = [step[name] for name in ("scheduled_drafts", "observed_drafts", "accepted_drafts")]
            if any(type(v) is not int or v < 0 for v in values) or not values[2] <= values[1] <= values[0]:
                errors.append(f"invalid draft counts at step {step.get('step')}: {values}")
            for name, value in zip(("scheduled_drafts", "observed_drafts", "accepted_drafts"), values):
                totals[name] += value
            totals["steps_with_observed_drafts"] += step["observed_drafts"] > 0
            totals["steps_with_draft_host_call"] += step.get("draft_host_call_ns", 0) > 0
            totals["draft_host_call_ns"] += step.get("draft_host_call_ns", 0)
        totals["engine_steps"] += len(steps)
        for name in ("scheduled_drafts", "accepted_drafts"):
            if plugin["summary"][name] != sum(s[name] for s in steps):
                errors.append(f"plugin summary disagrees with steps: {name}")
    return dict(totals), errors


def inspect(run, expected_source, report):
    errors = report["errors"]
    def require(ok, reason):
        if not ok:
            errors.append(reason)

    state = read(run / "status.json")
    plan = read(run / "plan.json")
    report.update(run=str(run), expected_draft_source=expected_source,
                  campaign_status=state.get("status"), plan_sha256=sha(run / "plan.json"), cells=[])
    require(state.get("status") == "complete", "campaign status is not complete")
    campaign_cells = state.get("cells", [])
    require(len(campaign_cells) == 2 and {c.get("gamma") for c in campaign_cells} == {0, 3},
            "promotion expects exactly two completed smoke gamma cells: 0 and 3")
    require(all(c.get("returncode") == 0 for c in campaign_cells), "a campaign child did not exit 0")
    require(plan.get("draft_source") == expected_source, "plan draft_source differs from expected source")
    controller_args = state.get("args", {})
    require(controller_args.get("mode", "native") == "native", "smoke must run native mode")
    require(controller_args.get("proposal_budget") is None,
            "two-gamma native smoke must use each gamma as its proposal budget")
    require(not controller_args.get("native_table"), "smoke must not use a dynamic native gamma table")
    require(not controller_args.get("natural_stop", False), "this smoke gate expects fixed-length outputs")
    batches = controller_args.get("batches") or plan["batches"]
    repeats = controller_args.get("repeats") or plan["repeats"]
    output_tokens = controller_args.get("output_tokens") or plan["output_tokens"]
    population = plan.get("measured_requests", 64)
    boundary = plan.get("semantic_token_boundary")
    report["expected_population"] = population
    report["expected_output_tokens_per_request"] = output_tokens
    report["semantic_token_boundary"] = boundary
    signatures, work_hashes = {}, set()
    gamma_totals = {0: Counter(), 3: Counter()}
    for gamma in (0, 3):
        folder = run / f"gamma{gamma}"
        require(read(folder / "status.json").get("status") == "complete", f"gamma{gamma}: status not complete")
        config = read(folder / "configuration.json")
        env = read(folder / "environment.json")
        args = read(folder / "engine_args.json")
        child = config["args"]
        require(child.get("gamma") == gamma and child.get("mode") == "native", f"gamma{gamma}: child gamma/mode mismatch")
        require(child.get("proposal_budget") in (None, gamma), f"gamma{gamma}: child proposal budget mismatch")
        require(not child.get("native_table"), f"gamma{gamma}: dynamic native table unexpectedly active")
        require(config["plan"] == plan, f"gamma{gamma}: child plan differs from campaign snapshot")
        work_hashes.add(config["workload_sha256"])
        require(args.get("model") == plan["model"], f"gamma{gamma}: wrong target model")
        require(env.get("model_runner_version") == "v1_explicit" and env.get("VLLM_USE_V2_MODEL_RUNNER") == "0",
                f"gamma{gamma}: explicit V1 runner evidence missing")
        source_checks = {}
        for name in CRITICAL_SOURCE:
            expected_hash = sha(run / "source" / name)
            source_checks[name] = env.get("source_sha256", {}).get(name) == expected_hash
            require(source_checks[name], f"gamma{gamma}: source hash differs from snapshot: {name}")
        spec = args.get("speculative_config")
        if gamma == 0:
            require(spec is None, "gamma0 must be target-only AR without a loaded speculative config")
        else:
            require(isinstance(spec, dict), "gamma3 lacks speculative engine config")
            if isinstance(spec, dict):
                require(spec.get("method") == expected_source and spec.get("num_speculative_tokens") == 3,
                        "gamma3 speculative source/production cap mismatch")
                require(not spec.get("num_speculative_tokens_per_batch_size"), "gamma3 native production table unexpectedly active")
                if expected_source in ("draft_model", "eagle3"):
                    require(bool(plan.get("draft_model")) and spec.get("model") == plan.get("draft_model"),
                            "gamma3 neural draft model does not match the plan")
                if expected_source == "eagle3" and plan.get("quantization"):
                    require("quantization" in spec and spec["quantization"] is None,
                            "quantized-target EAGLE must explicitly retain independent unquantized draft config")
        prompt_boundary_count = config.get("all_workload_prompt_tokens_at_or_above_boundary")
        if boundary is not None:
            require(type(prompt_boundary_count) is int and prompt_boundary_count == 0,
                    f"gamma{gamma}: semantic prompt boundary violated or existing count missing")
        cell = {"gamma": gamma, "engine_args_sha256": sha(folder / "engine_args.json"),
                "source_checks": source_checks, "all_workload_prompt_tokens_at_or_above_boundary": prompt_boundary_count,
                "performance_records": []}
        report["cells"].append(cell)
        expected_files = {folder / f"perf-r{rep}-b{batch}.json" for rep in range(repeats) for batch in batches}
        require(set(folder.glob("perf-*.json")) == expected_files, f"gamma{gamma}: performance file set differs from expected smoke cells")
        for path in sorted(expected_files):
            record = read(path)
            require(record.get("mode") == "native" and record.get("gamma") == gamma and record.get("proposal_budget") == gamma,
                    f"{path.name}: native current gamma/proposal budget mismatch")
            require(record.get("draft_source") == expected_source, f"{path.name}: recorded draft source mismatch")
            require(record.get("all_finished") is True and record.get("output_tokens") == population * output_tokens,
                    f"gamma{gamma}/{path.name}: incomplete fixed-length population")
            requests = record["requests"]
            ids = sorted(r["source_index"] for r in requests)
            require(len(requests) == population and len(set(ids)) == population,
                    f"gamma{gamma}/{path.name}: request population is missing or duplicated")
            require(all(r.get("finished") and len(r["output_token_ids"]) == output_tokens for r in requests),
                    f"gamma{gamma}/{path.name}: unfinished or shortened outputs")
            if path.name in signatures:
                require(ids == signatures[path.name], f"{path.name}: gamma0/3 prompt populations differ")
            else:
                signatures[path.name] = ids
            counts, counter_errors = record_counts(record)
            errors.extend(f"gamma{gamma}/{path.name}: {e}" for e in counter_errors)
            gamma_totals[gamma].update(counts)
            for episode in record.get("episodes", [record]):
                cfg = episode["plugin"]["config"]
                require(cfg.get("mode") == "native" and cfg.get("max_gamma") == gamma,
                        f"gamma{gamma}/{path.name}: runtime cap/mode mismatch")
            boundary_records = []
            if boundary is not None:
                for r in requests:
                    observed = r.get("output_tokens_at_or_above_semantic_boundary")
                    recomputed = sum(t >= boundary for t in r["output_token_ids"])
                    boundary_records.append({"request_id": r["request_id"], "source_index": r["source_index"],
                                             "recorded_count": observed, "recomputed_count": recomputed})
                    require(type(observed) is int and observed == recomputed == 0,
                            f"gamma{gamma}/{path.name}: semantic output boundary violated/missing for {r['request_id']}")
            cell["performance_records"].append({"path":str(path.relative_to(run)), "sha256":sha(path),
                "source_indices":ids, **counts, "semantic_output_boundary_counts":boundary_records})
    require(len(work_hashes) == 1, "gamma0 and gamma3 workload hashes differ")
    report["workload_sha256"] = sorted(work_hashes)
    report["draft_counts_by_gamma"] = {str(k):dict(v) for k,v in gamma_totals.items()}
    require(gamma_totals[0]["scheduled_drafts"] == gamma_totals[0]["observed_drafts"] == gamma_totals[0]["accepted_drafts"] == 0,
            "gamma0 unexpectedly scheduled, verified, or accepted draft tokens")
    require(gamma_totals[3]["scheduled_drafts"] > 0 and gamma_totals[3]["observed_drafts"] > 0,
            "gamma3 completed without any actual scheduled and verified draft tokens")
    report["zero_accepted_drafts"] = gamma_totals[3]["accepted_drafts"] == 0
    report["zero_acceptance_allowed"] = True
    report["nonzero_draft_gate_scope"] = "aggregate over all gamma3 measured smoke records; per-record counts retained; no acceptance threshold"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--draft-source", choices=("ngram", "ngram_gpu", "draft_model", "eagle3"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = (args.run if args.run.is_absolute() else PROJECT / args.run).resolve()
    if not run.is_relative_to((PROJECT / "runs").resolve()):
        parser.error("--run must remain under S/runs")
    output = args.output or Path("runs") / (run.name + "_promotion_check") / "report.json"
    output = (output if output.is_absolute() else PROJECT / output).resolve()
    if not output.is_relative_to((PROJECT / "runs").resolve()) or output.is_relative_to(run) or output.name != "report.json":
        parser.error("--output must be an independent S/runs/.../report.json, outside the raw run")
    report = {"schema_version":1, "checked_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(), "errors":[],
              "scope":"native loading/generation and exercised-draft evidence only; no performance, useful-acceptance, quality, or route-capture claim",
              "analysis_code_sha256":sha(Path(__file__))}
    try:
        inspect(run, args.draft_source, report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["errors"].append(f"required result could not be inspected: {type(exc).__name__}: {exc}")
    report["passed"] = not report["errors"]
    report["status"] = "passed" if report["passed"] else "rejected"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(output)
    print(json.dumps({"output":str(output), "status":report["status"], "errors":report["errors"],
                      "draft_counts_by_gamma":report.get("draft_counts_by_gamma")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

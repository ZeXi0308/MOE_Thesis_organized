#!/usr/bin/env python3
"""Full GSM8K with fourteen frozen controls; reuse the existing task harness.

Planning may validate the isolated new policy payload. Execution requires that
exact policy file already installed in root/code: it never deploys live code.
Only the main session may launch execution, directly under run_locked.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POPULATION = 1319
FILTERS = ("strict-match", "flexible-extract")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy_import(root, planning, sources):
    """Plan against payload; actual evaluator and worker must use deployed core."""
    live = (root / "code/expert_skip.py").resolve()
    payload = (root / sources["policy_payload"]["path"]).resolve()
    expected = sources["policy_payload"]["sha256"]
    actual = sha(live)
    ready = actual == expected
    require(planning or ready,
            "Policy is PENDING_DEPLOYMENT: wait for the DS chain to exit, then have the main session "
            "deploy the exact staged core. No GPU job was started by this entry.")
    selected = payload if planning else live
    require(sha(selected) == expected, "Selected policy payload SHA changed")
    if planning:
        spec = importlib.util.spec_from_file_location("expert_skip", selected)
        module = importlib.util.module_from_spec(spec)
        sys.modules["expert_skip"] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module("expert_skip")
    require(Path(module.__file__).resolve() == selected and sha(Path(module.__file__)) == expected,
            "Actual expert_skip import does not match the selected policy file")
    require(module.SkipConfig(mode="marginal", require_singleton=False).require_singleton is False,
            "Strict count-control schema is absent")
    return {"status": "DEPLOYED" if ready else "PENDING_DEPLOYMENT",
            "live_path": str(live), "live_sha256": actual, "required_sha256": expected,
            "planning_import_only": planning, "actual_import_path": str(selected)}


def verify_population(summary, folder, names):
    """Both extraction filters must cover every original test document per arm."""
    require(summary.get("status") == "COMPLETE", "Task summary is not complete")
    entries = summary.get("evaluations", [])
    require([r["arm"] for r in entries] == names, "Fourteen-arm sequence is incomplete or changed")
    for entry in entries:
        require(entry["task"] == "gsm8k" and entry["control_only"] is False,
                "Expected GSM8K generation, not a prefill identity control")
        for filter_name in FILTERS:
            metric = entry["sample_weighted_metrics"]["exact_match," + filter_name]
            require(metric["sample_rows"] == POPULATION, "An extraction filter lacks 1319 scored documents")
        result = json.loads((folder / entry["file"]).read_text())
        require(result["n-samples"] == {"gsm8k": {"original": POPULATION, "effective": POPULATION}}
                and result["n-shot"] == {"gsm8k": 5}, "Full population or official five shots changed")
        cfg = result["config"]
        require(cfg["limit"] is None and all(cfg[k] == 1701 for k in
                ("random_seed", "numpy_seed", "torch_seed", "fewshot_seed")), "Limit or seed changed")
        require(cfg["gen_kwargs"]["max_gen_toks"] == 256 and cfg["batch_size"] == 8,
                "Generation cap or batch changed")
        samples = result["samples"]["gsm8k"]
        require(len(samples) == 2 * POPULATION, "Unexpected raw sample/filter population")
        for filter_name in FILTERS:
            selected = [row for row in samples if row["filter"] == filter_name]
            require(len(selected) == POPULATION and {r["doc_id"] for r in selected} == set(range(POPULATION)),
                    "Missing, repeated, or unexpected original test IDs")
    raw = Counter()
    for line in (folder / "raw_outputs.jsonl").open():
        row = json.loads(line)
        require(row["generate"] is True, "Unexpected non-generation task output")
        raw[row["arm"]] += 1
    require(raw == Counter({"gsm8k." + name: POPULATION for name in names}),
            "Raw generated completion counts do not cover the entire arm population")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/moe-research-20261002/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-run-dir", type=Path)
    parser.add_argument("--budget-seconds", type=int, default=7080)
    parser.add_argument("--job-timeout", type=int, default=7070)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.budget_seconds > 30 and 0 < args.job_timeout <= args.budget_seconds - 5,
            "Positive job timeout must leave at least five seconds inside the suite budget")
    args.root = args.root.expanduser().resolve()
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Execution's direct script entry and any worker imports share this live code.
    # Do not inherit a policy-only PYTHONPATH overlay from a planning shell.
    os.environ["PYTHONPATH"] = str(args.root / "code")
    sys.path.insert(0, str(args.root / "code"))
    sources = json.loads((HERE / "sources.json").read_text())
    import_state = policy_import(args.root, args.plan_only, sources)
    import run_task_quality as runner

    for key in ("model", "output", "outer_run_dir"):
        if getattr(args, key) is not None:
            setattr(args, key, runner.resolved(getattr(args, key), args.root))
    args.data_dir = args.root / "datasets/task_eval_r02"
    args.prompts = args.root / "datasets/free_generation_validation_r01/prompts.jsonl"
    args.arms_file = args.root / sources["arms_file"]["path"]
    require(sha(args.arms_file) == sources["arms_file"]["sha256"], "Fourteen-arm source changed")
    args.formats = ["bf16"]; args.tasks = ["gsm8k"]; args.limit = None
    args.free_limit = 200; args.prompt_tokens = 128; args.batch = 8
    args.max_tokens = 256; args.max_model_len = 4096; args.seed = 1701
    args.gpu_memory_utilization = .35; args.per_arm_isolation = False
    lengths_path = args.data_dir / "context_lengths_olmoe_full.json"
    require(sha(lengths_path) == sources["full_context_input"]["sha256"], "Full-context input metadata changed")
    lengths = json.loads(lengths_path.read_text())
    protocol, checked = lengths["protocol"], lengths["families"]["gsm8k"]
    require(lengths["status"] == "FULL_INPUT_LENGTHS_CHECKED"
            and protocol["limit_per_task_leaf"] is None and protocol["seed"] == 1701
            and protocol["shots"]["gsm8k"] == 5 and protocol["generation_max_tokens"] == 256
            and protocol["max_model_len"] == 4096 and not protocol["apply_chat_template"]
            and checked["documents"] == checked["requests"] == POPULATION and checked["over_requests"] == 0,
            "Full 1319-question input check does not match this protocol")
    arms, provenance, warnings = runner.inputs(args)
    names = [row["name"] for row in arms["arms"]]
    require(len(names) == len(set(names)) == 14 and names == sources["arm_names"], "Expected frozen fourteen-arm set")
    require(provenance["dataset_source"]["sha256"] == lengths["dataset_source"]["sha256"], "Task source changed")
    if "model_config" in provenance:
        require(provenance["model_config"]["sha256"] == lengths["tokenizer"]["files"]["config.json"]["sha256"],
                "This entry requires the checked OLMoE checkpoint")
    if import_state["status"] != "DEPLOYED":
        warnings.append("New policy is NOT deployed: CPU plan uses staged payload; execution refuses old live core")
    provenance.update(policy_import=import_state,
        full_context_lengths={"path": str(lengths_path), "sha256": sha(lengths_path)},
        deployment_entry={"path": str(Path(__file__)), "sha256": sha(Path(__file__))},
        reused_code={name: sha(args.root / "code" / name) for name in
                     ("run_task_quality.py", "eval_tasks.py", "free_generate.py", "skip_worker.py")})
    plan = runner.build_plan(args, arms, provenance, warnings)
    plan["jobs"] = [job for job in plan["jobs"] if job["unit"] == "gsm8k"]
    require(len(plan["jobs"]) == 1 and "--limit" not in plan["jobs"][0]["command"], "Unexpected task plan")
    plan.update(expected_documents_per_arm=POPULATION, exact_selected_arms=names,
                total_planned_completions=POPULATION * 14, full_fourteen_arm_matrix=True)
    plan["semantics"].update(
        generation="Full official GSM8K 5-shot; 1319 test questions/arm; seed1701; greedy; no chat template; natural EOS/task stop; cap256; first prediction is prefill.",
        limit="No limit or document filtering. Both extraction filters must cover all 1319 original IDs per arm.",
        native_reuse="One model load and one full native baseline shared by all fourteen arms. Existing harness checks native/candidate document/prompt/target identity and native/dense response identity.",
        policy_import="Execution requires the exact deployed root/code core. Child executes root/code/eval_tasks.py; inherited PYTHONPATH is root/code, MP=0. Planning-only staged import is never used as evidence of worker deployment.",
        failure="Explicit job/suite/outer deadlines. Preserve partial raw files and complete accepted arms; incomplete arms have no full accuracy. No automatic retries, arm omission, population reduction, or budget extension.",
        statistics="One paired task run; strict/flexible extraction stay separate. Request stderr is not independent-run uncertainty.")
    if not args.plan_only:
        plan["outer_run"] = runner.verify_outer(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arms").mkdir()
    runner.save(args.output / "arms/all_arms.json", plan["arm_files"]["all_arms"])
    runner.save(args.output / "suite.json", plan)
    print(json.dumps({"status": "PLANNED", "jobs": 1, "arms": names, "questions_per_arm": POPULATION,
                      "policy_import": import_state["status"], "warnings": warnings}), flush=True)
    if args.plan_only:
        return 0
    code = runner.execute(args, plan)
    if code == 0:
        job = plan["jobs"][0]; folder = Path(job["output"])
        try:
            verify_population(json.loads((folder / "summary.json").read_text()), folder, names)
            plan["full_population_verified"] = True
        except (OSError, ValueError, KeyError) as error:
            plan["status"] = "INCOMPLETE"; code = 1
            job.update(status="FAILED", error=f"Full-population validation: {error}")
        runner.save(args.output / "suite.json", plan)
    print(json.dumps({"status": plan["status"], "manifest": str(args.output / "suite.json")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

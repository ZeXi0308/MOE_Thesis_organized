"""Validate the frozen block/token model on context-calibration executions."""
import argparse
import hashlib
import importlib.util
import json
from math import ceil
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
OUTS = ROOT / "outputs/admission_capacity"
BUNDLE = OUTS / "20260914_context_victim_calibration_r01"
RESULTS = BUNDLE / "execution/readback/results"
MODEL_PATH = Path(__file__).with_name("model_source.py")
CONTEXT_SELECTOR = BUNDLE / "execution/readback/pkg/absence_rotation.py"
FUNDING_SELECTOR = (OUTS / "20260914_funding_filter_comparison_r01/preparation/pkg"
                    / "absence_rotation.py")
CELLS = (("context-block0-most_output", "continuous_most"),
         ("context-block0-least_progress", "least"),
         ("context-block1-most_output", "continuous_most"),
         ("context-block1-least_progress", "least"))


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_input(cell, raw, decisions):
    memory = {row["attempted_step"]: row for row in raw["memory_trace"]}
    prior_forced = False
    for decision in decisions:
        before = memory[decision["step"]]["before"]
        running = before["running_ids"]
        pure = all(before["requests"][rid]["output_tokens"] > 0 and
                   before["requests"][rid]["computed_tokens"] ==
                   before["requests"][rid]["prompt_tokens"] +
                   before["requests"][rid]["output_tokens"] - 1 for rid in running)
        if (decision["active"] and len(running) == len(before["requests"]) == 32
                and before["waiting_count"] == 0 and pure and not prior_forced):
            step = decision["step"]
            break
        prior_forced |= bool(decision["forced_preempted"])
    else:
        raise ValueError(f"{cell}: no qualified all-32 pure before-state")
    config = load(RESULTS / cell / "config.json")
    requests = {rid: dict(blocks=[0] * sum(row["block_counts"]),
                    prompt=row["prompt_tokens"], output=row["output_tokens"],
                    computed=row["computed_tokens"], max_tokens=config["output_tokens"],
                    preemptions=row["num_preemptions"], status="RUNNING")
                for rid, row in before["requests"].items()}
    history = [dict(step=d["step"], preempted=d["preempted"], resumed=d["resumed"])
               for d in decisions if d["step"] < step and (d["preempted"] or d["resumed"])]
    initial = dict(requests=requests, running=list(running), waiting=[],
                   free=before["pool"]["free_blocks"])
    qualified = dict(step=step, active=True, requests=len(requests), running=len(running),
                     waiting=0, pure_running=len(running), free_blocks=initial["free"],
                     prior_event_history=history, prior_forced_rotations=0)
    return initial, history, qualified, config, memory


def recomputed(prediction, initial):
    total = 0
    for row in prediction["trace"]:
        for rid, scheduled in row["scheduled"].items():
            high_water = initial["requests"][rid]["prompt"] + scheduled["output"] - 1
            total += min(scheduled["tokens"], max(0, high_water - scheduled["computed"]))
    return total


def summarize(prediction, initial):
    return dict(status=prediction["status"], modeled_calls_from_start=len(prediction["trace"]),
                last_step=prediction["last_step"], forced_rotations=sum(
                    row["forced"] is not None for row in prediction["trace"]),
                total_preemptions=sum(len(row["preempted"]) for row in prediction["trace"]),
                total_resumes=sum(len(row["resumed"]) for row in prediction["trace"]),
                recomputed_positions=recomputed(prediction, initial),
                completion_steps=prediction["completed"])


def first_mismatch(prediction, raw, decisions, memory, initial):
    steps = {row["step"]: row for row in raw["scheduler_steps"]}
    calls = {row["scheduler_step_start"]: row for row in raw["engine_steps"]}
    ext2int = {row["external_request_id"]: row["internal_request_id"]
               for row in raw["requests"]}
    completion = {row["internal_request_id"]: next(
        call["scheduler_step_start"] for call in raw["engine_steps"]
        if call["returned_s"] == row["completion_s"]) for row in raw["requests"]}
    compared = 0
    for modeled in prediction["trace"]:
        step = modeled["step"]
        if step not in steps:
            return dict(step=step, field="actual_trace_ended"), compared, completion
        actual = steps[step]
        scheduled = {r["internal_request_id"]: dict(computed=r["scheduled_start_computed"],
                     tokens=r["scheduled_tokens"], output=r["output_tokens_before"])
                     for r in actual["scheduled"]}
        outputs = [ext2int[rid] for rid in calls[step]["output_request_ids"]]
        completed = [rid for rid in outputs if completion[rid] == step]
        forced = decisions[step]["forced_preempted"]
        if len(forced) > 1:
            raise ValueError("multiple forced victims in one call")
        checks = (("free_before", modeled["free_before"], memory[step]["before"]["pool"]["free_blocks"]),
                  ("schedule", list(modeled["scheduled"].items()), list(scheduled.items())),
                  ("free_after_schedule", modeled["free_after_schedule"], memory[step]["after"]["pool"]["free_blocks"]),
                  ("forced", modeled["forced"], forced[0] if forced else None),
                  ("preempted", modeled["preempted"], decisions[step]["preempted"]),
                  ("resumed", modeled["resumed"], decisions[step]["resumed"]),
                  ("outputs", modeled["outputs"], outputs), ("completed", modeled["completed"], completed))
        for field, got, expected in checks:
            if got != expected:
                return dict(step=step, field=field, modeled=got, actual=expected), compared, completion
        compared += 1
    return None, compared, completion


def state_before(initial, trace, stop):
    state = {rid: dict(allocated=len(r["blocks"]), computed=r["computed"], output=r["output"],
                       prompt=r["prompt"]) for rid, r in initial["requests"].items()}
    for row in trace:
        if row["step"] >= stop:
            break
        for rid in row["preempted"]:
            state[rid]["allocated"] = 0; state[rid]["computed"] = 0
        for rid, scheduled in row["scheduled"].items():
            state[rid]["allocated"] = max(state[rid]["allocated"], ceil(
                (scheduled["computed"] + scheduled["tokens"]) / 16))
            state[rid]["computed"] = scheduled["computed"] + scheduled["tokens"]
            if state[rid]["computed"] == state[rid]["prompt"] + state[rid]["output"]:
                state[rid]["output"] += 1
        for rid in row["completed"]:
            del state[rid]
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else BUNDLE / args.output
    model = load_module("context_progress_model", MODEL_PATH)
    selector = load_module("context_frozen_selector", CONTEXT_SELECTOR)
    inputs = {}; validations = []
    for cell, action in CELLS:
        raw = load(RESULTS / cell / "raw.json"); decisions = load(RESULTS / cell / "headroom-decisions.json")
        initial, history, qualified, config, memory = build_input(cell, raw, decisions)
        prediction = model.simulate(initial, history, action, selector.AbsenceRotation,
                                    selector.RequestView, start_step=qualified["step"],
                                    max_steps=6000)
        mismatch, compared, completions = first_mismatch(prediction, raw, decisions, memory, initial)
        summary = summarize(prediction, initial)
        summary.update(cell=cell, action=action, qualified_initial=qualified,
                       actual_last_step=raw["scheduler_steps"][-1]["step"],
                       actual_recomputed_positions=sum(r["recompute_tokens"] for r in raw["scheduler_steps"]),
                       compared_steps=compared, first_mismatch=mismatch,
                       completion_steps_equal=prediction["completed"] == completions,
                       exact=mismatch is None and prediction["completed"] == completions)
        validations.append(summary); inputs[cell] = (initial, history, qualified)
    conditional = []
    adapter = None
    if all(row["exact"] for row in validations):
        funding = load_module("funding_frozen_selector", FUNDING_SELECTOR)
        source = MODEL_PATH.read_text()
        old = "proposal=tracker.decide(step,rows,waiting,free,{r:need(states[r]) for r in waiting})"
        new = old[:-1] + ",released_blocks={r:states[r]['allocated'] for r in running})"
        if source.count(old) != 1:
            raise ValueError("model adapter anchor is not unique")
        adapted_source = source.replace(old, new)
        adapted = types.ModuleType("funding_adapted_progress_model")
        exec(compile(adapted_source, "<funding-adapted-model>", "exec"), adapted.__dict__)
        adapter = dict(kind="single_callsite_source_adapter", original=old, replacement=new,
                       adapted_source_sha256=hashlib.sha256(adapted_source.encode()).hexdigest())
        for cell in ("context-block0-least_progress", "context-block1-least_progress"):
            initial, history, qualified = inputs[cell]; base_log = []; filtered_log = []
            kwargs = dict(start_step=qualified["step"], max_steps=6000)
            base = model.simulate(initial, history, "least", funding.AbsenceRotation,
                                  funding.RequestView, decision_log=base_log, **kwargs)
            filtered = adapted.simulate(initial, history, "least", funding.AbsenceRotation,
                                        funding.RequestView, decision_log=filtered_log, **kwargs)
            divergence = next(row[0]["step"] for row in zip(base["trace"], filtered["trace"])
                              if any(row[0][k] != row[1][k] for k in
                                     ("scheduled", "free_after_schedule", "outputs", "completed",
                                      "preempted", "resumed", "forced")))
            before = state_before(initial, base["trace"], divergence)
            bdec = next(row for row in base_log if row["step"] == divergence)
            fdec = next(row for row in filtered_log if row["step"] == divergence)
            target = bdec["target"]
            first_output = {name: next(row["step"] for row in prediction["trace"]
                            if row["step"] >= divergence and target in row["outputs"])
                            for name, prediction in (("least_progress", base),
                                                     ("least_feasible", filtered))}
            changed = sorted(set(base["trace"][divergence-qualified["step"]]["scheduled"]) ^
                             set(filtered["trace"][divergence-qualified["step"]]["scheduled"]))
            conditional.append(dict(cell=cell, status="CONDITIONAL_STRUCTURAL_PREDICTION",
                first_divergence=dict(step=divergence, target=target, free_blocks=bdec["free"],
                    target_need_blocks=bdec["target_need"], least_selected_victim=bdec["default_victim"],
                    least_selected_released_blocks=before[bdec["default_victim"]]["allocated"],
                    least_selected_fundable=bdec["default_victim"] in bdec["fundable"],
                    feasible_selected_victim=fdec["default_victim"],
                    feasible_selected_released_blocks=before[fdec["default_victim"]]["allocated"],
                    feasible_selected_fundable=fdec["default_victim"] in fdec["fundable"],
                    fundable_candidates=len(fdec["fundable"]),
                    target_first_new_output_step=first_output,
                    changed_scheduled_request_ids=changed),
                least_progress=summarize(base, initial), least_feasible=summarize(filtered, initial)))
    result = dict(status="COMPLETE", evidence_type="CPU_STRUCTURAL_MODEL_VALIDATION",
        source=dict(model_source=str(MODEL_PATH), model_sha256=sha(MODEL_PATH),
                    context_selector_sha256=sha(CONTEXT_SELECTOR), funding_selector_sha256=sha(FUNDING_SELECTOR)),
        validation=validations, all_four_exact=all(row["exact"] for row in validations),
        adapter=adapter, conditional_predictions=conditional,
        constraints=dict(saved_prefixes_passed=False, observed_load_completions_passed=False,
                         future_scheduler_events_used_to_advance_model=False,
                         observed_futures_used_as_validation_labels=True,
                         simulation_horizon=dict(value=6000, source="predeclared_constant")),
        correction=dict(r01_issue="validation horizon used observed last scheduler step plus two",
                        effect="horizon dependency only; all four trajectories completed before the bound",
                        r02_fix="validation and conditional simulations both use predeclared max_steps=6000",
                        gpu_raw_modified=False),
        scope=("Observed futures are validation labels only. Conditional least_feasible rows are independent "
               "token/block trajectories, not GPU observations, an Oracle, KV tensor equality, wall time, or latency ranking."))
    with output.open("x") as handle:
        json.dump(result, handle, indent=2); handle.write("\n")
    print(json.dumps(dict(output=str(output), all_four_exact=result["all_four_exact"],
                          validation=[(r["cell"], r["first_mismatch"]) for r in validations],
                          conditional=[r["first_divergence"] for r in conditional]), indent=2))


if __name__ == "__main__":
    main()

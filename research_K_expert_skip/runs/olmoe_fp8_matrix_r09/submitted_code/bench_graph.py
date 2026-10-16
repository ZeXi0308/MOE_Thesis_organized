"""Fixed-arm graph benchmark with independent process repeats and accounting.

Must run under run_locked.py. Every arm/repeat starts its own performance
process and a separate counted process. Only equal output-token hashes allow
joining counters to the uninstrumented policy performance. The merged output
uses bench_decode's figure schema; child timings are retained but not pooled.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from bench_decode import append_json, write_json


def action_spec(spec):
    """Only native-k's accounting toggle may differ between paired processes."""
    result = copy.deepcopy(spec)
    if result is not None and result.get("mode") == "global_native":
        result.pop("count_routes", None)
    return result


def pair_rows(performance, counted):
    by_key = {(r["repeat"], r["batch"], r["arm"]): r for r in counted}
    performance_keys = {(r["repeat"], r["batch"], r["arm"]) for r in performance}
    if (len(by_key) != len(counted) or len(performance_keys) != len(performance)
            or performance_keys != set(by_key)):
        raise ValueError("duplicate or incomplete graph counter cells")
    paired = []
    for row in performance:
        key = (row["repeat"], row["batch"], row["arm"])
        counter = by_key.pop(key)
        if (row.get("execution") != "graph" or counter.get("execution") != "graph"
                or row.get("graph_counter_pass") is not False
                or counter.get("graph_counter_pass") is not True):
            raise ValueError(f"expected uncounted/counted graph process pair: {key}")
        if action_spec(row["spec"]) != action_spec(counter["spec"]):
            raise ValueError(f"graph counting changed action spec: {key}")
        if (row["spec"] or {}).get("mode") == "global_native":
            if row["spec"].get("count_routes") is not False or counter["spec"].get("count_routes") is not True:
                raise ValueError(f"native-k count_routes disagrees with process role: {key}")
        if row["output_tokens_sha256"] != counter["output_tokens_sha256"]:
            raise RuntimeError(f"graph counting changed output tokens: {key}")
        if row["decode_output_tokens"] != counter["decode_output_tokens"]:
            raise RuntimeError(f"graph counter workload mismatch: {key}")
        stats = counter["logical_counters"]
        if not stats or stats.get("events", 0) <= 0:
            raise RuntimeError(f"graph counter process did not observe decode: {key}")
        result = dict(row)
        result.update(logical_counters=stats,
            logical_counter_source="separate_graph_process_identical_output_tokens",
            counter_identity_verified=True,
            counter_spec=copy.deepcopy(counter["spec"]))
        paired.append(result)
    if by_key:
        raise ValueError("unused graph counter cells")
    return paired


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def validate_snapshot(row, snapshot, counted):
    """Detect capture-reset/padding mistakes before joining logical bytes."""
    graph = snapshot["graph"]
    batch, tokens = row["batch"], row["decode_output_tokens"]
    if tokens <= 0 or tokens % batch:
        raise ValueError("graph workload must have a complete fixed decode cohort")
    steps = tokens // batch
    runtime = [r for r in graph["runtime_rows"] if r["phase"] == "decode"]
    if (graph.get("count_routes") is not counted or not graph.get("fixed_arm")
            or graph.get("cudagraph_mode") != "FULL_DECODE_ONLY"
            or graph.get("compilation_mode") != 0
            or snapshot["spec"] != row["spec"]):
        raise RuntimeError("graph snapshot settings disagree with measured process")
    if (len(runtime) != steps or snapshot["phase_steps"].get("decode") != steps
            or batch not in graph["captured_token_buckets"]
            or any(r["runtime_mode"] != "FULL" or r["actual_tokens"] != batch
                   or r["padded_tokens"] != batch for r in runtime)):
        raise RuntimeError("every measured decode must replay its captured unpadded FULL graph")
    layers = snapshot["execution"]
    if not layers:
        raise RuntimeError("graph snapshot has no installed MoE layers")
    if any(m["current_router_top_k"] != m["original_top_k"] for m in layers.values()):
        raise RuntimeError("native router top_k was not restored")
    stats = snapshot.get("layers", {}).get("totals")
    if counted:
        if (not stats or stats.get("events") != len(layers) * steps
                or stats.get("tokens") != len(layers) * tokens):
            raise RuntimeError("graph counters do not cover every real layer/token/step exactly once")
    elif stats and (stats.get("events", 0) or stats.get("tokens", 0)):
        raise RuntimeError("performance process unexpectedly collected route counters")


def read_checked_rows(folder, counted):
    status = json.loads((folder / "status.json").read_text())
    if status.get("status") != "COMPLETE":
        raise RuntimeError(f"incomplete graph child: {folder}")
    rows = read_rows(folder / "measurements.jsonl")
    for row in rows:
        cell = folder / row["folder"]
        snapshot = json.loads((cell / "counters.json").read_text())
        validate_snapshot(row, snapshot, counted)
        if row["logical_counters"] != snapshot.get("layers", {}).get("totals"):
            raise RuntimeError("measurement counters disagree with saved snapshot")
        energy_file = cell / "energy.json"
        if energy_file.exists():
            energy = json.loads(energy_file.read_text())
            token_hash = hashlib.sha256(json.dumps([r["token_ids"] for r in energy["requests"]],
                separators=(",", ":")).encode()).hexdigest()
            if token_hash != row["output_tokens_sha256"]:
                raise RuntimeError("energy replay changed output tokens")
            validate_snapshot(row, json.loads((cell / "energy_counters.json").read_text()), counted)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arms-file", type=Path, required=True)
    p.add_argument("--arms", nargs="+")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--newtokens", type=int, default=512)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--prompt-offset", type=int, default=0)
    p.add_argument("--warmup-seconds", type=float, default=3.)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=.35)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantization")
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    if (args.repeats < 1 or len(set(args.batches)) != len(args.batches)
            or any(b not in (1, 2, 4, 8, 16, 32, 64) for b in args.batches)):
        raise ValueError("positive repeats and the seven unpadded matrix batches are required")
    arms = json.loads(args.arms_file.read_text())
    if len({a["name"] for a in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    if args.arms:
        wanted = set(args.arms)
        arms = [a for a in arms if a["name"] in wanted]
        if wanted != {a["name"] for a in arms}:
            raise ValueError("unknown arm")
    if not arms or "native" not in {a["name"] for a in arms}:
        raise ValueError("every graph comparison must include native")
    if next(a for a in arms if a["name"] == "native")["spec"] is not None:
        raise ValueError("native reference must have spec=null")
    cells = []
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        order = list(arms)
        rng.shuffle(order)
        cells.extend((repeat, i, a["name"]) for i, a in enumerate(order))
    plan = {"model": args.model, "process_order": cells, "batches": args.batches,
            "processes": len(cells) * 2, "performance_cells": len(cells) * len(args.batches),
            "counter_identity": "exact generated token hash equality per arm/batch/repeat",
            "repetition_unit": "fresh LLM process per arm per repeat; randomized arm order",
            "timing": "counter-disabled FULL decode graph, including real-token scalar update and runtime metadata",
            "energy": "separate pure-decode NVML integral; windows below 3 s remain diagnostic"}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", plan)
    state = {"status": "RUNNING", "started_unix": time.time(), "completed_cells": 0}
    write_json(out / "status.json", state)
    common = [sys.executable, str(Path(__file__).with_name("bench_decode.py").resolve()),
        "--model", args.model, "--prompts", str(args.prompts.resolve()),
        "--arms-file", str(args.arms_file.resolve()), "--batches", *map(str, args.batches),
        "--repeats", "1", "--execution", "graph", "--backend", "triton",
        "--newtokens", str(args.newtokens), "--prompt-tokens", str(args.prompt_tokens),
        "--prompt-offset", str(args.prompt_offset), "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype, "--seed", str(args.seed)]
    if args.quantization:
        common += ["--quantization", args.quantization]
    try:
        for repeat, order, arm in cells:
            base = out / f"repeat{repeat:03d}"
            base.mkdir(exist_ok=True)
            perf, count = base / f"{arm}_performance", base / f"{arm}_counters"
            commands = [common + ["--arms", arm, "--output-dir", str(perf),
                         "--warmup-seconds", str(args.warmup_seconds)] + (["--no-energy"] if args.no_energy else []),
                        common + ["--arms", arm, "--output-dir", str(count),
                         "--graph-counter-pass", "--no-energy", "--warmup-seconds", "0"]]
            for name, command in zip(("performance", "counters"), commands):
                write_json(base / f"{arm}_{name}_command.json", command)
                with (base / f"{arm}_{name}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not (out / "config.json").exists():
                config = json.loads((perf / "config.json").read_text())
                config["args"].update({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
                config["args"].update(execution="graph", graph_counter_pass=False)
                config["arms"] = copy.deepcopy(arms)
                config["process_protocol"] = plan
                write_json(out / "config.json", config)
                device = perf / "power_device.json"
                if device.exists():
                    write_json(out / "power_device.json", json.loads(device.read_text()))
            pairs = pair_rows(read_checked_rows(perf, False), read_checked_rows(count, True))
            for row in pairs:
                row.update(repeat=repeat, order_index=order,
                    folder=str((perf / row["folder"]).relative_to(out)),
                    counter_process_folder=str(count.relative_to(out)))
                append_json(out / "measurements.jsonl", row)
                state["completed_cells"] += 1
            write_json(out / "status.json", state)
            print(json.dumps({"repeat": repeat, "arm": arm, "completed_cells": state["completed_cells"]}), flush=True)
        state.update(status="COMPLETE", completed_unix=time.time())
    except BaseException as exc:
        state.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}", completed_unix=time.time())
        raise
    finally:
        write_json(out / "status.json", state)


if __name__ == "__main__":
    main()

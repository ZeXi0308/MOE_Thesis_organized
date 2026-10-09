"""HISTORICAL SYNCHRONOUS COMPONENT REFERENCE; not the revised async branch.

Whole-trace encoding comparison on captured, genuine top-20 candidates.

This is an UNPACED encoding-stage benchmark, not HTTP/client delivery or model
co-run evidence. Natural batch boundaries, order, candidate values, and each
request's UTF-8 context are preserved. Nothing fabricates a missing capture.

CPU-only (explicitly excludes compact-array transfer):
  python bench_real.py --capture capture --output bench_cpu --cpu-only

GPU (must be started by the actual gpu_guard parent, with its record path):
  python gpu_guard.py --record bench_gpu.guard.json --seconds 300 -- \
    python bench_real.py --capture capture --output bench_gpu \
    --guard-record bench_gpu.guard.json

The stage begins at an ALREADY CONTIGUOUS eligible raw batch. Each natural
batch's real heavy rows are packed in their original order once as a common
fixture, then uploaded once per array; host packing and upload time/bytes are
reported separately. Native and CPU use the same three compact-array D2H
copies, with only required dtype conversion. GPU consumes the same resident
batch directly, charging expanded D2H, sampled-ID/probability header D2H, raw
fallback-row gather/D2H, and native fallback/finishing. CPU-only starts at the
same host-packed input. Common eligible-row selection and actual sampler
packaging are outside this stage boundary and MUST be charged in model co-run
evaluation. Batch sizes, candidate content and order are never changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_guard(record):
    if record is None:
        raise RuntimeError("GPU mode requires --guard-record from an outer gpu_guard.py invocation")
    data = json.loads(Path(record).read_text())
    if data.get("guard_pid") != os.getppid() or data.get("child_pid") != os.getpid():
        raise RuntimeError("guard record does not identify this process and its live parent")
    os.kill(data["guard_pid"], 0)
    if data.get("ended_unix") is not None:
        raise RuntimeError("guard record is already terminal")
    if data.get("lock") != "/root/autodl-tmp/moe-research-gpu.lock":
        raise RuntimeError("unexpected GPU guard lock")
    # Verify that someone actually holds the expected lock; the parent PID/child
    # record above ties this execution to gpu_guard, rather than a stale file.
    with open(data["lock"], "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise RuntimeError("GPU guard record exists but the lock is not held")
    return data


def read_capture(directory, np):
    trace_path, arrays_path = directory / "trace.json", directory / "candidates.npz"
    for path in (trace_path, arrays_path):
        if not path.is_file():
            raise FileNotFoundError(f"real capture missing: {path}; run capture_real.py under gpu_guard first")
    trace = json.loads(trace_path.read_text())
    requests = {r["request_id"]: r for r in trace["requests"]}
    if not requests or not all(r.get("finished") for r in requests.values()):
        raise ValueError("capture contains no requests or unfinished requests")
    cursor = {rid: 0 for rid in requests}
    batches, total_rows = [], 0
    previous_time = -1.0
    with np.load(arrays_path, allow_pickle=False) as archive:
        for index, batch in enumerate(trace["batches"]):
            when = float(batch["time_s"])
            if when < previous_time:
                raise ValueError("capture batches are not in their original time order")
            previous_time = when
            groups = []
            for item in batch["outputs"]:
                rid = item["request_id"]
                req = requests[rid]
                start, end = int(item["token_start"]), int(item["token_end"])
                if start != cursor[rid] or end < start:
                    raise ValueError(f"noncontiguous captured output: {rid}:{start}:{end}")
                expected = req["output_token_ids"][start:end]
                if item["new_token_ids"] != expected:
                    raise ValueError("batch tokens differ from saved complete request")
                cursor[rid] = end
                key = item.get("logprobs_key")
                if not key:
                    if req.get("heavy") and end > start:
                        raise ValueError("a heavy output is missing real logprobs")
                    continue
                ids = np.array(archive[key + "_ids"], copy=True, order="C")
                values = np.array(archive[key + "_values"], copy=True, order="C")
                ranks = np.array(archive[key + "_ranks"], copy=True, order="C").reshape(-1)
                if ids.dtype not in (np.dtype("int32"), np.dtype("int64")):
                    raise TypeError("capture IDs must retain int32/int64 dtype")
                if values.dtype != np.float32 or ids.shape != values.shape:
                    raise TypeError("capture logprobs must be float32 and match IDs")
                if ids.ndim != 2 or ids.shape != (end - start, 21) or ranks.shape != (end - start,):
                    raise ValueError("this prototype requires real sampled+top20 rows")
                if ids[:, 0].tolist() != expected:
                    raise ValueError("sampled logprob IDs differ from generated tokens")
                if ranks.dtype not in (np.dtype("int32"), np.dtype("int64")):
                    raise TypeError("unexpected captured rank dtype")
                if not req.get("heavy"):
                    raise ValueError("light request unexpectedly has logprobs")
                if len(ids):
                    groups.append({"rid": rid, "start": start, "rows": len(ids),
                                   "ids": ids, "values": values, "ranks": ranks})
                    total_rows += len(ids)
            batches.append({"index": index, "time_s": when, "groups": groups,
                            "rows": sum(g["rows"] for g in groups)})
    for rid, req in requests.items():
        if cursor[rid] != len(req["output_token_ids"]):
            raise ValueError(f"incomplete capture for {rid}")
    if not total_rows:
        raise ValueError("capture has no real top-20 rows")
    return requests, batches, {
        "trace_sha256": sha(trace_path), "candidates_sha256": sha(arrays_path),
        "requests": len(requests), "heavy_requests": sum(bool(r.get("heavy")) for r in requests.values()),
        "heavy_rows": total_rows, "natural_batches": len(batches),
        "actual_heavy_rows_per_batch": dict(sorted(Counter(b["rows"] for b in batches).items())),
        "capture_elapsed_s": trace.get("elapsed_s"),
        "capture_cpu_wall_overhead_ns": trace.get("capture_cpu_wall_overhead_ns"),
    }


class NativeMethods:
    """Actual installed vLLM logprob materialization and response serializer."""
    def __init__(self, tokenizer, requests):
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.v1.engine.logprobs import LogprobsProcessor
        from vllm.v1.outputs import LogprobsLists
        from frontend_bridge import DirectFrontend
        self.Lists = LogprobsLists
        self.tokenizer = tokenizer
        self.chat = OpenAIServingChat.__new__(OpenAIServingChat)
        self.chat.return_tokens_as_token_ids = False
        self.states = {
            rid: LogprobsProcessor(tokenizer=tokenizer, logprobs=[], prompt_logprobs=None,
                                   cumulative_logprob=0.0, num_logprobs=20, num_prompt_logprobs=None)
            for rid, req in requests.items() if req.get("heavy")
        }
        # Reuse the production prototype's exact fallback and fragment wrapping,
        # without installing global serializer patches in a stage-only benchmark.
        self.direct = DirectFrontend.__new__(DirectFrontend)
        self.direct.tokenizer = tokenizer
        self.direct.original_chat = self.chat._create_chat_logprobs
        self.direct.processor = SimpleNamespace(request_states={
            rid: SimpleNamespace(logprobs_processor=lp) for rid, lp in self.states.items()
        })
        self.direct.histories = {rid: deque(maxlen=4) for rid in self.states}
        self.direct.stats = {"fallback_cpu_ns": 0, "fallback_rows": 0}

    def native(self, groups, ids, values, ranks):
        outputs = []
        at = 0
        for group in groups:
            count, rid = group["rows"], group["rid"]
            lp = self.states[rid]
            start = len(lp.logprobs)
            raw = self.Lists(ids[at:at+count], values[at:at+count], ranks[at:at+count], None)
            lp._update_sample_logprobs(raw)
            model = self.chat._create_chat_logprobs(ids[at:at+count, 0].tolist(),
                lp.logprobs[start:], self.tokenizer, num_output_top_logprobs=20)
            outputs.append((rid, group["start"], model.model_dump_json().encode("utf-8")))
            at += count
        return outputs

    def finish_direct(self, groups, encoded, sample_ids, sample_values, fallback):
        outputs = []
        at = 0
        for group in groups:
            rid, count = group["rid"], group["rows"]
            fragments = []
            for row in range(at, at + count):
                state = int(encoded.status[row])
                if state == 0:
                    fragment = encoded.row(row)
                elif state == 1:
                    ids, values, rank = fallback[row]
                    fragment = self.direct.fallback(rid, ids, values, rank)
                else:
                    raise ValueError(f"invalid real row, encoder status={state}, row={row}")
                fragments.append(fragment)
                # DirectFrontend keeps one byte fragment per generated position
                # just as the native LogprobsProcessor keeps its object row.
                self.states[rid].logprobs.append(fragment)
                self.states[rid].cumulative_logprob += float(sample_values[row])
                self.direct.histories[rid].append(int(sample_ids[row]))
            model = self.direct.chat_logprobs(sample_ids[at:at+count].tolist(),
                fragments, self.tokenizer, num_output_top_logprobs=20)
            outputs.append((rid, group["start"], model._fragment.encode("utf-8")))
            at += count
        return outputs

    def final_state(self):
        return {rid: {"positions": len(lp.logprobs), "cumulative_logprob": lp.cumulative_logprob,
                      "cumulative_double_hex": struct.pack(">d", lp.cumulative_logprob).hex()}
                for rid, lp in self.states.items()}


def pack_fixture(batches, np):
    started = time.perf_counter()
    byte_count = 0
    for batch in batches:
        batch["packed"] = {}
        if not batch["rows"]:
            continue
        for name in ("ids", "values", "ranks"):
            array = np.ascontiguousarray(np.concatenate([g[name] for g in batch["groups"]], axis=0))
            batch["packed"][name] = array
            byte_count += array.nbytes
    return {"host_pack_wall_s": time.perf_counter() - started,
            "packed_compact_input_bytes": byte_count,
            "stage_boundary": "already contiguous same-ready eligible raw batch",
            "common_selection_and_sampler_packaging_timed": False}


def upload_fixture(batches, torch):
    started = time.perf_counter()
    byte_count = 0
    for batch in batches:
        batch["device_batch"] = {}
        for name, array in batch["packed"].items():
            byte_count += array.nbytes
            batch["device_batch"][name] = torch.from_numpy(array).to(device="cuda:0")
    torch.cuda.synchronize()
    return {"upload_wall_s": time.perf_counter() - started,
            "compact_input_bytes": byte_count,
            "torch_allocated_bytes_after_upload": torch.cuda.memory_allocated(),
            "note": "Replay fixture upload excluded from timed paths; real sampler already owns CUDA arrays."}


def compact_host(batch, np, cuda_mode, require_int64):
    if cuda_mode:
        host = {name: array.cpu().numpy() for name, array in batch["device_batch"].items()}
    else:
        host = batch["packed"]
    # CPUEncoder's C ABI requires int64; native vLLM accepts captured int32.
    # Charge that conversion only to the path that actually needs it.
    ids = np.ascontiguousarray(host["ids"], dtype=np.int64) if require_int64 else host["ids"]
    return ids, host["values"], host["ranks"]


def run_trace(arm, repeat, batches, requests, tokenizer, encoder, np, torch, cuda_mode):
    methods = NativeMethods(tokenizer, requests)
    observations, outputs = [], []
    for batch in batches:
        rows = batch["rows"]
        if not rows:
            observations.append({"batch": batch["index"], "ready_s": batch["time_s"],
                "rows": 0, "wall_s": 0.0, "cpu_s": 0.0, "output_bytes": 0,
                "fallback_rows": 0, "compact_d2h_bytes": 0, "header_d2h_bytes": 0,
                "fallback_d2h_bytes": 0})
            continue
        start, cpu_start = time.perf_counter(), time.process_time()
        compact_bytes = header_bytes = fallback_bytes = 0
        fallback = {}
        metric = {}
        if arm in ("native", "cpu"):
            ids, values, ranks = compact_host(batch, np, cuda_mode, arm == "cpu")
            if cuda_mode:
                compact_bytes = sum(array.numel() * array.element_size()
                    for array in batch["device_batch"].values())
            if arm == "native":
                result = methods.native(batch["groups"], ids, values, ranks)
            else:
                encoded = encoder.encode(ids, values, top_k=20)
                for row in np.flatnonzero(encoded.status):
                    fallback[int(row)] = (ids[row], values[row], ranks[row])
                result = methods.finish_direct(batch["groups"], encoded, ids[:, 0], values[:, 0], fallback)
        else:
            # Same contiguous ready batch used by native/CPU; no extra pack,
            # future-token aggregation or per-request transfer asymmetry.
            ids = batch["device_batch"]["ids"]
            values = batch["device_batch"]["values"]
            ranks = batch["device_batch"]["ranks"]
            encoded = encoder.encode_device(ids, values, top_k=20)
            metric = dict(encoder.last_metrics)
            # Numeric generation state still needs the sampled column, including
            # the original first probability rather than the last duplicate's.
            sample_ids = ids[:, 0].cpu().numpy()
            sample_values = values[:, 0].cpu().numpy()
            header_bytes = rows * (ids.element_size() + values.element_size())
            indices = np.flatnonzero(encoded.status)
            if len(indices):
                device_indices = torch.as_tensor(indices, dtype=torch.int64, device=ids.device)
                raw_ids = ids.index_select(0, device_indices).cpu().numpy()
                raw_values = values.index_select(0, device_indices).cpu().numpy()
                raw_ranks = ranks.index_select(0, device_indices).cpu().numpy()
                fallback_bytes = raw_ids.nbytes + raw_values.nbytes + raw_ranks.nbytes
                for local, row in enumerate(indices):
                    fallback[int(row)] = (raw_ids[local], raw_values[local], raw_ranks[local])
            result = methods.finish_direct(batch["groups"], encoded, sample_ids, sample_values, fallback)
        elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu_start
        observations.append({"batch": batch["index"], "ready_s": batch["time_s"],
            "rows": rows, "wall_s": elapsed, "cpu_s": cpu_elapsed,
            "output_bytes": sum(len(body) for _, _, body in result),
            "fallback_rows": len(fallback), "compact_d2h_bytes": compact_bytes,
            "header_d2h_bytes": header_bytes, "fallback_d2h_bytes": fallback_bytes,
            "encoder": metric})
        outputs.extend(result)
    # Parsing and equivalence validation occur after the timing window, equally
    # for all arms. They are NOT included in claimed encoding-stage cost.
    parsed = [(rid, position, json.loads(body)) for rid, position, body in outputs]
    semantic_bytes = json.dumps(parsed, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8")
    shape_summary = {}
    for count in sorted({x["rows"] for x in observations if x["rows"]}):
        cells = [x for x in observations if x["rows"] == count]
        shape_summary[str(count)] = {"batches": len(cells),
            "wall_ms_median": float(np.median([x["wall_s"] * 1000 for x in cells])),
            "wall_ms_p95": float(np.percentile([x["wall_s"] * 1000 for x in cells], 95))}
    result = {
        "arm": arm, "repeat": repeat, "whole_trace_repeat": True,
        "timed_wall_s": sum(x["wall_s"] for x in observations),
        "timed_cpu_s": sum(x["cpu_s"] for x in observations),
        "output_bytes": sum(x["output_bytes"] for x in observations),
        "fallback_rows": sum(x["fallback_rows"] for x in observations),
        "native_fallback_cpu_s": methods.direct.stats["fallback_cpu_ns"] / 1e9,
        "semantic_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
        "per_request_numeric_final_state": methods.final_state(),
        "max_process_rss_kib_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "observed_batch_shape_descriptive_only": shape_summary,
        "batches": observations,
    }
    return result, parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/moe-research-20261002/model")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--guard-record", type=Path)
    parser.add_argument("--core", type=int, default=2)
    args = parser.parse_args()
    # Refuse missing real data before compilation, CUDA initialization or output.
    for name in ("trace.json", "candidates.npz", "engine_args.json"):
        if not (args.capture / name).is_file():
            raise FileNotFoundError(f"real capture missing: {args.capture / name}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prior results: {args.output}")
    guard = None
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        guard = require_guard(args.guard_record)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.core})
    from bootstrap import apply
    repair = apply()
    import numpy as np
    import torch
    from vllm.tokenizers import get_tokenizer
    from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens
    from cpu_encoder import CPUEncoder, build_token_table
    torch.set_num_threads(1)
    requests, batches, capture_info = read_capture(args.capture, np)
    engine_args_path = args.capture / "engine_args.json"
    engine_args = json.loads(engine_args_path.read_text())
    captured_model = Path(engine_args["model"])
    if captured_model.resolve() != Path(args.model).resolve():
        raise ValueError("benchmark tokenizer/model path differs from capture engine_args.json")
    config_path = captured_model / "config.json"
    model_config = json.loads(config_path.read_text())
    model_vocab_size = int(model_config["vocab_size"])
    if model_vocab_size < 1:
        raise ValueError("model config has no valid vocabulary size")
    capture_info.update(engine_args_sha256=sha(engine_args_path),
                        model_config_sha256=sha(config_path), model_vocab_size=model_vocab_size)
    args.output.mkdir(parents=True, exist_ok=False)
    run_data = {"status": "started", "scope": "unpaced encoding stage only; no HTTP or model co-run",
        "cpu_only": args.cpu_only, "compact_transfer_included": not args.cpu_only,
        "validation_parse_timed": False, "guard_record": guard, "capture": capture_info,
        "command": sys.argv, "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "bootstrap": repair, "sources": {name: sha(HERE / name) for name in
            ("bench_real.py", "cpu_encoder.py", "cpu_encoder.cpp", "gpu_encoder.py",
             "gpu_encoder.cu", "frontend_bridge.py", "protocol.md")},
        "versions": {name: importlib.metadata.version(name) for name in
            ("numpy", "torch", "vllm", "pydantic", "tokenizers")}, "runs": []}
    write_json(args.output / "results.json", run_data)
    cpu = gpu = None
    try:
        init_start = time.perf_counter()
        tokenizer = get_tokenizer(args.model, trust_remote_code=False)
        tokenizer_bound = max(int(x) for x in tokenizer.get_vocab().values()) + 1
        unresolved_ids = []
        table_conversion_error = None
        def convert_full_model_vocab(tok, ids):
            nonlocal table_conversion_error
            try:
                return convert_ids_list_to_tokens(tok, ids)
            except Exception as exc:
                # A model may pad its legal output vocabulary beyond the
                # tokenizer's named vocabulary. Keep those rows in the table;
                # unsupported strings are explicit native fallbacks, never
                # removed from candidates or relabelled as invalid IDs.
                table_conversion_error = repr(exc)
                split = min(tokenizer_bound, len(ids))
                decoded = list(convert_ids_list_to_tokens(tok, ids[:split]))
                for token_id in ids[split:]:
                    try:
                        one = convert_ids_list_to_tokens(tok, [token_id])
                        if len(one) != 1:
                            raise ValueError("native conversion did not return one token")
                        decoded.append(one[0])
                    except Exception:
                        unresolved_ids.append(int(token_id))
                        decoded.append(None)  # TokenTable marks this row unsafe.
                return decoded
        table = build_token_table(tokenizer, converter=convert_full_model_vocab,
                                  vocab_size=model_vocab_size)
        if table.vocab_size != model_vocab_size:
            raise AssertionError("token table omitted model vocabulary IDs")
        table.metadata.update({"model_vocab_size": model_vocab_size,
            "tokenizer_named_id_bound": tokenizer_bound,
            "model_padding_id_count": max(0, model_vocab_size - tokenizer_bound),
            "whole_table_conversion_error": table_conversion_error,
            "unresolved_model_padding_ids_native_fallback": unresolved_ids,
            "native_converter": "vllm.tokenizers.detokenizer_utils.convert_ids_list_to_tokens"})
        table.save(args.output / "token_table.npz")
        cpu = CPUEncoder(table)
        run_data["initialization"] = {"token_table_build_s": table.build_seconds,
            "token_table_nbytes": table.nbytes, "token_table_metadata": table.metadata,
            "cpu_initialization_s": cpu.initialization_seconds, "cpu_build": cpu.build_info}
        run_data["initialization"]["fixture"] = pack_fixture(batches, np)
        if not args.cpu_only:
            from gpu_encoder import GPUEncoder
            run_data["gpu_name"] = torch.cuda.get_device_name(0)
            gpu = GPUEncoder(table, max_rows=max(b["rows"] for b in batches), device=0)
            run_data["initialization"]["gpu"] = gpu.initialization_info
            run_data["initialization"]["fixture"].update(upload_fixture(batches, torch))
        run_data["initialization"]["total_wall_s"] = time.perf_counter() - init_start
        order = ["native", "cpu", "cpu", "native"] if args.cpu_only else ["native", "cpu", "gpu", "gpu", "cpu", "native"]
        run_data["frozen_order"] = order
        write_json(args.output / "results.json", run_data)
        reference, reference_state = None, None
        repeats = Counter()
        for index, arm in enumerate(order):
            repeat = repeats[arm]
            repeats[arm] += 1
            result, parsed = run_trace(arm, repeat, batches, requests, tokenizer,
                cpu if arm == "cpu" else gpu, np, torch, not args.cpu_only)
            result["order_index"] = index
            if reference is None:
                reference, reference_state = parsed, result["per_request_numeric_final_state"]
            result["parsed_semantics_equal"] = parsed == reference
            result["numeric_state_equal"] = result["per_request_numeric_final_state"] == reference_state
            run_data["runs"].append(result)
            write_json(args.output / "results.json", run_data)
            print(json.dumps({key: result[key] for key in ("order_index", "arm", "repeat",
                "timed_wall_s", "timed_cpu_s", "output_bytes", "fallback_rows",
                "parsed_semantics_equal", "numeric_state_equal")}), flush=True)
            if not result["parsed_semantics_equal"] or not result["numeric_state_equal"]:
                # Preserve the failing run and actual mismatched response pair.
                mismatch = next(((i, a, b) for i, (a, b) in enumerate(zip(reference, parsed)) if a != b), None)
                write_json(args.output / "semantic_failure.json", {
                    "arm": arm, "repeat": repeat, "first_mismatch": mismatch,
                    "reference_length": len(reference), "candidate_length": len(parsed),
                    "reference_state": reference_state, "candidate_state": result["per_request_numeric_final_state"]})
                raise AssertionError("real trace semantic/state mismatch; retained, not benchmark success")
        run_data["status"] = "complete"
        run_data["all_semantics_equal"] = True
        run_data["interpretation_limit"] = (
            "Two whole-trace repetitions, not independent per-token samples. Observed shape summaries "
            "are descriptive; no utilization/SLO/HTTP/model-interference claim. All arms begin at the same "
            "already contiguous same-ready eligible raw batch. Common row selection and actual sampler "
            "packaging are excluded here and must be charged in subsequent model co-run evaluation.")
    except BaseException as exc:
        run_data["status"] = "failed"
        run_data["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        run_data["ended_unix"] = time.time()
        write_json(args.output / "results.json", run_data)
        if cpu is not None:
            cpu.close()
        if gpu is not None:
            gpu.close()


if __name__ == "__main__":
    main()

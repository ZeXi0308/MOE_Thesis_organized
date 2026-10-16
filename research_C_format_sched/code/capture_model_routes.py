#!/usr/bin/env python3
"""Capture executed HF 5.17 OLMoE/Qwen3-MoE routes, not reconstructed logits.

Reads local unquantized checkpoints and fixed rendered-prompt JSONL only:
  {"request_id":"p0","prompt":"The ..."}
Sequential batch-one greedy argmax generation uses the real KV cache and stops
at checkpoint/tokenizer EOS or the token cap. No sampling, penalties, forced
EOS, chat-template insertion, prompt truncation, or artificial output padding.
The final selected token (including EOS) is not fed back through the model.

Reuses July capture_routes_v2.py's per-token expert-ID/weight preservation idea,
but hooks HF5's fused 3D Experts call directly. The hook copies the EXACT ids
and weights supplied to the executed experts and never recomputes top-k from
logits. CPU transfers/synchronization, Python experts and file IO make this a
diagnostic; it is NOT vLLM routing fidelity or serving performance evidence.

Inspect plan without loading a model:
  python code/capture_model_routes.py --checkpoint /local/model \
    --prompts /local/prompts.jsonl --output runs/routes/route_capture
Actual GPU execution, when authorized by the main controller:
  python code/run_locked.py --name olmoe_routes_r01 --timeout 1800 -- \
    python code/capture_model_routes.py --checkpoint /local/olmoe \
    --prompts /local/prompts.jsonl --output runs/olmoe_routes_r01/route_capture \
    --lock-status runs/olmoe_routes_r01/status.json --run
Use the Qwen3 local checkpoint and another run name for the second model.
Default/hard maxima: 32 prompts, 32 new tokens; prompts over 4096 tokens fail
explicitly rather than being truncated. Source/header hashes are retained.

Extract one completed route record for bench_expert_formats.py --histogram:
  python code/capture_model_routes.py --extract-routes runs/.../routes.jsonl \
    --extract-row 0 --output /local/one_route.json
This preserves exact token co-occurrence and weights, not just marginal counts.

--self-test runs tiny random models on CPU, with no checkpoint or download.
Verified source: transformers v5.17.0 models/{olmoe,qwen3_moe}/modeling_*.py
and integrations/moe.py (use_experts_implementation). The CLI fails closed on
different versions, unexpected expert classes/shapes or missing hooked calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_prompts(path, limit):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    selected, seen = [], set()
    for i, row in enumerate(rows[:limit]):
        rid = str(row.get("request_id", row.get("id", i)))
        if rid in seen or not isinstance(row.get("prompt"), str) or not row["prompt"]:
            raise ValueError("selected prompts need unique IDs and nonempty prompt strings")
        seen.add(rid)
        selected.append({"request_id": rid, "prompt": row["prompt"], "source_row": i,
                         "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest()})
    if not selected:
        raise ValueError("empty prompt selection")
    return selected, len(rows)


class RouteRecorder:
    """Observe the inputs consumed by each 3D expert module, without mutation."""
    def __init__(self, model, emit):
        self.model, self.emit = model, emit
        self.handles, self.layers, self.pending = [], {}, []
        self.context = None
        expected = {"olmoe": ("OlmoeSparseMoeBlock", "OlmoeExperts"),
                    "qwen3_moe": ("Qwen3MoeSparseMoeBlock", "Qwen3MoeExperts")}
        if model.config.model_type not in expected:
            raise ValueError("only checked OLMoE and Qwen3-MoE model types are supported")
        sparse_name, expert_name = expected[model.config.model_type]
        for index, layer in enumerate(model.model.layers):
            block = layer.mlp
            if type(block).__name__ != sparse_name:
                continue  # Qwen config may explicitly contain dense MLP layers.
            experts = block.experts
            if type(experts).__name__ != expert_name or experts.gate_up_proj.ndim != 3 or experts.down_proj.ndim != 3:
                raise ValueError("unexpected HF fused expert class or weight layout")
            parameters = list(inspect.signature(experts.forward).parameters)
            if parameters[:3] != ["hidden_states", "top_k_index", "top_k_weights"]:
                raise ValueError("expert forward contract changed: " + str(parameters))
            if experts.config._experts_implementation != "eager":
                raise ValueError("this diagnostic explicitly uses the checked eager expert implementation")
            self.layers[index] = {"model_type": model.config.model_type, "layer": index,
                "module": f"model.layers.{index}.mlp.experts",
                "num_experts": int(experts.num_experts), "top_k": int(block.gate.top_k),
                "hidden_size": int(experts.hidden_dim), "intermediate_size": int(experts.intermediate_dim),
                "norm_topk_prob": bool(block.gate.norm_topk_prob),
                "gate_up_shape": list(experts.gate_up_proj.shape), "down_shape": list(experts.down_proj.shape)}
            self.handles.append(experts.register_forward_pre_hook(self._pre(index), with_kwargs=True))
            self.handles.append(experts.register_forward_hook(self._post(index), with_kwargs=True))
        if not self.layers:
            raise ValueError("no checked sparse expert modules were found")

    def _pre(self, layer):
        def hook(module, args, kwargs):
            import torch
            if self.context is None:
                raise RuntimeError("expert call outside an explicitly labeled forward")
            hidden = args[0] if len(args) > 0 else kwargs["hidden_states"]
            ids = args[1] if len(args) > 1 else kwargs["top_k_index"]
            weights = args[2] if len(args) > 2 else kwargs["top_k_weights"]
            info = self.layers[layer]
            if ids.shape != weights.shape or tuple(ids.shape) != (self.context["input_tokens"], info["top_k"]):
                raise ValueError("actual route shape does not match this batch-one forward")
            if hidden.shape[0] != ids.shape[0] or hidden.shape[1] != info["hidden_size"]:
                raise ValueError("actual expert input shape mismatch")
            if ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("expert indices must be integers")
            cpu_ids = ids.detach().to(device="cpu", dtype=torch.int64)
            cpu_weights = weights.detach().to(device="cpu", dtype=torch.float32)
            if bool((cpu_ids < 0).any()) or bool((cpu_ids >= info["num_experts"]).any()):
                raise ValueError("expert ID outside model range")
            if not bool(torch.isfinite(cpu_weights).all()) or bool((cpu_weights < 0).any()):
                raise ValueError("nonfinite or negative actual routing weights")
            ordered = cpu_ids.sort(dim=1).values
            if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
                raise ValueError("duplicate actual expert within one token")
            counts = torch.bincount(cpu_ids.flatten(), minlength=info["num_experts"]).tolist()
            self.pending.append({**self.context, **info, "topk_ids": cpu_ids.tolist(),
                "topk_weights": cpu_weights.tolist(), "routing_weight_dtype": str(weights.dtype),
                "counts": counts, "assignments": sum(counts),
                "mean_per_expert": sum(counts) / info["num_experts"],
                "max_per_expert": max(counts), "active_experts": sum(c > 0 for c in counts),
                "expert_call_completed": False, "forward_completed": False,
                "route_source": "actual_experts_forward_arguments", "weights_renormalized_by_capture": False})
        return hook

    def _post(self, layer):
        def hook(module, args, kwargs, output):
            matching = [r for r in self.pending if r["layer"] == layer and not r["expert_call_completed"]]
            if len(matching) != 1 or output is None:
                raise RuntimeError("unexpected expert invocation multiplicity or output")
            matching[0]["expert_call_completed"] = True
        return hook

    def begin(self, request_id, forward_index, input_ids, position_start):
        if self.context is not None:
            raise RuntimeError("previous forward not finalized")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("capture requires unpadded batch size one")
        self.context = {"request_id": request_id, "forward_index": forward_index,
            "phase": "prefill" if forward_index == 0 else "decode", "batch_size": 1,
            "input_tokens": int(input_ids.shape[1]), "position_start": position_start,
            "input_token_ids": input_ids[0].detach().cpu().tolist()}
        self.pending = []

    def finish(self, success, next_token_id=None, error=None):
        counts = Counter(r["layer"] for r in self.pending)
        valid = counts == Counter({i: 1 for i in self.layers}) and all(r["expert_call_completed"] for r in self.pending)
        if success and not valid:
            error = "not every expected expert module completed exactly once"
        try:
            for record in self.pending:
                record.update(forward_completed=bool(success and valid), next_token_id=next_token_id,
                              forward_error=error)
                self.emit(record)
        finally:
            self.pending, self.context = [], None
        if success and not valid:
            raise RuntimeError(error)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def greedy_capture(model, input_ids, eos_ids, max_new_tokens, recorder, result):
    """Actual cached forwards and argmax, without a second router invocation."""
    import torch
    prompt_length = int(input_ids.shape[1])
    past, current = None, input_ids
    with torch.inference_mode():
        for step in range(max_new_tokens):
            position = 0 if step == 0 else prompt_length + step - 1
            recorder.begin(result["request_id"], step, current, position)
            try:
                total_context = prompt_length + step
                output = model(input_ids=current,
                    attention_mask=torch.ones((1, total_context), dtype=torch.long, device=current.device),
                    position_ids=torch.arange(position, position + current.shape[1], device=current.device).unsqueeze(0),
                    past_key_values=past, use_cache=True, logits_to_keep=1, output_router_logits=False,
                    return_dict=True)
                if output.past_key_values is None:
                    raise RuntimeError("model did not return its KV cache")
                if int(output.past_key_values.get_seq_length()) != total_context:
                    raise RuntimeError("KV cache length does not match consumed prompt/generated tokens")
                logits = output.logits[0, -1]
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("model returned nonfinite logits")
                token = int(logits.argmax().item())
                recorder.finish(True, next_token_id=token)
                past = output.past_key_values
                result["generated_token_ids"].append(token)
                result["forward_count"] += 1
                if token in eos_ids:
                    result["termination"] = "eos"
                    break
                current = torch.tensor([[token]], device=current.device, dtype=torch.long)
            except BaseException as error:
                if recorder.context is not None:
                    recorder.finish(False, error=f"{type(error).__name__}: {error}")
                raise
        else:
            result["termination"] = "cap"
    result["generated_tokens"] = len(result["generated_token_ids"])
    result["cap_reached"] = result["generated_tokens"] == max_new_tokens
    result["eos_reached"] = bool(result["generated_token_ids"] and result["generated_token_ids"][-1] in eos_ids)
    result["status"] = "ok"
    return result


def extract_replay(path, row_index, output):
    with Path(path).open() as source:
        for i, line in enumerate(source):
            if i == row_index:
                row = json.loads(line)
                break
        else:
            raise IndexError("route row out of range")
    if not row["forward_completed"] or not row["expert_call_completed"]:
        raise ValueError("cannot export an incomplete/failed forward as a replay")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    row["source_routes_sha256"] = sha256(path)
    row["source_row"] = row_index
    row["replay_scope"] = "exact HF eager layer routes/weights; hidden inputs and weight format in a microbenchmark remain separate"
    write_json(target, row)
    print(json.dumps({"output": str(target), "tokens": row["input_tokens"], "layer": row["layer"], "phase": row["phase"]}))


def self_test():
    """Tiny actual HF models, CPU only; assert hooks preserve real router outputs."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
    if importlib.metadata.version("transformers") != "5.17.0":
        raise RuntimeError("self-test expects the inspected Transformers 5.17.0")
    torch.set_num_threads(1)
    torch.manual_seed(5)
    assert not torch.cuda.is_initialized()
    for config_cls, model_cls in ((OlmoeConfig, OlmoeForCausalLM), (Qwen3MoeConfig, Qwen3MoeForCausalLM)):
        config = config_cls(vocab_size=64, hidden_size=16, intermediate_size=8,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
            eos_token_id=63, **({"moe_intermediate_size": 8} if config_cls is Qwen3MoeConfig else {}))
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        model = model_cls(config).to("cpu").eval()
        prompt = torch.tensor([[4, 5, 6]])
        with torch.inference_mode():
            reference = model(input_ids=prompt, use_cache=False, logits_to_keep=1).logits[0, -1].argmax().item()
        records, gate_returns = [], []
        handles = []
        for layer in model.model.layers:
            def observe_gate(module, args, output):
                gate_returns.append((output[2].detach().clone(), output[1].detach().float().clone()))
            handles.append(layer.mlp.gate.register_forward_hook(observe_gate))
        recorder = RouteRecorder(model, records.append)
        result = {"request_id": "cpu", "generated_token_ids": [], "forward_count": 0}
        greedy_capture(model, prompt, set(), 3, recorder, result)
        assert result["termination"] == "cap" and result["generated_token_ids"][0] == reference
        assert len(records) == 6 and all(r["forward_completed"] for r in records)
        assert [r["input_tokens"] for r in records] == [3, 3, 1, 1, 1, 1]
        assert [r["phase"] for r in records] == ["prefill", "prefill", "decode", "decode", "decode", "decode"]
        for row, (ids, weights) in zip(records, gate_returns):
            assert row["topk_ids"] == ids.tolist() and row["topk_weights"] == weights.tolist()
            assert row["counts"] == torch.bincount(ids.flatten(), minlength=4).tolist()
            assert sum(row["counts"]) == row["input_tokens"] * 2
        # Same actual model naturally stops when its greedy first token is EOS.
        eos_result = {"request_id": "cpu_eos", "generated_token_ids": [], "forward_count": 0}
        greedy_capture(model, prompt, {int(reference)}, 3, recorder, eos_result)
        assert eos_result["termination"] == "eos" and eos_result["generated_tokens"] == 1
        recorder.close()
        for handle in handles:
            handle.remove()
        print(config.model_type, "CPU 3D-expert hook/cached decode/EOS checks passed")
    assert not torch.cuda.is_initialized()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--prompts", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--max-prompts", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lock-status", type=Path)
    p.add_argument("--run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--extract-routes", type=Path)
    p.add_argument("--extract-row", type=int, default=0)
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.extract_routes:
        if args.output is None or args.extract_row < 0:
            p.error("extraction needs --output and a nonnegative --extract-row")
        extract_replay(args.extract_routes, args.extract_row, args.output)
        return
    if args.checkpoint is None or args.prompts is None or args.output is None:
        p.error("provide local --checkpoint, --prompts and --output")
    if not (1 <= args.max_prompts <= 32 and 1 <= args.max_new_tokens <= 32 and args.max_prompt_tokens > 0):
        p.error("prompt/generation limits must be 1..32 and prompt-token limit positive")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        p.error("checkpoint must be an existing local model directory")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") not in ("olmoe", "qwen3_moe"):
        p.error("only OLMoE and Qwen3-MoE are supported")
    if config.get("quantization_config"):
        p.error("this capture validates unquantized HF checkpoints; quantized loading is not implemented")
    prompts, total_prompts = read_prompts(args.prompts, args.max_prompts)
    plan = {"schema_version": 1, "checkpoint": str(checkpoint), "model_type": config["model_type"],
        "config_sha256": sha256(checkpoint / "config.json"), "prompts_sha256": sha256(args.prompts),
        "prompt_rows_available": total_prompts, "selected_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens, "max_prompt_tokens": args.max_prompt_tokens,
        "dtype": args.dtype, "device": args.device, "add_special_tokens": args.add_special_tokens,
        "decoding": "batch-one cached greedy argmax of unchanged logits; EOS or cap; no penalties or forced EOS",
        "scope": "HF eager diagnostic with synchronous CPU route copies; not vLLM serving performance",
        "experts_implementation": "eager", "attn_implementation": "eager", "status": "planned"}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    if args.lock_status is None or args.device != "cuda:0":
        p.error("full-model capture requires --lock-status and the single locked cuda:0")
    from bench_serving import verify_lock
    plan["lock"] = verify_lock(args.lock_status)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    if transformers.__version__ != "5.17.0":
        raise RuntimeError("capture contract was checked against Transformers 5.17.0")
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=False)
    plan.update(status="loading", transformers=transformers.__version__, torch=torch.__version__, created_unix_ns=time.time_ns())
    source_root = Path(transformers.__file__).parent
    name = config["model_type"]
    plan["source_sha256"] = {f: sha256(source_root / f) for f in (
        f"models/{name}/modeling_{name}.py", "integrations/moe.py")}
    plan["checkpoint_files"] = [{"name": f.name, "bytes": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns}
        for f in sorted(checkpoint.iterdir()) if f.is_file() and (f.suffix in (".safetensors", ".bin", ".json"))]
    plan["checkpoint_revision"] = config.get("_commit_hash")
    plan["checkpoint_revision_note"] = "local path/config hash and file sizes retained; no base revision inferred if missing"
    write_json(output_dir / "metadata.json", plan)
    (output_dir / "prompts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts))
    def interrupted(signum, frame):
        raise KeyboardInterrupt("capture interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    recorder = None
    completed, failed, route_count = 0, 0, 0
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
            trust_remote_code=False, dtype=getattr(torch, args.dtype), device_map={"": args.device},
            attn_implementation="eager", experts_implementation="eager").eval()
        eos = model.generation_config.eos_token_id
        eos_source = "checkpoint_generation_config"
        if eos is None:
            eos, eos_source = model.config.eos_token_id, "model_config"
        if eos is None:
            eos, eos_source = tokenizer.eos_token_id, "tokenizer"
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if not eos_ids or any(type(x) is not int for x in eos_ids):
            raise ValueError("no usable local EOS token IDs")
        plan.update(status="running", eos_token_ids=sorted(eos_ids), eos_source=eos_source,
                    tokenizer_class=type(tokenizer).__name__)
        with (output_dir / "routes.jsonl").open("w") as routes, (output_dir / "generations.jsonl").open("w") as generations:
            def emit(record):
                nonlocal route_count
                routes.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                routes.flush()
                route_count += 1
            recorder = RouteRecorder(model, emit)
            plan["layers"] = list(recorder.layers.values())
            write_json(output_dir / "metadata.json", plan)
            for prompt in prompts:
                result = {**prompt, "status": "error", "termination": "error", "generated_token_ids": [], "forward_count": 0}
                try:
                    ids = tokenizer(prompt["prompt"], return_tensors="pt", add_special_tokens=args.add_special_tokens,
                                    truncation=False)["input_ids"]
                    result["input_token_ids"] = ids[0].tolist()
                    result["input_tokens"] = int(ids.shape[1])
                    if ids.shape[1] == 0 or ids.shape[1] > args.max_prompt_tokens:
                        raise ValueError("prompt token count outside configured range; prompt not truncated")
                    if ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
                        raise ValueError("prompt plus generation cap exceeds model context")
                    greedy_capture(model, ids.to(args.device), eos_ids, args.max_new_tokens, recorder, result)
                    result["generated_text"] = tokenizer.decode(result["generated_token_ids"], skip_special_tokens=False)
                    completed += 1
                except BaseException as error:
                    result.update(status="error", termination="error", error=f"{type(error).__name__}: {error}",
                                  generated_tokens=len(result["generated_token_ids"]))
                    failed += 1
                    if isinstance(error, (KeyboardInterrupt, SystemExit)) or isinstance(error, torch.OutOfMemoryError):
                        raise
                finally:
                    generations.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    generations.flush()
                print(json.dumps({"request_id": result["request_id"], "status": result["status"],
                                  "termination": result["termination"], "generated_tokens": len(result["generated_token_ids"])}), flush=True)
        plan["status"] = "complete" if failed == 0 else "completed_with_errors"
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if recorder is not None:
            recorder.close()
        plan.update(completed_prompts=completed, failed_prompts=failed,
                    unattempted_prompts=len(prompts) - completed - failed, route_records=route_count,
                    ended_unix_ns=time.time_ns())
        write_json(output_dir / "metadata.json", plan)
    print(json.dumps({"output": str(output_dir), "status": plan["status"], "route_records": route_count}))
    raise SystemExit(0 if plan["status"] == "complete" else 1)


if __name__ == "__main__":
    main()

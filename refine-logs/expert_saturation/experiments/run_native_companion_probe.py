#!/usr/bin/env python3
"""Fixed-input native MoE numerical probe; no serving-performance claims."""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_state():
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                         "--format=csv,noheader"], text=True).strip()
    others = [r for r in csv.reader(processes.splitlines()) if r and int(r[0]) != os.getpid()]
    if others:
        raise RuntimeError(f"GPU occupied by another process: {others}")
    return dict(processes=processes, gpu=subprocess.check_output(["nvidia-smi",
        "--query-gpu=name,uuid,memory.total,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader"], text=True).strip())


def arrangements(target, width, position):
    original = [i for i in range(target, target + width) if i != target]
    replacement = list(range(width, 2 * width))[:width - 1] if target == 0 else list(range(width - 1))
    result = {}
    for arm, companions in (("original", original), ("permuted", original[::-1]), ("replaced", replacement)):
        indices = companions[:position] + [target] + companions[position:]
        assert len(indices) == width and len(set(indices)) == width and indices[position] == target
        result[arm] = indices
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    config, inputs = json.loads(a.config.read_text()), json.loads(a.inputs.read_text())
    a.output_dir.mkdir(parents=True, exist_ok=False)
    dump(a.output_dir / "config.json", config)
    (a.output_dir / "commands.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    dump(a.output_dir / "status.json", dict(status="INITIALIZING", measured_calls=0))
    engine = None
    try:
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                          VLLM_USE_FLASHINFER_SAMPLER="0")
        import torch
        import vllm
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        from vllm.forward_context import set_forward_context, get_forward_context
        from vllm.utils.torch_utils import _USE_LAYERNAME
        from vllm import SamplingParams
        assert vllm.__version__ == "0.26.0" and torch.cuda.device_count() == 1
        kwargs = dict(model=config["model"]["id"], revision=config["model"]["revision"],
            tokenizer_revision=config["model"]["tokenizer_revision"], dtype="bfloat16", seed=config["seed"],
            enforce_eager=True, max_model_len=256, max_num_seqs=32, max_num_batched_tokens=4096,
            gpu_memory_utilization=0.70, enable_chunked_prefill=False, enable_prefix_caching=False,
            scheduling_policy="fcfs", async_scheduling=False, stream_interval=1, enable_return_routed_experts=False)
        dump(a.output_dir / "engine_args.json", kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        mlp = runner.get_model().model.layers[config["layer"]].mlp
        moe, re = mlp.experts, mlp.experts.routed_experts
        triton_module = importlib.import_module("vllm.model_executor.layers.fused_moe.experts.triton_moe")
        package = Path(vllm.__file__).parent
        source_paths = [Path(__file__), Path(triton_module.__file__), package / "model_executor/models/olmoe.py",
                        package / "model_executor/layers/fused_moe/routed_experts.py"]
        dump(a.output_dir / "environment.json", dict(python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, vllm=vllm.__version__, gpu_before=before, cpu_threads=torch.get_num_threads(),
            use_layername=_USE_LAYERNAME, model_layer_type=type(mlp).__name__,
            expert_type=type(re).__name__, quant_method_type=type(re.quant_method).__name__,
            source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths}))
        tokens = inputs["actual_prompt_token_ids"]
        assert len(tokens) == 32 and all(len(ids) == 128 for ids in tokens)
        for source, ids in zip(inputs["source_requests"], tokens):
            assert hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == source["prompt_token_ids_sha256"]
        input_digest = hashlib.sha256(a.inputs.read_bytes()).hexdigest()
        if not a.fixture.exists():
            mapping, captured = {}, []
            for i, ids in enumerate(tokens):
                internal = engine.add_request(f"fixture/{i}", {"prompt_token_ids": ids},
                    SamplingParams(temperature=0, max_tokens=1, min_tokens=1, ignore_eos=True, detokenize=False))
                mapping[internal] = i

            def hook(module, args):
                batch = runner.input_batch
                names = list(batch.req_ids[:batch.num_reqs])
                offsets = runner.query_start_loc.cpu[:len(names) + 1].tolist()
                assert len(names) == 32 and set(names) == set(mapping)
                assert offsets[0] == 0 and offsets[-1] == 4096
                assert all(b - x == 128 for x, b in zip(offsets, offsets[1:]))
                hidden = args[0]
                assert hidden.shape[0] == 4096
                selected = torch.stack([hidden[offsets[j + 1] - 1].detach().clone() for j in range(len(names))])
                captured.append((names, offsets, selected))

            handle = mlp.register_forward_pre_hook(hook)
            try:
                outputs = engine.step()
            finally:
                handle.remove()
            assert len(outputs) == 32 and not engine.has_unfinished_requests() and len(captured) == 1
            names, offsets, selected = captured[0]
            pool = torch.empty_like(selected)
            for j, name in enumerate(names):
                pool[mapping[name]].copy_(selected[j])
            fixture = dict(hidden=pool.cpu(), input_sha256=input_digest, layer=config["layer"],
                source_requests=inputs["source_requests"], native_query_offsets=offsets,
                native_request_source_order=[mapping[n] for n in names])
            a.fixture.parent.mkdir(parents=True, exist_ok=True)
            torch.save(fixture, a.fixture)
            print("FIXTURE_CAPTURED", flush=True)
        else:
            fixture = torch.load(a.fixture, map_location="cpu", weights_only=True)
            assert fixture["input_sha256"] == input_digest and fixture["layer"] == config["layer"]
            assert fixture["source_requests"] == inputs["source_requests"]
            pool = fixture["hidden"].cuda()
            print("FIXTURE_REUSED", flush=True)
        assert pool.dtype == torch.bfloat16 and torch.isfinite(pool).all()
        dispatch, tensors, records = [], {}, []
        original_prepare = triton_module._prepare_expert_assignment

        def prepare(*args, **kw):
            result = original_prepare(*args, **kw)
            dispatch.append((dict(args[1]), tuple(t.detach().clone() if t is not None else None for t in result)))
            return result

        def execute(x, full=False):
            with torch.inference_mode(), set_forward_context(None, engine.vllm_config,
                    num_tokens=x.shape[0], skip_compiled=True):
                if full:
                    if not _USE_LAYERNAME:
                        ctx = get_forward_context()
                        ctx.moe_layer_index = ctx.all_moe_layers.index(moe.layer_name)
                    return mlp(x.clone())
                logits, _ = mlp.gate(x)
                weights, ids = moe.router.select_experts(x, logits,
                    topk_indices_dtype=re.quant_method.topk_indices_dtype)
                y = re.forward_modular(x.clone(), weights, ids)
                return logits, weights, ids, y

        shapes = {t: arrangements(t, config["width"], config["target_position"]) for t in config["target_source_indices"]}
        sanity = []
        for target, arms in shapes.items():
            for arm, indices in arms.items():
                x = pool[indices].contiguous()
                *_, y = execute(x)
                full = execute(x, full=True)
                equal = torch.equal(y.view(torch.uint8), full.view(torch.uint8))
                sanity.append(dict(target=target, arm=arm, native_full_equals_split=equal,
                                   max_abs=float((y.float() - full.float()).abs().max())))
                if not equal:
                    raise ValueError("split extraction differs from full native MLP")
        dump(a.output_dir / "extraction_checks.json", sanity)
        triton_module._prepare_expert_assignment = prepare
        try:
            for repeat, order in enumerate((("original", "permuted", "replaced"),
                                           ("replaced", "permuted", "original"),
                                           ("permuted", "original", "replaced"))):
                for target, arms in shapes.items():
                    for arm in order:
                        gpu_state()
                        dispatch.clear()
                        indices = arms[arm]
                        x = pool[indices].contiguous()
                        logits, weights, ids, y = execute(x)
                        torch.cuda.synchronize()
                        assert all(torch.isfinite(t).all() for t in (x, logits, weights, y))
                        assert len(dispatch) == 1, "expected the actual native Triton dispatch metadata"
                        key = f"r{repeat}/target{target}/{arm}"
                        hist = torch.bincount(ids.flatten().long(), minlength=logits.shape[1])
                        selected_config, (sorted_ids, expert_ids, padded) = dispatch[0]
                        valid = int(padded.item())
                        block = selected_config["BLOCK_SIZE_M"]
                        sorted_ids = sorted_ids[:valid].cpu()
                        expert_ids = expert_ids[:(valid + block - 1) // block].cpu()
                        target_assignments = range(config["target_position"] * ids.shape[1],
                                                   (config["target_position"] + 1) * ids.shape[1])
                        locations = {str(k): (sorted_ids == k).nonzero().flatten().tolist() for k in target_assignments}
                        assert all(len(v) == 1 for v in locations.values())
                        tensors[key] = dict(logits=logits.cpu(), weights=weights.cpu(), ids=ids.cpu(), output=y.cpu(),
                                            sorted_token_ids=sorted_ids, expert_ids=expert_ids)
                        records.append(dict(key=key, repeat=repeat, target=target, arm=arm, source_indices=indices,
                            target_position=config["target_position"], actual_kernel_config=selected_config,
                            padded_count=valid, expert_counts=hist.cpu().tolist(),
                            U=float((hist > 0).float().mean()), C=float(hist.max().float() / hist.float().mean()),
                            target_sorted_locations=locations))
                        print("MEASURED", key, flush=True)
        finally:
            triton_module._prepare_expert_assignment = original_prepare
        torch.save(tensors, a.output_dir / "tensors.pt")
        dump(a.output_dir / "records.json", records)
        dump(a.output_dir / "status.json", dict(status="COMPLETE", measured_calls=len(records), gpu_after=gpu_state(),
            fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(), claim_ceiling=config["evidence_ceiling"]))
    except Exception as exc:
        dump(a.output_dir / "status.json", dict(status="INCOMPLETE_OR_BLOCKED", error=f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()


if __name__ == "__main__":
    main()

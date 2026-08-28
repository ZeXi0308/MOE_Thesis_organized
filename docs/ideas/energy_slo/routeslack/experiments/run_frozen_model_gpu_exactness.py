from __future__ import annotations

"""Compare unpatched and instrumented cached decode on one frozen MoE model.

The runner holds weights, dtype, prompt, token prefixes, and cache progression
fixed.  It is a Gate-0 development qualification only: batch size is one and
there is no continuous-serving or routed/dispatched/executed/combined ledger.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", default="wikitext103_docs")
    parser.add_argument("--split", default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _choose_non_eos(logits, eos_token_id: int | None):
    import torch

    ordered = torch.argsort(logits[:, -1, :], dim=-1, descending=True)
    token = ordered[:, :1]
    if eos_token_id is not None and int(token.item()) == int(eos_token_id):
        token = ordered[:, 1:2]
    return token


def _baseline_cached_decode(model, inputs, *, steps: int, eos_token_id: int | None):
    import torch

    capture_dir = next(
        candidate / "docs/ideas/bcrd/experiments"
        for candidate in Path(__file__).resolve().parents
        if (candidate / "docs/ideas/bcrd/experiments").is_dir()
    )
    sys.path.insert(0, str(capture_dir))
    from capture_native_routes import _cache_sequence_length

    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("exactness runner requires batch size 1")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    logits: list[object] = []
    token_ids: list[int] = []
    cache_lengths: list[int] = []
    with torch.inference_mode():
        output = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        next_logits = output.logits[:, -1:, :]
        for _ in range(steps):
            next_token = _choose_non_eos(next_logits, eos_token_id)
            token_ids.append(int(next_token.item()))
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1
            )
            position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            next_logits = output.logits[:, -1:, :]
            logits.append(next_logits.detach().float().cpu())
            cache_lengths.append(_cache_sequence_length(past_key_values))
    return token_ids, logits, cache_lengths


def main() -> None:
    args = parse_args()
    if args.decode_steps <= 0 or args.seq_len <= 0:
        raise SystemExit("decode-steps and seq-len must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise SystemExit("rtol and atol must be non-negative")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for frozen-model exactness")

    shared = next(
        candidate / "experiments/shared"
        for candidate in Path(__file__).resolve().parents
        if (candidate / "experiments/shared").is_dir()
    )
    bcrd = next(
        candidate / "docs/ideas/bcrd/experiments"
        for candidate in Path(__file__).resolve().parents
        if (candidate / "docs/ideas/bcrd/experiments").is_dir()
    )
    # Insert BCRD first and shared second so shared/policies.py has priority
    # over the unrelated BCRD module with the same top-level name.
    for path in (bcrd, shared):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from capture_moe import patch_mixtral_moe
    from capture_native_routes import run_cached_decode_steps
    from modeling import load_model, load_tokenizer
    from prompts import get_prompts

    torch.manual_seed(20260728)
    text = get_prompts(
        args.dataset,
        1,
        offset=args.offset,
        split=args.split,
        seed=20260728,
    )[0]
    tokenizer = load_tokenizer(
        args.model,
        local_files_only=args.offline,
        revision=args.model_revision,
    )
    model, load_seconds = load_model(
        args.model,
        dtype_name=args.dtype,
        local_files_only=args.offline,
        revision=args.model_revision,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    token_ids, baseline_logits, baseline_cache_lengths = _baseline_cached_decode(
        model,
        inputs,
        steps=args.decode_steps,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()

    recorder = patch_mixtral_moe(model, "full", num_receiver_groups=1, record_routes=True)
    recorder.update_contrib = lambda *a, **k: None
    recorder.update_receiver = lambda *a, **k: None
    recorder.update_error = lambda *a, **k: None
    recorder.update_pair_audit = lambda *a, **k: None
    recorder.set_sample_id(args.offset)
    forced = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    instrumented_steps = run_cached_decode_steps(
        model,
        recorder,
        inputs,
        max_steps=args.decode_steps,
        eos_token_id=None,
        forced_decode_ids=forced,
        capture_logits=True,
    )
    torch.cuda.synchronize()

    if len(instrumented_steps) != len(baseline_logits):
        raise RuntimeError("instrumented step count differs from baseline")
    comparisons: list[dict[str, object]] = []
    all_close = True
    all_argmax_equal = True
    route_contributions = 0
    route_layers: set[int] = set()
    route_topk: set[int] = set()
    for index, (baseline, step) in enumerate(zip(baseline_logits, instrumented_steps)):
        instrumented = step.logits[:, -1:, :]
        difference = (baseline - instrumented).abs()
        close = bool(torch.allclose(baseline, instrumented, rtol=args.rtol, atol=args.atol))
        argmax_equal = bool(
            torch.equal(baseline.argmax(dim=-1), instrumented.argmax(dim=-1))
        )
        all_close = all_close and close
        all_argmax_equal = all_argmax_equal and argmax_equal
        for batch in step.route_batches:
            selected = batch["selected_experts"]
            route_contributions += int(selected.numel())
            route_layers.add(int(batch["layer"]))
            route_topk.add(int(selected.shape[-1]))
        comparisons.append(
            {
                "decode_step": index,
                "executed_token_id": token_ids[index],
                "baseline_cache_length": baseline_cache_lengths[index],
                "instrumented_cache_length": step.cache_length,
                "cache_length_equal": baseline_cache_lengths[index] == step.cache_length,
                "logits_close": close,
                "argmax_equal": argmax_equal,
                "max_abs_error": float(difference.max().item()),
                "mean_abs_error": float(difference.mean().item()),
            }
        )

    cache_equal = all(row["cache_length_equal"] for row in comparisons)
    passed = all_close and all_argmax_equal and cache_equal
    result = {
        "schema": "routeslack-frozen-model-gpu-exactness-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_result": False,
        "scientific_result_eligible": False,
        "gate0_eligible": False,
        "model": args.model,
        "model_key": args.model_key,
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "dataset": args.dataset,
        "split": args.split,
        "offset": args.offset,
        "seq_len_limit": args.seq_len,
        "decode_steps": args.decode_steps,
        "load_seconds": load_seconds,
        "gpu": torch.cuda.get_device_name(model.device),
        "tolerance": {"rtol": args.rtol, "atol": args.atol},
        "executed_token_ids": token_ids,
        "executed_token_ids_sha256": _sha256_json(token_ids),
        "route": {
            "layers": sorted(route_layers),
            "layer_count": len(route_layers),
            "observed_topk": sorted(route_topk),
            "contributions": route_contributions,
        },
        "comparisons": comparisons,
        "checks": {
            "logits_close_all_steps": all_close,
            "argmax_equal_all_steps": all_argmax_equal,
            "cache_length_equal_all_steps": cache_equal,
            "passed": passed,
        },
        "evidence_boundary": (
            "one batch-1 frozen-model GPU exactness qualification; no natural "
            "continuous serving, stage ledger, energy window, SLO, or EP actuator"
        ),
        "remaining_blockers": [
            "natural multi-request continuous batching and per-request KV ownership",
            "routed/dispatched/executed/combined physical ledger",
            "matched completion/output identity in the energy window",
            "real expert-parallel source/target replica actuator",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model_key, "checks": result["checks"]}, sort_keys=True))
    print(f"wrote {output}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

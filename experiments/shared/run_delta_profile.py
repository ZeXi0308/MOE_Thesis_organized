"""Delta profiling: measure marginal KL degradation delta_{l,R,p}.

For each optimizable layer L, each rank R, each precision P, applies the
approximation to ONLY that (layer, rank) and measures end-to-end KL vs full
baseline. This produces the delta table used by the LUT optimizer.

High-sensitivity layers (0, 1, 2, 3, 15 — from layer_sensitivity experiment)
are skipped because they will be fixed to BF16 in the optimizer.

Usage:

    python run_delta_profile.py \
        --model allenai/OLMoE-1B-7B-0924 \
        --num-samples 16 --seq-len 128 \
        --dtype bfloat16 --dataset wikitext2 \
        --num-receiver-groups 4
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from paths import resolve_output_dir
from prompts import get_prompts

PRECISIONS = ["int8", "int4", "drop"]
BYTE_SIZES = {"int8": 1.0, "int4": 0.5, "drop": 0.0}
HIGH_SENS_LAYERS = {0, 1, 2, 3, 15}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--dataset", default="wikitext2", choices=["builtin", "wikitext2"])
    parser.add_argument("--num-receiver-groups", type=int, default=4)
    parser.add_argument("--receiver-mapping", default="contiguous", choices=["contiguous", "mod"])
    return parser.parse_args()


def compute_kl(full_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    full = full_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    p = F.softmax(full, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean").item())


def main() -> None:
    args = parse_args()
    out = resolve_output_dir(args.model, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    texts = get_prompts(args.dataset, args.num_samples)
    tokenizer = load_tokenizer(args.model)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype)
    print(f"model loaded in {load_seconds:.1f}s", flush=True)

    num_layers = len(model.model.layers)
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    optimizable_layers = [l for l in range(num_layers) if l not in HIGH_SENS_LAYERS]
    print(f"num_layers={num_layers}  top_k={top_k}  optimizable={optimizable_layers}", flush=True)

    # ---- baseline ----
    print("running full baseline...", flush=True)
    baseline_logits: list[torch.Tensor] = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
        patch_mixtral_moe(
            model, "full",
            num_receiver_groups=args.num_receiver_groups,
            receiver_mapping=args.receiver_mapping,
        )
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        baseline_logits.append(logits)
    print(f"baseline done ({len(baseline_logits)} samples)", flush=True)

    # ---- delta profiling ----
    rows: list[dict] = []
    total = len(optimizable_layers) * top_k * len(PRECISIONS)
    idx = 0

    for layer in optimizable_layers:
        for rank in range(1, top_k + 1):
            for precision in PRECISIONS:
                idx += 1
                policy_name = f"rank{rank}_{precision}"
                print(f"[{idx}/{total}] L={layer} R={rank} P={precision} ", end="", flush=True)

                recorder = patch_mixtral_moe(
                    model, policy_name,
                    num_receiver_groups=args.num_receiver_groups,
                    receiver_mapping=args.receiver_mapping,
                    target_layer=layer,
                )

                kls: list[float] = []
                for i, text in enumerate(texts):
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
                    with torch.no_grad():
                        logits = model(**inputs).logits.detach().cpu()
                    kls.append(compute_kl(baseline_logits[i], logits))

                mean_kl = sum(kls) / len(kls)

                error_rows = recorder.error_rows()
                rel_mse = 0.0
                for r in error_rows:
                    if r["layer"] == layer:
                        rel_mse = r["relative_mse"]
                        break

                bpe = BYTE_SIZES[precision]
                byte_saving = (2.0 - bpe) / (2.0 * top_k)

                rows.append({
                    "layer": layer,
                    "rank": rank,
                    "precision": precision,
                    "delta_kl": mean_kl,
                    "local_relative_mse": rel_mse,
                    "bytes_per_element": bpe,
                    "byte_saving_per_rank": byte_saving,
                })
                print(f"delta={mean_kl:.6f}  mse={rel_mse:.8f}", flush=True)
                pd.DataFrame(rows).to_csv(out / "delta_profile.partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "delta_profile.csv", index=False)
    print(f"\ndone: {len(rows)} entries -> {out / 'delta_profile.csv'}", flush=True)


if __name__ == "__main__":
    main()

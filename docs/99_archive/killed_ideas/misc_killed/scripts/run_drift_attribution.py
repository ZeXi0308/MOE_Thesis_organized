"""Routing drift attribution experiment (C2).

Decomposes the end-to-end accuracy loss of an approximation strategy into:
  - **numerical error**: loss when routing is locked to the full-precision model's
    expert selection (approx forward + forced routing).
  - **routing drift**: additional loss caused by the router changing its expert
    selection due to perturbed hidden states (free routing minus locked routing).

For each sample and strategy, three forward passes are compared:
  1. Full forward (BF16, no approximation) — caches per-layer routing decisions.
  2. Approx forward (locked routing) — applies approximation but forces the
     router to reuse the full model's expert selection. Isolates numerical error.
  3. Approx forward (free routing) — applies approximation and lets the router
     re-select experts from perturbed hidden states. Includes both error sources.

  drift_contribution = KL_free - KL_locked
  drift_fraction     = drift_contribution / KL_free

Usage example (OLMoE top-8):

    python run_drift_attribution.py \
        --model allenai/OLMoE-1B-7B-0924 \
        --strategies rank8_int4 rank1_int4 uniform_int4 \
        --num-samples 32 --seq-len 128 \
        --dtype bfloat16 --dataset wikitext2 \
        --num-receiver-groups 4
"""
from __future__ import annotations


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from paths import resolve_output_dir
from prompts import get_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["rankk_int4", "rank1_int4", "uniform_int4"],
    )
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

    per_sample_rows: list[dict] = []

    for sample_idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
        print(f"sample {sample_idx + 1}/{len(texts)}", flush=True)

        # 1. Full forward — cache routing decisions per layer
        recorder_full = patch_mixtral_moe(
            model, "full",
            num_receiver_groups=args.num_receiver_groups,
            receiver_mapping=args.receiver_mapping,
            cache_routing=True,
        )
        with torch.no_grad():
            full_logits = model(**inputs).logits.detach().cpu()

        routing_cache = recorder_full.routing_cache

        # 2. For each strategy: locked routing then free routing
        for strategy in args.strategies:
            # --- locked routing: force router to reuse full model's expert selection ---
            patch_mixtral_moe(
                model, strategy,
                num_receiver_groups=args.num_receiver_groups,
                receiver_mapping=args.receiver_mapping,
                lock_routing=True,
                routing_cache=routing_cache,
            )
            with torch.no_grad():
                locked_logits = model(**inputs).logits.detach().cpu()

            # --- free routing: router re-selects from perturbed hidden states ---
            patch_mixtral_moe(
                model, strategy,
                num_receiver_groups=args.num_receiver_groups,
                receiver_mapping=args.receiver_mapping,
            )
            with torch.no_grad():
                free_logits = model(**inputs).logits.detach().cpu()

            kl_locked = compute_kl(full_logits, locked_logits)
            kl_free = compute_kl(full_logits, free_logits)
            drift = kl_free - kl_locked
            drift_frac = drift / max(kl_free, 1e-12)

            per_sample_rows.append({
                "strategy": strategy,
                "sample_idx": sample_idx,
                "kl_free": kl_free,
                "kl_locked": kl_locked,
                "drift_contribution": drift,
                "drift_fraction": drift_frac,
                "numerical_fraction": 1.0 - drift_frac,
            })
            print(
                f"  {strategy}: KL_free={kl_free:.6f}  KL_locked={kl_locked:.6f}  "
                f"drift_frac={drift_frac:.4f}",
                flush=True,
            )

        pd.DataFrame(per_sample_rows).to_csv(out / "drift_per_sample.partial.csv", index=False)

    # ---- final per-sample output ----
    df_samples = pd.DataFrame(per_sample_rows)
    df_samples.to_csv(out / "drift_per_sample.csv", index=False)

    # ---- summary by strategy ----
    summary_rows: list[dict] = []
    for strategy in args.strategies:
        sub = df_samples[df_samples["strategy"] == strategy]
        summary_rows.append({
            "strategy": strategy,
            "mean_kl_free": float(sub["kl_free"].mean()),
            "mean_kl_locked": float(sub["kl_locked"].mean()),
            "mean_drift_contribution": float(sub["drift_contribution"].mean()),
            "mean_drift_fraction": float(sub["drift_fraction"].mean()),
            "mean_numerical_fraction": float(sub["numerical_fraction"].mean()),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out / "drift_summary.csv", index=False)

    # ---- report ----
    report = f"""# Routing Drift Attribution Report

model: `{args.model}`
samples: `{args.num_samples}`
seq_len: `{args.seq_len}`
dtype: `{args.dtype}`
dataset: `{args.dataset}`

## Method

For each sample and strategy, three forward passes are compared:

1. **Full forward** (BF16, no approximation) — caches per-layer routing decisions.
2. **Approx (locked routing)** — applies approximation but forces the router to
   reuse the full model's expert selection. Isolates pure numerical error.
3. **Approx (free routing)** — applies approximation and lets the router
   re-select experts from perturbed hidden states. Includes numerical error
   + routing drift.

**drift contribution** = KL_free - KL_locked
**drift fraction** = drift_contribution / KL_free

## Summary

| strategy | mean KL (free) | mean KL (locked) | drift contribution | drift fraction | numerical fraction |
|---|---|---|---|---|---|
"""
    for _, r in df_summary.iterrows():
        report += (
            f"| {r['strategy']} | {r['mean_kl_free']:.6f} | {r['mean_kl_locked']:.6f} | "
            f"{r['mean_drift_contribution']:.6f} | {r['mean_drift_fraction']:.4f} | "
            f"{r['mean_numerical_fraction']:.4f} |\n"
        )

    report += "\n## Interpretation\n\n"
    for _, r in df_summary.iterrows():
        frac = float(r["mean_drift_fraction"])
        if frac > 0.6:
            verdict = (
                "routing drift dominates — the linear additive delta model needs "
                "cascading correction or routing alignment (EAQuant-style)"
            )
        elif frac > 0.3:
            verdict = (
                "moderate routing drift — single-layer delta profiles may "
                "underestimate cascading loss; validate additivity with a sanity check"
            )
        else:
            verdict = (
                "numerical error dominates — the linear additive delta model is "
                "well justified; single-layer profiles are reliable"
            )
        report += f"- **{r['strategy']}**: drift fraction {frac:.2%} → {verdict}\n"

    (out / "drift_attribution_report.md").write_text(report, encoding="utf-8")
    print(f"\nresults saved to {out}/", flush=True)


if __name__ == "__main__":
    main()

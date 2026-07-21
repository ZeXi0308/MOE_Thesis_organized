"""LUT Evaluation: apply LUT end-to-end and measure actual KL / byte saving.

Loads a LUT from JSON, applies it via the LUT policy, and measures:
  - Actual end-to-end KL vs full baseline
  - Actual byte saving (from receiver stats)
  - Predicted KL (from delta profile, for additivity check)
  - PPL delta

Compares multiple LUTs in one run.

Usage:

    python run_lut_evaluation.py \
        --model allenai/OLMoE-1B-7B-0924 \
        --delta-csv outputs/delta_profile/olmoe_wikitext16_g4/delta_profile.csv \
        --lut-dir outputs/lut_optimizer/olmoe \
        --epsilon 0.1 \
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
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from paths import resolve_output_dir
from prompts import get_prompts

HIGH_SENS_LAYERS = {0, 1, 2, 3, 15}
NON_BF16 = ["int8", "int4", "drop"]
BYTE_SIZES = {"bf16": 2.0, "int8": 1.0, "int4": 0.5, "drop": 0.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--delta-csv", required=True)
    p.add_argument("--lut-dir", required=True)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16", "auto"])
    p.add_argument("--dataset", default="wikitext2", choices=["builtin", "wikitext2"])
    p.add_argument("--num-receiver-groups", type=int, default=4)
    p.add_argument("--receiver-mapping", default="contiguous", choices=["contiguous", "mod"])
    return p.parse_args()


def compute_kl(full_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    full = full_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    p_ = F.softmax(full, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    return float(F.kl_div(log_q, p_, reduction="batchmean").item())


def compute_ppl(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    if logits.shape[1] < 2:
        return float("nan")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return float(torch.exp(loss).item())


def load_lut(path: Path) -> dict:
    raw = json.loads(path.read_text())
    return {tuple(int(x) for x in k.split(",")): v for k, v in raw.items()}


def load_delta(path: str) -> dict:
    df = pd.read_csv(path)
    d = {}
    for _, row in df.iterrows():
        d[(int(row["layer"]), int(row["rank"]), row["precision"])] = float(row["delta_kl"])
    return d


def predict_kl(lut: dict, delta: dict) -> float:
    """Predict KL from delta profile (additivity assumption)."""
    freq_path = Path("outputs/main_experiments/olmoe_wikitext256_g4/receiver_rank_share.csv")
    if not freq_path.exists():
        return -1.0
    freq_df = pd.read_csv(freq_path)
    freq = {}
    for _, row in freq_df.iterrows():
        freq[(int(row["layer"]), int(row["receiver_group"]), int(row["rank"]))] = int(row["count"])
    total_freq = sum(freq.values())
    pred = 0.0
    for (l, r, R), p in lut.items():
        if p in NON_BF16:
            d = delta.get((l, R, p), 0.0)
            f = freq.get((l, r, R), 0)
            pred += d * f / total_freq
    return pred


def run_lut_policy(model, tokenizer, lut, texts, seq_len, num_receiver_groups, receiver_mapping):
    """Apply LUT policy and return (logits_list, byte_saving, bottleneck_saving, group_traffic)."""
    recorder = patch_mixtral_moe(
        model, "lut",
        num_receiver_groups=num_receiver_groups,
        receiver_mapping=receiver_mapping,
        lut=lut,
    )
    all_logits = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        all_logits.append(logits)

    byte_saving = recorder.total_byte_saving() if recorder.receiver_rank_stats else 0.0

    group_traffic = [0.0] * num_receiver_groups
    group_baseline = [0.0] * num_receiver_groups
    for (layer_id, group, rank_idx), stat in recorder.receiver_rank_stats.items():
        group_traffic[group] += stat.policy_bytes
        group_baseline[group] += stat.full_bytes
    bottleneck_saving = 1.0 - max(group_traffic) / max(max(group_baseline), 1e-12)

    return all_logits, byte_saving, bottleneck_saving, group_traffic


def main():
    args = parse_args()
    out = resolve_output_dir(args.model, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    texts = get_prompts(args.dataset, args.num_samples)
    tokenizer = load_tokenizer(args.model)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype)
    print(f"model loaded in {load_seconds:.1f}s", flush=True)

    delta = load_delta(args.delta_csv)

    # ---- baseline ----
    print("running full baseline...", flush=True)
    baseline_logits = []
    baseline_ppls = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
        patch_mixtral_moe(model, "full",
                          num_receiver_groups=args.num_receiver_groups,
                          receiver_mapping=args.receiver_mapping)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        baseline_logits.append(logits)
        baseline_ppls.append(compute_ppl(logits, inputs["input_ids"]))
    full_ppl = sum(baseline_ppls) / len(baseline_ppls)
    print(f"baseline PPL: {full_ppl:.4f}", flush=True)

    # ---- collect LUTs to evaluate ----
    lut_dir = Path(args.lut_dir)
    luts_to_eval = {}

    for method in ["milp", "rank_only", "greedy"]:
        p = lut_dir / f"lut_{method}_eps{args.epsilon}.json"
        if p.exists():
            luts_to_eval[method] = load_lut(p)

    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    num_layers = len(model.model.layers)
    luts_to_eval["all_bf16"] = {(l, r, R): "bf16"
                                 for l in range(num_layers)
                                 for r in range(args.num_receiver_groups)
                                 for R in range(1, top_k + 1)}
    luts_to_eval["uniform_int4"] = {(l, r, R): "int4"
                                     for l in range(num_layers)
                                     for r in range(args.num_receiver_groups)
                                     for R in range(1, top_k + 1)}

    # ---- evaluate each LUT ----
    rows = []
    for name, lut in luts_to_eval.items():
        print(f"evaluating {name}...", end=" ", flush=True)
        logits_list, byte_saving, bottleneck_saving, group_traffic = run_lut_policy(
            model, tokenizer, lut, texts, args.seq_len,
            args.num_receiver_groups, args.receiver_mapping,
        )

        kls = [compute_kl(baseline_logits[i], logits_list[i]) for i in range(len(texts))]
        ppls = []
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
            ppls.append(compute_ppl(logits_list[i], inputs["input_ids"]))
        mean_ppl = sum(ppls) / len(ppls)

        actual_kl = sum(kls) / len(kls)
        pred_kl = predict_kl(lut, delta)

        row = {
            "method": name,
            "actual_kl": actual_kl,
            "predicted_kl": pred_kl,
            "additivity_ratio": actual_kl / max(pred_kl, 1e-12) if pred_kl > 0 else 0,
            "mean_ppl": mean_ppl,
            "ppl_delta": mean_ppl - full_ppl,
            "byte_saving": byte_saving,
            "bottleneck_saving": bottleneck_saving,
        }
        rows.append(row)
        print(f"KL={actual_kl:.6f}  pred={pred_kl:.6f}  ratio={row['additivity_ratio']:.2f}  "
              f"byte_save={byte_saving:.4f}  bottleneck_save={bottleneck_saving:.4f}", flush=True)

    # ---- save ----
    df = pd.DataFrame(rows)
    df.to_csv(out / "lut_evaluation.csv", index=False)

    report = f"""# LUT Evaluation Report

model: `{args.model}`
samples: `{args.num_samples}`
seq_len: `{args.seq_len}`
dtype: `{args.dtype}`
epsilon: `{args.epsilon}`
baseline PPL: `{full_ppl:.4f}`

## Results

| method | actual KL | predicted KL | additivity ratio | PPL delta | byte saving | bottleneck saving |
|---|---|---|---|---|---|---|
"""
    for _, r in df.iterrows():
        report += (
            f"| {r['method']} | {r['actual_kl']:.6f} | {r['predicted_kl']:.6f} | "
            f"{r['additivity_ratio']:.2f} | {r['ppl_delta']:.6f} | "
            f"{r['byte_saving']:.4f} | {r['bottleneck_saving']:.4f} |\n"
        )

    report += """
## Additivity Check

The additivity ratio (actual KL / predicted KL) measures how well the linear
additive delta model predicts the end-to-end accuracy loss.

- ratio ≈ 1.0: additivity holds well
- ratio > 1.5: cascading effects (routing drift) cause underestimation
- ratio < 0.8: overestimation (delta profiles are conservative)
"""
    (out / "lut_evaluation_report.md").write_text(report, encoding="utf-8")
    print(f"\nresults saved to {out}/", flush=True)


if __name__ == "__main__":
    main()

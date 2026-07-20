from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from paths import resolve_output_dir
from policies import make_policy
from prompts import get_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full", "uniform_int4", "rankk_int4", "rank1_int4"],
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--dataset", default="builtin", choices=["builtin", "wikitext2"])
    parser.add_argument("--num-receiver-groups", type=int, default=1)
    parser.add_argument("--receiver-mapping", default="contiguous", choices=["contiguous", "mod"])
    return parser.parse_args()


def tokenized_inputs(tokenizer, texts: list[str], seq_len: int):
    for text in texts:
        yield tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)


def compute_ppl(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    if logits.shape[1] < 2:
        return float("nan")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return float(torch.exp(loss).item())


def compute_kl(full_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    full = full_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    p = F.softmax(full, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    kl = F.kl_div(log_q, p, reduction="batchmean")
    return float(kl.item())


def run_strategy_loaded(
    model,
    tokenizer,
    strategy: str,
    texts: list[str],
    seq_len: int,
    baseline_logits=None,
    collect_logits: bool = False,
    num_receiver_groups: int = 1,
    receiver_mapping: str = "contiguous",
):
    recorder = patch_mixtral_moe(
        model,
        policy_name=strategy,
        num_receiver_groups=num_receiver_groups,
        receiver_mapping=receiver_mapping,
    )
    policy = make_policy(strategy)

    ppls: list[float] = []
    kls: list[float] = []
    all_logits = [] if collect_logits else None
    top_k = None

    for idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len)):
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        if all_logits is not None:
            all_logits.append(logits)
        ppls.append(compute_ppl(logits, inputs["input_ids"]))
        if baseline_logits is not None:
            kls.append(compute_kl(baseline_logits[idx], logits))

    error_rows = recorder.error_rows()
    rel_mse = sum(r["sq_error"] for r in error_rows) / max(sum(r["sq_full"] for r in error_rows), 1e-12)
    top_k = recorder.top_k or int(getattr(model.config, "num_experts_per_tok", 0))
    byte_saving = recorder.total_byte_saving() if recorder.receiver_rank_stats else policy.byte_saving(top_k)
    return {
        "strategy": strategy,
        "samples": len(texts),
        "seq_len": seq_len,
        "top_k": top_k,
        "byte_saving": byte_saving,
        "mean_ppl": sum(ppls) / max(len(ppls), 1),
        "mean_kl_vs_full": (sum(kls) / len(kls)) if kls else 0.0,
        "local_relative_mse": rel_mse,
    }, all_logits


def write_partial_results(rows: list[dict], out: Path) -> None:
    df = pd.DataFrame(rows)
    if not df.empty and "full" in set(df["strategy"]):
        full_ppl = float(df[df["strategy"] == "full"]["mean_ppl"].iloc[0])
        df["ppl_delta"] = df["mean_ppl"] - full_ppl
    df.to_csv(out / "approx_results.partial.csv", index=False)


def main() -> None:
    args = parse_args()
    out = resolve_output_dir(args.model, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = get_prompts(args.dataset, args.num_samples)
    tokenizer = load_tokenizer(args.model)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype)
    print({"model_loaded_seconds": load_seconds, "model": args.model, "dtype": args.dtype}, flush=True)

    rows = []
    baseline_logits = None
    if "full" not in args.strategies:
        args.strategies = ["full"] + args.strategies

    for strategy in args.strategies:
        row, logits = run_strategy_loaded(
            model,
            tokenizer,
            strategy,
            texts,
            args.seq_len,
            baseline_logits=baseline_logits,
            collect_logits=(strategy == "full"),
            num_receiver_groups=args.num_receiver_groups,
            receiver_mapping=args.receiver_mapping,
        )
        rows.append(row)
        if strategy == "full":
            baseline_logits = logits
        write_partial_results(rows, out)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "approx_results.csv", index=False)

    full_ppl = float(df[df["strategy"] == "full"]["mean_ppl"].iloc[0])
    df["ppl_delta"] = df["mean_ppl"] - full_ppl
    df.to_csv(out / "approx_results.csv", index=False)

    rankk = df[df["strategy"] == "rankk_int4"]
    rank1 = df[df["strategy"] == "rank1_int4"]
    if not rankk.empty and not rank1.empty:
        rank_msg = (
            "Rank-k INT4 is better than Rank-1 INT4 on KL."
            if float(rankk["mean_kl_vs_full"].iloc[0]) < float(rank1["mean_kl_vs_full"].iloc[0])
            else "Rank-k INT4 is not better than Rank-1 INT4 on KL in this run."
        )
    else:
        rank_msg = "Rank-k vs Rank-1 INT4 comparison was not available."

    report = f"""# Rank-Aware Approximation Report

model: `{args.model}`
samples: `{args.num_samples}`
dataset: `{args.dataset}`
seq_len: `{args.seq_len}`
dtype: `{args.dtype}`

{rank_msg}

See `approx_results.csv` for byte saving, KL, PPL delta, and local relative MSE.
"""
    (out / "rank_aware_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

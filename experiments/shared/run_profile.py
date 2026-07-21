from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from capture_moe import patch_mixtral_moe
from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from paths import resolve_output_dir
from prompts import get_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--dataset", default="builtin", choices=["builtin", "wikitext2"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = resolve_output_dir(args.model, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model)
    model, _ = load_model(args.model, dtype_name=args.dtype)
    recorder = patch_mixtral_moe(model, policy_name="full")

    prompts = get_prompts(args.dataset, args.num_samples)
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
        with torch.no_grad():
            model(**inputs)

    rows = recorder.rank_rows()
    df = pd.DataFrame(rows)
    df.to_csv(out / "rank_share_by_layer.csv", index=False)

    top_k = recorder.top_k or 0
    rankk = df[df["rank"] == top_k]
    if not rankk.empty:
        rankk_median = float(rankk["median_share"].median())
        ratio_median = float(rankk["rank1_over_rankk_median"].median())
    else:
        rankk_median = float("nan")
        ratio_median = float("nan")

    if rankk_median < 0.10 and ratio_median > 3:
        verdict = "强成立"
    elif 0.10 <= rankk_median <= 0.20:
        verdict = "弱成立"
    else:
        verdict = "不成立或证据不足"

    report = f"""# C1 Long-Tail Report

model: `{args.model}`
samples: `{args.num_samples}`
dataset: `{args.dataset}`
seq_len: `{args.seq_len}`
top_k: `{top_k}`
dtype: `{args.dtype}`

- rank-k median share across layers: `{rankk_median:.6f}`
- rank1/rankk median ratio across layers: `{ratio_median:.6f}`
- C1 verdict: **{verdict}**

Interpretation:

- 强成立：多数层 rank-k contribution 很小，可以继续保留 drop / aggressive quantization。
- 弱成立：可以做差分量化，但 drop 要谨慎。
- 不成立或证据不足：不要主打 top-k internal long-tail，先收缩为 rank 是 deployable importance proxy。
"""
    (out / "c1_long_tail_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

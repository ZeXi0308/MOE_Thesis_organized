"""Compare PLTB under INT4 proxy, MXFP4, and NVFP4-style fake quantization."""
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
import math
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import load_model, load_tokenizer
from prompts import get_prompts
from run_layer_budget_experiment import build_lut, dataframe_to_markdown, run_logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--dataset-split", default="validation")
    p.add_argument("--test-samples", type=int, default=32)
    p.add_argument("--test-offset", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--allocations", required=True)
    p.add_argument("--allocation-strategy", default="kl_profile_2_4_6")
    p.add_argument("--formats", default="int4,mxfp4,nvfp4")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/fp4_formats",
    )
    return p.parse_args()


def vector_wire_bytes(precision: str, hidden_size: int) -> int:
    if precision in ("full", "bf16"):
        return 2 * hidden_size
    if precision == "fp8":
        return hidden_size + math.ceil(hidden_size / 128)  # UE8M0 scale per 128
    if precision == "int4":
        return math.ceil(hidden_size / 2) + 4  # current per-row FP32 scale proxy
    if precision == "mxfp4":
        return math.ceil(hidden_size / 2) + math.ceil(hidden_size / 32)
    if precision == "nvfp4":
        return math.ceil(hidden_size / 2) + math.ceil(hidden_size / 16) + 4
    raise ValueError(f"unknown precision: {precision}")


def metadata_saving(
    tail_counts: list[int], top_k: int, hidden_size: int, tail_precision: str
) -> float:
    fp8_bytes = vector_wire_bytes("fp8", hidden_size)
    tail_bytes = vector_wire_bytes(tail_precision, hidden_size)
    bf16_bytes = vector_wire_bytes("bf16", hidden_size)
    actual = sum((top_k - count) * fp8_bytes + count * tail_bytes for count in tail_counts)
    baseline = len(tail_counts) * top_k * bf16_bytes
    return 1.0 - actual / baseline


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    allocation_doc = json.loads(Path(args.allocations).read_text(encoding="utf-8"))
    pltb_counts = allocation_doc["allocations"][args.allocation_strategy]
    base_tail = int(allocation_doc["base_tail"])
    fixed_counts = [base_tail] * len(pltb_counts)
    formats = [value.strip() for value in args.formats.split(",") if value.strip()]

    texts = get_prompts(
        args.dataset,
        args.test_samples,
        offset=args.test_offset,
        split=args.dataset_split,
    )
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model, dtype_name=args.dtype, local_files_only=args.offline
    )
    num_layers = len(model.model.layers)
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    hidden_size = int(model.config.hidden_size)
    if len(pltb_counts) != num_layers:
        raise ValueError(f"allocation has {len(pltb_counts)} layers, model has {num_layers}")
    print(
        f"model loaded in {load_seconds:.1f}s; layers={num_layers}, top_k={top_k}, hidden={hidden_size}",
        flush=True,
    )

    full_metrics, full_logits, _ = run_logits(
        model, tokenizer, texts, args.seq_len, "full", 1, args.test_offset
    )
    rows: list[dict[str, float | int | str]] = []
    full_row = full_metrics.bootstrap_summary(args.bootstrap)
    full_row.update(
        {
            "strategy": "full",
            "format": "bf16",
            "raw_payload_saving_vs_bf16": 0.0,
            "metadata_aware_wire_saving_vs_bf16": 0.0,
            "ppl_delta_vs_full": 0.0,
        }
    )
    rows.append(full_row)

    uniform_metrics, _, uniform_recorder = run_logits(
        model,
        tokenizer,
        texts,
        args.seq_len,
        "uniform_fp8",
        1,
        args.test_offset,
        baseline_logits=full_logits,
    )
    uniform_row = uniform_metrics.bootstrap_summary(args.bootstrap)
    uniform_row.update(
        {
            "strategy": "uniform_fp8",
            "format": "fp8",
            "raw_payload_saving_vs_bf16": uniform_recorder.total_byte_saving(),
            "metadata_aware_wire_saving_vs_bf16": 1.0
            - vector_wire_bytes("fp8", hidden_size) / vector_wire_bytes("bf16", hidden_size),
            "ppl_delta_vs_full": uniform_metrics.corpus_ppl - full_metrics.corpus_ppl,
        }
    )
    rows.append(uniform_row)

    for precision in formats:
        for allocation_name, counts in (("fixed", fixed_counts), ("pltb", pltb_counts)):
            strategy = f"{allocation_name}_{precision}"
            print(f"running {strategy}...", flush=True)
            lut = build_lut(counts, top_k, 1)
            for key in list(lut):
                if lut[key] == "int4":
                    lut[key] = precision
            metrics, _, recorder = run_logits(
                model,
                tokenizer,
                texts,
                args.seq_len,
                "lut",
                1,
                args.test_offset,
                baseline_logits=full_logits,
                lut=lut,
            )
            row = metrics.bootstrap_summary(args.bootstrap)
            row.update(
                {
                    "strategy": strategy,
                    "format": precision,
                    "raw_payload_saving_vs_bf16": recorder.total_byte_saving(),
                    "metadata_aware_wire_saving_vs_bf16": metadata_saving(
                        counts, top_k, hidden_size, precision
                    ),
                    "ppl_delta_vs_full": metrics.corpus_ppl - full_metrics.corpus_ppl,
                }
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "format_comparison.partial.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(out / "format_comparison.csv", index=False)
    columns = [
        "strategy",
        "raw_payload_saving_vs_bf16",
        "metadata_aware_wire_saving_vs_bf16",
        "corpus_ppl",
        "ppl_delta_vs_full",
        "mean_token_kl",
        "mean_token_kl_ci_low",
        "mean_token_kl_ci_high",
    ]
    report = f"""# FP4 Format Comparison for PLTB

## Boundary

This is fake quantization, not a native FP4 communication kernel.  NVFP4 uses
E2M1 values, per-16 E4M3 block scales, and one FP32 global scale per vector.
MXFP4 uses E2M1 values with a per-32 power-of-two scale.  Metadata-aware wire
bytes include scale bytes but exclude message alignment and collective headers.

## Setup

- model: `{args.model}`; hidden size: `{hidden_size}`; top-k: `{top_k}`
- test: `{args.dataset}:{args.dataset_split}` offset `{args.test_offset}`, n=`{args.test_samples}`
- allocation: `{args.allocation_strategy}` = `{pltb_counts}`

## Results

{dataframe_to_markdown(results, columns)}
"""
    (out / "format_comparison_report.md").write_text(report, encoding="utf-8")
    print(results[columns].to_string(index=False), flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

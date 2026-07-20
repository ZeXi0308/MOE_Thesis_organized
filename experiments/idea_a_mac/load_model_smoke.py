from __future__ import annotations

from pathlib import Path
import argparse
import time

import torch

from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from paths import resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "auto"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = resolve_output_dir(args.model, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype)
    text = "Mixture of Experts models route tokens to experts."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    forward_seconds = time.time() - start

    cfg = model.config
    num_experts = getattr(cfg, "num_local_experts", getattr(cfg, "num_experts", None))
    top_k = getattr(cfg, "num_experts_per_tok", None)
    report = f"""# Model Smoke Result

model: `{args.model}`

- model_type: `{cfg.model_type}`
- layers: `{cfg.num_hidden_layers}`
- hidden_size: `{cfg.hidden_size}`
- intermediate_size: `{cfg.intermediate_size}`
- num_experts: `{num_experts}`
- num_experts_per_tok: `{top_k}`
- load_seconds: `{load_seconds:.2f}`
- forward_seconds: `{forward_seconds:.2f}`
- logits_shape: `{tuple(outputs.logits.shape)}`
- torch_device: `cpu`
- dtype: `{args.dtype}`
"""
    (out / "model_smoke_result.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

"""Lightweight (non-sealed-ceremony) real-route capture on a fresh corpus.

This is exploratory infrastructure for the receiver-aware-v2 systematic
experiment, not a confirmatory/sealed capture.  It captures exact top-k
routes for N documents from a chosen dataset/split/offset by running the real
model forward pass with ``capture_moe.patch_mixtral_moe(..., record_routes=True)``.

Unlike ``capture_route_fidelity_p0b.py`` (which requires a fully frozen JSON
config with a historical-exclusion registry before it will run), this script
has no such ceremony -- it is meant for switching to a fresh corpus
(wikitext-103) where article offset 0 has never been touched by any prior
experiment in this project, so a heavyweight exclusion registry is not
needed. Do not use this script's output as a sealed confirmatory holdout for
a primary paper claim; use it only for the receiver-aware-v2 mechanism study.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch

from capture_moe import patch_mixtral_moe
from modeling import load_model, load_tokenizer
from prompts import get_prompts

ROUTE_COLUMNS = (
    "sample_id",
    "layer",
    "token_position",
    "rank",
    "expert_id",
    "gate_weight",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="wikitext103_docs")
    p.add_argument("--split", default="test")
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    texts = get_prompts(args.dataset, args.samples, offset=args.offset, split=args.split)
    doc_hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]

    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype, local_files_only=args.offline)

    recorder = patch_mixtral_moe(model, "full", num_receiver_groups=1, record_routes=True)
    # Route capture does not need the other diagnostics; keep them as no-ops
    # so memory does not grow with document count.
    recorder.update_contrib = lambda *a, **k: None
    recorder.update_receiver = lambda *a, **k: None
    recorder.update_error = lambda *a, **k: None
    recorder.update_pair_audit = lambda *a, **k: None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(ROUTE_COLUMNS)
        for local_idx, text in enumerate(texts):
            sample_id = args.offset + local_idx
            recorder.set_sample_id(sample_id)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            for batch in recorder.route_batches:
                experts = batch["selected_experts"]
                weights = batch["routing_weights"]
                layer_id = int(batch["layer"])
                for token_position in range(experts.shape[0]):
                    for rank_idx in range(experts.shape[1]):
                        writer.writerow((
                            sample_id,
                            layer_id,
                            token_position,
                            rank_idx + 1,
                            int(experts[token_position, rank_idx].item()),
                            float(weights[token_position, rank_idx].item()),
                        ))
                        rows_written += 1
            recorder.route_batches.clear()
            recorder.routing_weight_batches.clear()
            print(f"  captured sample {sample_id} ({local_idx + 1}/{len(texts)})", flush=True)

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "samples": args.samples,
        "offset": args.offset,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "document_sha256": doc_hashes,
        "rows_written": rows_written,
        "load_seconds": load_seconds,
        "evidence_boundary": (
            "exploratory route capture on wikitext-103 (never used by any "
            "prior experiment in this project); not a sealed/frozen "
            "confirmatory holdout"
        ),
    }
    (out_path.parent / (out_path.stem + "_meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {rows_written} rows to {out_path}")


if __name__ == "__main__":
    main()

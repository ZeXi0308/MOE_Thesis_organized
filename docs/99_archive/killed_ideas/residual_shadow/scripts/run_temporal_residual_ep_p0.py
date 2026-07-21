"""Mac P0 for stateful temporal-residual MoE combine communication.

This experiment asks whether consecutive autoregressive token positions revisit
the same expert often enough, and whether a closed-loop residual representation
is more accurate than quantizing the current expert output directly at the same
logical byte budget.

It is a numerical and logical-payload experiment.  It does not measure a GPU
kernel, collective, NIC traffic, TPOT, or P99 latency.
"""


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

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MethodType

import pandas as pd
import torch
import torch.nn.functional as F

from fake_quant import apply_precision
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from prompts import get_prompts


MODES = (
    "full",
    "uniform_fp8",
    "uniform_mxfp4",
    "revisit_abs_mxfp4",
    "temporal_delta_mxfp4",
)


def vector_wire_bytes(precision: str, hidden: int) -> int:
    """Logical payload accounting aligned with the existing Idea A scripts."""
    if precision in ("full", "bf16"):
        return 2 * hidden
    if precision == "fp8":
        return hidden + 4  # one FP32 row scale in the current proxy
    if precision == "mxfp4":
        return math.ceil(hidden / 2) + math.ceil(hidden / 32)
    raise ValueError(precision)


@dataclass
class CodecStats:
    pairs: int = 0
    revisits: int = 0
    same_rank_revisits: int = 0
    payload_bytes: int = 0
    mode_bits: int = 0
    raw_sq: float = 0.0
    temporal_residual_sq: float = 0.0
    direct_hit_error_sq: float = 0.0
    temporal_hit_error_sq: float = 0.0
    combine_full_sq: float = 0.0
    combine_error_sq: float = 0.0

    def merge(self, other: "CodecStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def row(self, mode: str, layer: int | str = "all") -> dict[str, int | float | str]:
        payload = self.payload_bytes + math.ceil(self.mode_bits / 8)
        return {
            "mode": mode,
            "layer": layer,
            "pairs": self.pairs,
            "revisits": self.revisits,
            "revisit_rate": self.revisits / max(self.pairs, 1),
            "same_rank_revisit_rate": self.same_rank_revisits / max(self.pairs, 1),
            "payload_bytes": payload,
            "payload_bytes_per_pair": payload / max(self.pairs, 1),
            "residual_to_raw_energy": self.temporal_residual_sq / max(self.raw_sq, 1e-30),
            "direct_hit_rel_mse": self.direct_hit_error_sq / max(self.raw_sq, 1e-30),
            "temporal_hit_rel_mse": self.temporal_hit_error_sq / max(self.raw_sq, 1e-30),
            "combine_relative_mse": self.combine_error_sq / max(self.combine_full_sq, 1e-30),
        }


def _revisit_mapping(
    current: torch.Tensor, previous: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hit mask, previous rank index, and same-rank hit mask."""
    matches = current[:, None] == previous[None, :]
    hit = matches.any(dim=1)
    previous_rank = matches.to(torch.int64).argmax(dim=1)
    ranks = torch.arange(current.shape[0], device=current.device)
    same_rank = hit & (previous_rank == ranks)
    return hit, previous_rank, same_rank


def encode_sequence(
    raw: torch.Tensor,
    selected_experts: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, CodecStats]:
    """Encode one layer's [token, top-k, hidden] expert outputs.

    The temporal predictor is closed loop: both endpoints cache the previously
    reconstructed output, and the sender encodes ``raw - cached``.  Therefore
    the current reconstruction error is the current residual quantization
    error rather than an open-loop sum of all prior errors.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if raw.ndim != 3 or selected_experts.shape != raw.shape[:2]:
        raise ValueError("expected raw [token, top_k, hidden] and matching routes")

    tokens, top_k, hidden = raw.shape
    out = torch.empty_like(raw)
    stats = CodecStats(pairs=tokens * top_k)
    full_bytes = vector_wire_bytes("full", hidden)
    fp8_bytes = vector_wire_bytes("fp8", hidden)
    fp4_bytes = vector_wire_bytes("mxfp4", hidden)

    previous_selected: torch.Tensor | None = None
    previous_raw: torch.Tensor | None = None
    previous_reconstructed: torch.Tensor | None = None

    for token_idx in range(tokens):
        current = raw[token_idx]
        if previous_selected is None:
            hit = torch.zeros(top_k, dtype=torch.bool, device=raw.device)
            previous_rank = torch.zeros(top_k, dtype=torch.long, device=raw.device)
            same_rank = hit
            predictor_raw = torch.zeros_like(current)
            predictor_closed = torch.zeros_like(current)
        else:
            hit, previous_rank, same_rank = _revisit_mapping(
                selected_experts[token_idx], previous_selected
            )
            predictor_raw = previous_raw[previous_rank]
            predictor_closed = previous_reconstructed[previous_rank]

        stats.revisits += int(hit.sum().item())
        stats.same_rank_revisits += int(same_rank.sum().item())

        direct_fp4 = apply_precision(current, "mxfp4")
        closed_delta = current - predictor_closed
        temporal = predictor_closed + apply_precision(closed_delta, "mxfp4")
        keyframe = apply_precision(current, "fp8")

        if hit.any():
            oracle_delta = current - predictor_raw
            oracle_temporal = predictor_raw + apply_precision(oracle_delta, "mxfp4")
            stats.raw_sq += float(current[hit].float().square().sum().item())
            stats.temporal_residual_sq += float(
                oracle_delta[hit].float().square().sum().item()
            )
            stats.direct_hit_error_sq += float(
                (direct_fp4[hit].float() - current[hit].float()).square().sum().item()
            )
            stats.temporal_hit_error_sq += float(
                (oracle_temporal[hit].float() - current[hit].float()).square().sum().item()
            )

        if mode == "full":
            reconstructed = current
            stats.payload_bytes += top_k * full_bytes
        elif mode == "uniform_fp8":
            reconstructed = keyframe
            stats.payload_bytes += top_k * fp8_bytes
        elif mode == "uniform_mxfp4":
            reconstructed = direct_fp4
            stats.payload_bytes += top_k * fp4_bytes
        elif mode == "revisit_abs_mxfp4":
            reconstructed = torch.where(hit[:, None], direct_fp4, keyframe)
            stats.payload_bytes += int(hit.sum().item()) * fp4_bytes
            stats.payload_bytes += int((~hit).sum().item()) * fp8_bytes
            stats.mode_bits += top_k
        else:
            reconstructed = torch.where(hit[:, None], temporal, keyframe)
            stats.payload_bytes += int(hit.sum().item()) * fp4_bytes
            stats.payload_bytes += int((~hit).sum().item()) * fp8_bytes
            stats.mode_bits += top_k

        out[token_idx] = reconstructed
        previous_selected = selected_experts[token_idx]
        previous_raw = current
        previous_reconstructed = reconstructed

    return out, stats


def _raw_expert_outputs(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if batch_size != 1:
        raise ValueError("temporal P0 currently requires batch_size=1")
    flat = hidden_states.reshape(-1, hidden_dim)
    router_logits = self.gate(flat)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, self.top_k, dim=-1
    )
    if getattr(self, "_temporal_normalize_topk", False) or getattr(
        self, "norm_topk_prob", False
    ):
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(flat.dtype)

    raw = torch.zeros(
        (flat.shape[0], self.top_k, hidden_dim),
        dtype=flat.dtype,
        device=flat.device,
    )
    expert_mask = F.one_hot(
        selected_experts, num_classes=self.num_experts
    ).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        rank_idx, token_idx = torch.where(expert_mask[expert_idx])
        current = flat[None, token_idx].reshape(-1, hidden_dim)
        raw[token_idx, rank_idx, :] = self.experts[expert_idx](current)
    return (
        batch_size,
        sequence_length,
        hidden_dim,
        router_logits,
        routing_weights,
        selected_experts,
        raw,
    )


def _combine_expert_order(
    outputs: torch.Tensor,
    weights: torch.Tensor,
    selected: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Match the pretrained OLMoE expert-id accumulation order."""
    total_tokens, _top_k, hidden = outputs.shape
    final = torch.zeros(
        (total_tokens, hidden), dtype=outputs.dtype, device=outputs.device
    )
    expert_mask = F.one_hot(selected, num_classes=num_experts).permute(2, 1, 0)
    for expert_idx in range(num_experts):
        rank_idx, token_idx = torch.where(expert_mask[expert_idx])
        current = outputs[token_idx, rank_idx, :] * weights[
            token_idx, rank_idx, None
        ]
        final.index_add_(0, token_idx, current.to(outputs.dtype))
    return final


def _patched_forward(self, hidden_states: torch.Tensor):
    (
        batch_size,
        sequence_length,
        hidden_dim,
        router_logits,
        routing_weights,
        selected_experts,
        raw,
    ) = _raw_expert_outputs(self, hidden_states)
    mode = self._temporal_mode
    encoded, stats = encode_sequence(raw, selected_experts, mode)
    full_final = _combine_expert_order(
        raw, routing_weights, selected_experts, self.num_experts
    )
    encoded_final = _combine_expert_order(
        encoded, routing_weights, selected_experts, self.num_experts
    )
    stats.combine_full_sq += float(full_final.float().square().sum().item())
    stats.combine_error_sq += float(
        (encoded_final.float() - full_final.float()).square().sum().item()
    )
    self._temporal_stats[self._temporal_layer_id].merge(stats)
    return encoded_final.reshape(batch_size, sequence_length, hidden_dim), router_logits


def _layer_moe(layer):
    if hasattr(layer, "block_sparse_moe"):
        return layer.block_sparse_moe, True
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        return layer.mlp, False
    raise TypeError(f"unsupported MoE layer structure: {type(layer)}")


def patch_model(model) -> None:
    for layer_id, layer in enumerate(model.model.layers):
        moe, force_normalize = _layer_moe(layer)
        if not (hasattr(moe, "experts") and hasattr(moe, "gate")):
            raise TypeError("temporal P0 currently supports OLMoE-style MoE layers")
        moe._temporal_layer_id = layer_id
        moe._temporal_normalize_topk = force_normalize
        moe._temporal_mode = "full"
        moe._temporal_stats = defaultdict(CodecStats)
        moe.forward = MethodType(_patched_forward, moe)


def set_mode(model, mode: str) -> dict[int, CodecStats]:
    shared: dict[int, CodecStats] = defaultdict(CodecStats)
    for layer in model.model.layers:
        moe, _force_normalize = _layer_moe(layer)
        moe._temporal_mode = mode
        moe._temporal_stats = shared
    return shared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--dataset", default="wikitext2_docs")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _source_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model, dtype_name=args.dtype, local_files_only=args.offline
    )
    prompts = get_prompts(
        args.dataset, args.samples, offset=args.offset, seed=20260718
    )

    first_inputs = tokenizer(
        prompts[0], return_tensors="pt", truncation=True, max_length=args.seq_len
    )
    with torch.no_grad():
        stock_logits = model(**first_inputs).logits.detach().cpu()
    patch_model(model)
    set_mode(model, "full")
    with torch.no_grad():
        patched_logits = model(**first_inputs).logits.detach().cpu()
    exact = torch.equal(stock_logits, patched_logits)
    max_abs = float((stock_logits.float() - patched_logits.float()).abs().max().item())
    if not exact:
        raise RuntimeError(
            f"patched full path is not bitwise exact; max_abs={max_abs}"
        )

    accumulators = {mode: MetricAccumulator() for mode in MODES}
    sample_rows: list[dict[str, int | float | str]] = []
    layer_totals: dict[str, dict[int, CodecStats]] = {
        mode: defaultdict(CodecStats) for mode in MODES
    }

    for sample_id, text in enumerate(prompts):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=args.seq_len
        )
        baseline_logits: torch.Tensor | None = None
        baseline_bytes = 0
        for mode in MODES:
            stats = set_mode(model, mode)
            with torch.no_grad():
                logits = model(**inputs).logits
            if mode == "full":
                baseline_logits = logits.detach().clone()
            assert baseline_logits is not None
            metric = accumulators[mode].add(
                sample_id,
                logits,
                inputs["input_ids"],
                baseline_logits=None if mode == "full" else baseline_logits,
                attention_mask=inputs.get("attention_mask"),
            )
            total = CodecStats()
            for layer_id, layer_stats in stats.items():
                total.merge(layer_stats)
                layer_totals[mode][layer_id].merge(layer_stats)
            row = total.row(mode)
            if mode == "full":
                baseline_bytes = int(row["payload_bytes"])
            row.update(
                {
                    "sample_id": sample_id,
                    "token_count": metric.token_count,
                    "mean_nll": metric.mean_nll,
                    "mean_token_kl": metric.mean_token_kl,
                    "saving_vs_bf16": 1.0
                    - float(row["payload_bytes"]) / max(baseline_bytes, 1),
                }
            )
            sample_rows.append(row)
        print(f"sample {sample_id + 1}/{len(prompts)} complete", flush=True)

    pd.DataFrame(sample_rows).to_csv(output_dir / "sample_metrics.csv", index=False)
    layer_rows = []
    for mode, layers in layer_totals.items():
        for layer_id, stats in sorted(layers.items()):
            layer_rows.append(stats.row(mode, layer_id))
    pd.DataFrame(layer_rows).to_csv(output_dir / "layer_stats.csv", index=False)

    summary: dict[str, object] = {
        "model": args.model,
        "dataset": args.dataset,
        "samples": args.samples,
        "offset": args.offset,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "model_load_seconds": load_seconds,
        "exactness": {"bitwise_equal": exact, "max_abs": max_abs},
        "strategies": {},
        "source_sha256": _source_hash(Path(__file__)),
    }
    aggregate_rows: dict[str, dict[str, int | float | str]] = {}
    for mode in MODES:
        total = CodecStats()
        for stats in layer_totals[mode].values():
            total.merge(stats)
        row = total.row(mode)
        aggregate_rows[mode] = row
        quality = accumulators[mode].bootstrap_summary(
            n_bootstrap=args.bootstrap, seed=20260718
        )
        summary["strategies"][mode] = {**row, **quality}

    temporal = aggregate_rows["temporal_delta_mxfp4"]
    direct = aggregate_rows["revisit_abs_mxfp4"]
    fp8 = aggregate_rows["uniform_fp8"]
    temporal_quality = summary["strategies"]["temporal_delta_mxfp4"]
    direct_quality = summary["strategies"]["revisit_abs_mxfp4"]
    revisit_rate = float(aggregate_rows["full"]["revisit_rate"])
    byte_saving_vs_fp8 = 1.0 - float(temporal["payload_bytes"]) / max(
        float(fp8["payload_bytes"]), 1.0
    )
    same_budget_kl_ratio = float(temporal_quality["mean_token_kl"]) / max(
        float(direct_quality["mean_token_kl"]), 1e-30
    )
    pass_route = revisit_rate >= 0.30
    pass_bytes = byte_saving_vs_fp8 >= 0.15
    pass_mechanism = same_budget_kl_ratio <= 0.80
    verdict = "PASS" if pass_route and pass_bytes and pass_mechanism else "FAIL"
    summary["decision"] = {
        "verdict": verdict,
        "revisit_rate": revisit_rate,
        "byte_saving_vs_uniform_fp8": byte_saving_vs_fp8,
        "same_budget_kl_ratio_temporal_over_direct": same_budget_kl_ratio,
        "gates": {
            "revisit_rate_ge_0.30": pass_route,
            "byte_saving_vs_uniform_fp8_ge_0.15": pass_bytes,
            "temporal_kl_le_0.80x_direct_same_budget": pass_mechanism,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# Temporal-Residual EP Mac P0

> Evidence boundary: numerical fake quantization + logical payload only; no GPU kernel,
> collective, NIC bytes, TPOT, or P99 evidence.

- model: `{args.model}`
- samples / seq_len: `{args.samples}` / `{args.seq_len}`
- stock-vs-patched exactness: `{exact}` (max abs `{max_abs}`)
- adjacent-token expert revisit rate: `{revisit_rate:.4%}`
- temporal logical-byte saving vs uniform FP8: `{byte_saving_vs_fp8:.4%}`
- temporal/direct same-budget end-to-end KL ratio: `{same_budget_kl_ratio:.6f}`
- preregistered-style P0 verdict: **{verdict}**

The causal control is `revisit_abs_mxfp4`: it compresses exactly the same routed
pairs with the same formats and bytes, but quantizes the absolute output instead
of a closed-loop temporal residual.  A win over this control isolates predictive
coding from merely selecting recurrent experts.

Promotion gates:

1. revisit rate >= 30%;
2. logical-byte saving vs uniform FP8 >= 15%;
3. temporal end-to-end KL <= 0.8x same-budget direct-MXFP4 KL.

Passing this P0 only licenses a fresh held-out replication and a GPU codec
microbenchmark.  It does not establish communication speedup.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

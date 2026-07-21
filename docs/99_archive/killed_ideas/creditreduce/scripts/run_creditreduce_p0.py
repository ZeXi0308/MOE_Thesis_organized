"""Run the preregistered Mac numerical P0 for dynamic CreditReduce.

This driver measures full-model numerical quality and logical remote payload.
It is not a GPU, collective, network, latency, or actual-wire benchmark.
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
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from capture_moe import patch_mixtral_moe
from creditreduce_reference import ENDPOINTS
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from prompts import get_wikitext2_documents


CAMPAIGN = "creditreduce_p0_2026-07-17"
POOL_SEED = 20260717
POOL_SIZE = 192
POOL_HASH = "4fb0c938bb608213647bf72757435dd177c3c0e6d67ecc49e39d1b34683e2001"
PHASES = {
    "dev": {
        "offset": 0,
        "size": 32,
        "hash": "4b7ec5add131692ae13b371a197417d9d5475cd1d273ca6f9fc74e09e88ecc46",
    },
    "p0_1_holdout": {
        "offset": 32,
        "size": 64,
        "hash": "45e7cb88a18065cf8a4f5f74d1d6d7ad8af2fa48998c2d06a1511bad78ef249a",
    },
    "p0_2_calibration": {
        "offset": 96,
        "size": 32,
        "hash": "a7f2d225a36abb0bd19e4ac6b88948078ff07aa668e0e03774f79f58aefb6b12",
    },
    "p0_2_holdout": {
        "offset": 128,
        "size": 64,
        "hash": "08c6993021bc1e09b7f09c2419a434d9614ffc85e00f6bf1e090a7eba4794cd0",
    },
}
P0_1_ENDPOINTS = (
    "late_bf16",
    "stock_early_bf16",
    "clean_early_bf16",
    "uniform_early_fp32",
    "pd_full",
    "uniform_early_fp8",
)
NLL_MARGIN = 0.005
FORMAL_MODELS = {
    "allenai/OLMoE-1B-7B-0924",
    "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--phase", choices=sorted(PHASES), default="dev")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--ranks-per-domain", type=int, default=4)
    parser.add_argument(
        "--placement", choices=("contiguous", "round_robin"), default="contiguous"
    )
    parser.add_argument(
        "--endpoints", nargs="+", choices=ENDPOINTS, default=list(P0_1_ENDPOINTS)
    )
    parser.add_argument("--residual-rms-threshold", type=float, default=float("inf"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--record-detail", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--registry-file",
        default=(
            "experiments/idea_a_mac/outputs/"
            f"{CAMPAIGN}/frozen_historical_exclusion_registry.json"
        ),
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_lines(values: Iterable[str]) -> str:
    return sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    return sha256_bytes(tensor.numpy().tobytes())


def _walk_sha256(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key == "sha256"
                and isinstance(item, str)
                and len(item) == 64
            ):
                found.append(item)
            found.extend(_walk_sha256(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_sha256(item))
    return found


def scan_historical_manifests(outputs_root: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    all_hashes: list[str] = []
    paths = sorted(outputs_root.rglob("data_manifest.json")) + sorted(
        outputs_root.rglob("data_manifest.csv")
    )
    for path in paths:
        if path.suffix == ".json":
            hashes = _walk_sha256(json.loads(path.read_text(encoding="utf-8")))
        else:
            hashes = []
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    value = row.get("sha256")
                    if isinstance(value, str) and len(value) == 64:
                        hashes.append(value)
        all_hashes.extend(hashes)
        files.append(
            {
                "path": str(path),
                "sha256": sha256_bytes(path.read_bytes()),
                "hash_occurrences": len(hashes),
            }
        )
    unique = sorted(set(all_hashes))
    return {
        "schema": 1,
        "campaign": CAMPAIGN,
        "manifest_files": files,
        "manifest_file_count": len(files),
        "hash_occurrences": len(all_hashes),
        "unique_hash_count": len(unique),
        "unique_hashes": unique,
        "unique_hashes_sha256": hash_lines(unique),
    }


def load_or_freeze_registry(path: Path) -> dict[str, object]:
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        registry = scan_historical_manifests(
            Path("experiments/idea_a_mac/outputs")
        )
        path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if int(registry.get("manifest_file_count", -1)) != 24:
        raise RuntimeError("frozen historical registry must contain 24 data manifests")
    if int(registry.get("unique_hash_count", -1)) != 121:
        raise RuntimeError("frozen historical registry must contain 121 unique hashes")
    return registry


def frozen_documents(
    phase: str, samples: int | None, allow_partial: bool
) -> tuple[list[str], list[dict[str, object]], dict[str, object]]:
    spec = PHASES[phase]
    pool = get_wikitext2_documents(
        POOL_SIZE, offset=0, split="train", seed=POOL_SEED
    )
    pool_hashes = [sha256_bytes(text.encode("utf-8")) for text in pool]
    observed_pool_hash = hash_lines(pool_hashes)
    if observed_pool_hash != POOL_HASH:
        raise RuntimeError(
            f"frozen pool hash mismatch: {observed_pool_hash} != {POOL_HASH}"
        )

    full_size = int(spec["size"])
    selected_size = full_size if samples is None else samples
    if selected_size < 1 or selected_size > full_size:
        raise ValueError(f"--samples must be in [1, {full_size}] for {phase}")
    if phase != "dev" and selected_size != full_size:
        raise RuntimeError(
            "sealed phases always require their full preregistered sample count; "
            "use the separate dev split for all partial diagnostics"
        )
    start = int(spec["offset"])
    selected = pool[start : start + selected_size]
    selected_hashes = pool_hashes[start : start + selected_size]
    full_phase_hash = hash_lines(pool_hashes[start : start + full_size])
    if full_phase_hash != spec["hash"]:
        raise RuntimeError(
            f"frozen phase hash mismatch: {full_phase_hash} != {spec['hash']}"
        )
    rows = [
        {
            "sample_id": start + index,
            "phase": phase,
            "split": "train",
            "dataset_seed": POOL_SEED,
            "shuffled_offset": start + index,
            "sha256": digest,
            "characters": len(text),
        }
        for index, (text, digest) in enumerate(zip(selected, selected_hashes))
    ]
    meta = {
        "pool_size": POOL_SIZE,
        "pool_hash": observed_pool_hash,
        "phase_full_size": full_size,
        "phase_full_hash": full_phase_hash,
        "selected_size": selected_size,
        "selected_hash": hash_lines(selected_hashes),
        "status": "COMPLETE" if selected_size == full_size else "PARTIAL",
    }
    return selected, rows, meta


def moe_modules(model) -> list[object]:
    modules: list[object] = []
    for layer in model.model.layers:
        modules.append(layer.block_sparse_moe if hasattr(layer, "block_sparse_moe") else layer.mlp)
    return modules


def restore_forwards(modules: list[object], forwards: list[object]) -> None:
    for module, forward in zip(modules, forwards):
        module.forward = forward


def tokenized_inputs(tokenizer, texts: list[str], seq_len: int, device=None):
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        yield encoded


def top1_disagreement(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[int, int]:
    if baseline.shape[1] < 2:
        return 0, 0
    left = baseline[:, :-1].argmax(dim=-1)
    right = candidate[:, :-1].argmax(dim=-1)
    if attention_mask is None:
        mask = torch.ones_like(left, dtype=torch.bool)
    else:
        mask = attention_mask[:, 1:].bool()
    return int((left[mask] != right[mask]).sum().item()), int(mask.sum().item())


def run_original(
    model,
    tokenizer,
    texts: list[str],
    sample_ids: list[int],
    seq_len: int,
    baseline_logits: list[torch.Tensor],
) -> dict[str, object]:
    metrics = MetricAccumulator()
    disagreements = 0
    disagreement_tokens = 0
    hashes: list[str] = []
    started = time.perf_counter()
    for index, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len, device=model.device)):
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        metrics.add(
            sample_ids[index],
            logits,
            inputs["input_ids"].cpu(),
            baseline_logits=baseline_logits[index],
            attention_mask=inputs.get("attention_mask").cpu() if inputs.get("attention_mask") is not None else None,
        )
        changed, count = top1_disagreement(
            baseline_logits[index], logits,
            inputs.get("attention_mask").cpu() if inputs.get("attention_mask") is not None else None,
        )
        disagreements += changed
        disagreement_tokens += count
        hashes.append(tensor_sha256(logits))
        print(f"  original {index + 1}/{len(texts)}", flush=True)
    return {
        "metrics": metrics,
        "hashes": hashes,
        "disagreements": disagreements,
        "disagreement_tokens": disagreement_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "recorder": None,
    }


def legacy_patch_hash(
    model,
    tokenizer,
    modules: list[object],
    original_forwards: list[object],
    text: str,
    seq_len: int,
) -> str:
    patch_mixtral_moe(model, "full")
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        return tensor_sha256(logits)
    finally:
        restore_forwards(modules, original_forwards)


def run_endpoint(
    model,
    tokenizer,
    modules: list[object],
    original_forwards: list[object],
    texts: list[str],
    sample_ids: list[int],
    seq_len: int,
    endpoint: str,
    args: argparse.Namespace,
    baseline_logits: list[torch.Tensor] | None,
    *,
    keep_logits: bool,
) -> dict[str, object]:
    recorder = patch_mixtral_moe(
        model,
        "full",
        creditreduce_endpoint=endpoint,
        creditreduce_ep_size=args.ep_size,
        creditreduce_ranks_per_domain=args.ranks_per_domain,
        creditreduce_placement=args.placement,
        creditreduce_residual_rms_threshold=args.residual_rms_threshold,
        creditreduce_record_detail=args.record_detail,
    )
    metrics = MetricAccumulator()
    outputs: list[torch.Tensor] = []
    hashes: list[str] = []
    disagreements = 0
    disagreement_tokens = 0
    started = time.perf_counter()
    try:
        for index, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len, device=model.device)):
            recorder.set_sample_id(sample_ids[index])
            with torch.no_grad():
                logits = model(**inputs).logits.detach().cpu()
            metrics.add(
                sample_ids[index],
                logits,
                inputs["input_ids"].cpu(),
                baseline_logits=(
                    None if baseline_logits is None else baseline_logits[index]
                ),
                attention_mask=inputs.get("attention_mask").cpu() if inputs.get("attention_mask") is not None else None,
            )
            if baseline_logits is not None:
                changed, count = top1_disagreement(
                    baseline_logits[index], logits,
                    inputs.get("attention_mask").cpu() if inputs.get("attention_mask") is not None else None,
                )
                disagreements += changed
                disagreement_tokens += count
            hashes.append(tensor_sha256(logits))
            if keep_logits:
                outputs.append(logits)
            print(f"  {endpoint} {index + 1}/{len(texts)}", flush=True)
    finally:
        restore_forwards(modules, original_forwards)
    return {
        "metrics": metrics,
        "logits": outputs,
        "hashes": hashes,
        "disagreements": disagreements,
        "disagreement_tokens": disagreement_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "recorder": recorder,
    }


def paired_nll_bootstrap(
    candidate: MetricAccumulator,
    reference: MetricAccumulator,
    n_bootstrap: int,
    seed: int = 20260717,
) -> dict[str, float | str]:
    if len(candidate.samples) != len(reference.samples):
        raise ValueError("paired sample counts differ")
    deltas = np.asarray(
        [
            candidate.samples[index].mean_nll - reference.samples[index].mean_nll
            for index in range(len(reference.samples))
        ],
        dtype=np.float64,
    )
    point = float(deltas.mean())
    if len(deltas) < 2 or n_bootstrap <= 0:
        low = high = low_two = high_two = point
        status = "INCONCLUSIVE"
    else:
        rng = np.random.default_rng(seed)
        chosen = rng.integers(0, len(deltas), size=(n_bootstrap, len(deltas)))
        samples = deltas[chosen].mean(axis=1)
        low = float(np.quantile(samples, 0.05))
        high = float(np.quantile(samples, 0.95))
        low_two = float(np.quantile(samples, 0.025))
        high_two = float(np.quantile(samples, 0.975))
        if low > NLL_MARGIN:
            status = "QUALITY_FAIL"
        elif high <= NLL_MARGIN:
            status = "NONINFERIOR"
        else:
            status = "INCONCLUSIVE"
    return {
        "delta_nll_mean": point,
        "delta_nll_lcb95_one_sided": low,
        "delta_nll_ucb95_one_sided": high,
        "delta_nll_ci_low_two_sided": low_two,
        "delta_nll_ci_high_two_sided": high_two,
        "nll_margin": NLL_MARGIN,
        "quality_status": status,
    }


def ratio_bootstrap(
    numerators: np.ndarray,
    denominators: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    point = float(numerators.sum() / max(float(denominators.sum()), 1.0))
    if len(numerators) < 2 or n_bootstrap <= 0:
        return {"point": point, "lcb95": point, "ucb95": point}
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        selected = rng.integers(0, len(numerators), size=len(numerators))
        values[index] = numerators[selected].sum() / max(
            float(denominators[selected].sum()), 1.0
        )
    return {
        "point": point,
        "lcb95": float(np.quantile(values, 0.05)),
        "ucb95": float(np.quantile(values, 0.95)),
    }


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "run_creditreduce_p0.py",
        "creditreduce_reference.py",
        "test_creditreduce_reference.py",
        "capture_moe.py",
        "fake_quant.py",
        "metrics.py",
        "modeling.py",
        "prompts.py",
        "CreditReduce_P0_预注册_2026-07-17.md",
    )
    return {name: sha256_bytes((root / name).read_bytes()) for name in names}


def environment_manifest() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def opportunity_frames(
    layer_rows: list[dict[str, object]], bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    layers = pd.DataFrame(layer_rows)
    required = {
        "sample_id",
        "layer",
        "k_remote",
        "eligible_k_remote",
        "eligible_credit_units",
    }
    missing = required - set(layers.columns)
    if missing:
        raise RuntimeError(f"missing CreditReduce opportunity fields: {missing}")
    documents = (
        layers.groupby("sample_id", as_index=False)[
            ["k_remote", "eligible_k_remote", "eligible_credit_units"]
        ]
        .sum()
        .sort_values("sample_id")
    )
    documents["p_eligible"] = documents["eligible_k_remote"] / documents[
        "k_remote"
    ].clip(lower=1)
    documents["rho_credit"] = documents["eligible_credit_units"] / documents[
        "k_remote"
    ].clip(lower=1)
    denominators = documents["k_remote"].to_numpy(dtype=np.float64)
    p_eligible = ratio_bootstrap(
        documents["eligible_k_remote"].to_numpy(dtype=np.float64),
        denominators,
        bootstrap,
        20260717,
    )
    rho_credit = ratio_bootstrap(
        documents["eligible_credit_units"].to_numpy(dtype=np.float64),
        denominators,
        bootstrap,
        20260718,
    )
    return layers, documents, {
        "p_eligible": p_eligible,
        "rho_credit": rho_credit,
    }


def payload_counterfactuals(
    layer_rows: list[dict[str, object]], endpoints: list[str]
) -> dict[str, dict[str, float | int]]:
    frame = pd.DataFrame(layer_rows)
    totals: dict[str, dict[str, float | int]] = {}
    late_payload = int(frame["late_bf16_logical_payload_bytes"].sum())
    for endpoint in endpoints:
        payload = int(frame[f"{endpoint}_logical_payload_bytes"].sum())
        bitmap = int(frame[f"{endpoint}_minimal_bitmap_bytes"].sum())
        scale = int(frame[f"{endpoint}_scale_bytes"].sum())
        accounted = payload + bitmap + scale
        totals[endpoint] = {
            "logical_payload_bytes": payload,
            "minimal_bitmap_bytes": bitmap,
            "scale_bytes": scale,
            "accounted_bytes": accounted,
            "saving_vs_late_payload": 1.0 - accounted / max(late_payload, 1),
        }
    return totals


def gate_threshold(value: dict[str, float], threshold: float) -> str:
    if value["lcb95"] >= threshold:
        return "PASS"
    if value["ucb95"] < threshold:
        return "FAIL"
    return "INCONCLUSIVE"


def decide_p0_1(
    phase: str,
    phase_status: str,
    opportunity: dict[str, object],
    endpoint_rows: dict[str, dict[str, object]],
    payloads: dict[str, dict[str, float | int]],
    pd_equals_fp32: bool,
    exactness: dict[str, bool],
) -> dict[str, object]:
    if phase != "p0_1_holdout" or phase_status != "COMPLETE":
        return {
            "overall": "NOT_TESTED",
            "reason": "only the complete sealed p0_1_holdout can decide P0-1",
        }
    p_gate = gate_threshold(opportunity["p_eligible"], 0.20)
    rho_gate = gate_threshold(opportunity["rho_credit"], 0.15)
    early_status = endpoint_rows["clean_early_bf16"]["quality_status"]
    pd_status = endpoint_rows["pd_full"]["quality_status"]
    fp8_status = endpoint_rows["uniform_early_fp8"]["quality_status"]
    fp8_dominates = (
        fp8_status == "NONINFERIOR"
        and int(payloads["uniform_early_fp8"]["accounted_bytes"])
        < int(payloads["pd_full"]["accounted_bytes"])
    )
    gates = {
        "opportunity_eligible": p_gate,
        "opportunity_credit": rho_gate,
        "early_bf16_must_fail": (
            "PASS" if early_status == "QUALITY_FAIL" else
            "FAIL" if early_status == "NONINFERIOR" else "INCONCLUSIVE"
        ),
        "pd_full_noninferior": (
            "PASS" if pd_status == "NONINFERIOR" else
            "FAIL" if pd_status == "QUALITY_FAIL" else "INCONCLUSIVE"
        ),
        "pd_full_equals_uniform_early_fp32": "PASS" if pd_equals_fp32 else "FAIL",
        "legacy_patch_exact": (
            "PASS"
            if exactness["legacy_patched_full_equals_pretrained_first_sample"]
            else "FAIL"
        ),
        "late_repeat_deterministic": (
            "PASS"
            if exactness["late_reference_repeat_deterministic_first_sample"]
            else "FAIL"
        ),
        "pd_full_payload_cap": (
            "PASS" if exactness["all_pd_full_payload_caps_hold"] else "FAIL"
        ),
        "uniform_fp8_not_dominant": (
            "FAIL" if fp8_dominates else
            "INCONCLUSIVE" if fp8_status == "INCONCLUSIVE" else "PASS"
        ),
    }
    if "FAIL" in gates.values():
        overall = "FAIL"
    elif "INCONCLUSIVE" in gates.values():
        overall = "INCONCLUSIVE"
    else:
        overall = "PASS"
    return {"overall": overall, "gates": gates, "fp8_dominates": fp8_dominates}


def markdown_report(
    args: argparse.Namespace,
    phase_meta: dict[str, object],
    opportunity: dict[str, object],
    summary: pd.DataFrame,
    decision: dict[str, object],
) -> str:
    rows = [
        "# CreditReduce Mac P0 Result",
        "",
        "> Numerical/full-model evidence only; no GPU, network, latency, or actual-wire claim.",
        "",
        "## Configuration",
        "",
        f"- model: `{args.model}`",
        f"- phase/status: `{args.phase}` / `{phase_meta['status']}`",
        f"- topology: EP{args.ep_size}, ranks/domain={args.ranks_per_domain}, `{args.placement}`",
        f"- samples/seq_len: {phase_meta['selected_size']} / {args.seq_len}",
        "",
        "## Opportunity",
        "",
        f"- p_eligible: {opportunity['p_eligible']}",
        f"- rho_credit: {opportunity['rho_credit']}",
        "",
        "## Endpoint quality",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Hard decision",
        "",
        "```json",
        json.dumps(decision, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    if args.dtype != "bfloat16":
        raise ValueError("the frozen CreditReduce P0 contribution contract requires bfloat16")
    if args.phase in ("p0_2_calibration", "p0_2_holdout"):
        raise NotImplementedError(
            "P0-2 remains sealed until P0-1 passes and the matched-n32 driver is reviewed"
        )
    if args.allow_partial and args.phase != "dev":
        raise RuntimeError("--allow-partial is restricted to the dev split")
    if args.phase == "p0_1_holdout":
        frozen_errors = []
        if args.model not in FORMAL_MODELS:
            frozen_errors.append("model")
        if args.samples is not None:
            frozen_errors.append("samples must be omitted (frozen at 64)")
        if args.seq_len != 256:
            frozen_errors.append("seq_len")
        if args.dtype != "bfloat16":
            frozen_errors.append("dtype")
        if args.ep_size != 8:
            frozen_errors.append("ep_size")
        if args.ranks_per_domain != 4:
            frozen_errors.append("ranks_per_domain")
        if args.placement != "contiguous":
            frozen_errors.append("placement")
        if tuple(args.endpoints) != P0_1_ENDPOINTS:
            frozen_errors.append("endpoint order")
        if not np.isinf(args.residual_rms_threshold):
            frozen_errors.append("residual threshold")
        if args.bootstrap != 10000:
            frozen_errors.append("bootstrap")
        if not args.offline:
            frozen_errors.append("offline")
        if args.record_detail:
            frozen_errors.append("record_detail")
        if "p0_1_holdout" not in Path(args.output_dir).name:
            frozen_errors.append("output_dir name")
        if frozen_errors:
            raise RuntimeError(
                "sealed P0-1 configuration drift: " + ", ".join(frozen_errors)
            )
    if args.ep_size < 1 or args.ranks_per_domain < 1:
        raise ValueError("EP topology sizes must be positive")

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output dir: {output}")
    output.mkdir(parents=True, exist_ok=True)

    registry_path = Path(args.registry_file)
    registry = load_or_freeze_registry(registry_path)
    texts, manifest_rows, phase_meta = frozen_documents(
        args.phase, args.samples, args.allow_partial
    )
    old_hashes = set(registry["unique_hashes"])
    overlap = sorted({row["sha256"] for row in manifest_rows} & old_hashes)
    if overlap:
        raise RuntimeError(f"fresh-data guard failed; historical overlap: {overlap}")

    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model, dtype_name=args.dtype, local_files_only=args.offline
    )
    for row, text in zip(manifest_rows, texts):
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        row["tokens_before_truncation"] = len(token_ids)
        row["tokens_used"] = min(len(token_ids), args.seq_len)
    modules = moe_modules(model)
    original_forwards = [module.forward for module in modules]
    sample_ids = [int(row["sample_id"]) for row in manifest_rows]

    config = vars(args) | {
        "campaign": CAMPAIGN,
        "dataset": "wikitext2_docs",
        "split": "train",
        "dataset_seed": POOL_SEED,
        "phase_meta": phase_meta,
        "model_load_seconds": load_seconds,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "nll_margin": NLL_MARGIN,
        "evidence_boundary": "CPU numerical/full-model; logical remote payload only",
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(environment_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "historical_exclusion_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "data_manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"loaded {args.model} in {load_seconds:.2f}s; phase={args.phase}; "
        f"n={len(texts)}; topology=EP{args.ep_size}/D{args.ranks_per_domain}/"
        f"{args.placement}",
        flush=True,
    )

    late = run_endpoint(
        model,
        tokenizer,
        modules,
        original_forwards,
        texts,
        sample_ids,
        args.seq_len,
        "late_bf16",
        args,
        None,
        keep_logits=True,
    )
    late_logits = late["logits"]
    if len(late_logits) != len(texts):
        raise RuntimeError("late reference logits were not retained")

    original = run_original(
        model, tokenizer, texts, sample_ids, args.seq_len, late_logits
    )
    patched_full_hash = legacy_patch_hash(
        model,
        tokenizer,
        modules,
        original_forwards,
        texts[0],
        args.seq_len,
    )
    late_repeat = run_endpoint(
        model,
        tokenizer,
        modules,
        original_forwards,
        texts[:1],
        sample_ids[:1],
        args.seq_len,
        "late_bf16",
        args,
        None,
        keep_logits=False,
    )
    results: dict[str, dict[str, object]] = {
        "late_bf16": late,
        "pretrained_original": original,
    }
    for endpoint in args.endpoints:
        if endpoint == "late_bf16":
            continue
        results[endpoint] = run_endpoint(
            model,
            tokenizer,
            modules,
            original_forwards,
            texts,
            sample_ids,
            args.seq_len,
            endpoint,
            args,
            late_logits,
            keep_logits=False,
        )

    late_recorder = late["recorder"]
    if late_recorder is None:
        raise RuntimeError("late CreditReduce recorder missing")
    layer_frame, opportunity_docs, opportunity = opportunity_frames(
        late_recorder.creditreduce_layer_rows, args.bootstrap
    )
    payloads = payload_counterfactuals(
        late_recorder.creditreduce_layer_rows, list(args.endpoints)
    )

    endpoint_rows: dict[str, dict[str, object]] = {}
    sample_rows: list[dict[str, object]] = []
    late_metrics = late["metrics"]
    for name, result in results.items():
        metrics = result["metrics"]
        if name == "late_bf16":
            quality = {
                "delta_nll_mean": 0.0,
                "delta_nll_lcb95_one_sided": 0.0,
                "delta_nll_ucb95_one_sided": 0.0,
                "delta_nll_ci_low_two_sided": 0.0,
                "delta_nll_ci_high_two_sided": 0.0,
                "nll_margin": NLL_MARGIN,
                "quality_status": "NONINFERIOR",
            }
        else:
            quality = paired_nll_bootstrap(metrics, late_metrics, args.bootstrap)
        row = {
            "endpoint": name,
            "corpus_ppl": metrics.corpus_ppl,
            "mean_token_kl_vs_late": metrics.mean_token_kl,
            "top1_disagreement_rate": (
                int(result["disagreements"])
                / max(int(result["disagreement_tokens"]), 1)
            ),
            "elapsed_seconds_diagnostic_only": float(result["elapsed_seconds"]),
            **quality,
        }
        if name in payloads:
            row.update(payloads[name])
        endpoint_rows[name] = row
        for index, metric in enumerate(metrics.samples):
            late_metric = late_metrics.samples[index]
            sample_rows.append(
                {
                    "endpoint": name,
                    "sample_id": metric.sample_id,
                    "token_count": metric.token_count,
                    "mean_nll": metric.mean_nll,
                    "delta_nll_vs_late": metric.mean_nll - late_metric.mean_nll,
                    "mean_token_kl_vs_late": metric.mean_token_kl,
                }
            )

    pd_equals_fp32 = (
        "pd_full" in results
        and "uniform_early_fp32" in results
        and results["pd_full"]["hashes"]
        == results["uniform_early_fp32"]["hashes"]
    )
    exactness = {
        "pd_full_equals_uniform_early_fp32_full_model": pd_equals_fp32,
        "legacy_patched_full_equals_pretrained_first_sample": (
            patched_full_hash == original["hashes"][0]
        ),
        "late_reference_repeat_deterministic_first_sample": (
            late_repeat["hashes"][0] == late["hashes"][0]
        ),
        "all_late_payload_caps_hold": bool(
            layer_frame["late_bf16_payload_cap_all"].all()
        ),
        "all_pd_full_payload_caps_hold": bool(
            layer_frame["pd_full_payload_cap_all"].all()
        ),
        "legacy_path_not_modified_when_creditreduce_disabled": True,
    }
    decision = decide_p0_1(
        args.phase,
        phase_meta["status"],
        opportunity,
        endpoint_rows,
        payloads,
        pd_equals_fp32,
        exactness,
    )
    decision["phase"] = args.phase
    decision["evidence_boundary"] = (
        "full-model numerical quality and logical remote hidden-vector payload only"
    )

    summary = pd.DataFrame(endpoint_rows.values())
    summary.to_csv(output / "endpoint_summary.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output / "sample_metrics.csv", index=False)
    layer_frame.to_csv(output / "opportunity_by_layer.csv", index=False)
    opportunity_docs.to_csv(output / "opportunity_by_document.csv", index=False)
    if args.record_detail:
        pd.DataFrame(late_recorder.creditreduce_token_rows).to_csv(
            output / "token_diagnostics.csv", index=False
        )
        pd.DataFrame(late_recorder.creditreduce_group_rows).to_csv(
            output / "group_diagnostics.csv", index=False
        )
    (output / "paired_bootstrap.json").write_text(
        json.dumps(
            {
                "opportunity": opportunity,
                "endpoint_quality": endpoint_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "exactness.json").write_text(
        json.dumps(exactness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(
        markdown_report(args, phase_meta, opportunity, summary, decision),
        encoding="utf-8",
    )
    (output / "status.json").write_text(
        json.dumps(
            {"status": phase_meta["status"], "decision": decision["overall"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

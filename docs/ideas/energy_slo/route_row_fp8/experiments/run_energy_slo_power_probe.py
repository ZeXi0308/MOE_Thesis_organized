#!/usr/bin/env python3
"""Energy-SLO Precision EP P0: real GPU power measurement (pynvml), not a
simulated J/token estimate.

Scope honesty: the project's existing precision-degradation mechanisms
(fake_quant.py) are downstream-quality proxies for *communication* bytes --
they quantize-dequantize in the same bf16 compute path, so they do not change
the actual GEMM FLOPs or dtype used on this single GPU, and therefore cannot
show a real energy difference (no real network/EP mesh exists on one GPU
either, so the "placement/replica" axis of the original idea cannot be
measured here at all). Rather than fabricating a precision-energy result the
current infrastructure cannot support, this script measures the two axes that
ARE real and measurable on a single rented GPU:

  (1) batch-size vs energy-per-token (a genuine throughput/power tradeoff
      curve using the real OLMoE/LLM-jp model, real nvidia-smi power draw).
  (2) a real (not proxy) low-precision compute path: casting the expert
      FFN's gate/up/down weights to torch.float8_e4m3fn and using
      `torch._scaled_mm` real FP8 tensor-core GEMM (supported on this
      Blackwell RTX 5090) vs bf16, to get a genuine FP8-vs-bf16 energy/token
      number -- the one part of "precision x energy" that can be measured for
      real without inventing a multi-GPU EP mesh.

Power is sampled via pynvml at ~50ms intervals during a sustained (>=3s)
inference loop; energy = integral(power dt); J/token = energy / tokens
processed. Evidence tag: [Observed], real hardware measurement.
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
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM


class PowerSampler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        import pynvml
        self.pynvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_s = interval_s
        self.samples: list[float] = []
        self.timestamps: list[float] = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.samples.append(mw / 1000.0)  # W
            self.timestamps.append(time.time())
            time.sleep(self.interval_s)

    def start(self):
        self.samples = []
        self.timestamps = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float, float]:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if len(self.samples) < 2:
            return 0.0, 0.0, 0.0
        power = np.array(self.samples)
        ts = np.array(self.timestamps)
        duration = float(ts[-1] - ts[0])
        energy_j = float(np.trapz(power, ts))
        return energy_j, duration, float(power.mean())


def run_forward_loop(model, batch_tokens: int, hidden_size: int, seq_len: int, device: str,
                      duration_s: float) -> int:
    top_k = int(model.config.num_experts_per_tok)
    num_experts = getattr(model.config, "num_experts", None)
    if num_experts is None:
        num_experts = model.config.num_local_experts
    input_ids = torch.randint(0, model.config.vocab_size, (batch_tokens, seq_len), device=device)
    total_tokens = 0
    start = time.time()
    with torch.no_grad():
        while time.time() - start < duration_s:
            model(input_ids)
            total_tokens += batch_tokens * seq_len
    torch.cuda.synchronize()
    return total_tokens


def measure_batch_energy_curve(model, hidden_size: int, seq_len: int, device: str,
                                batch_sizes: list[int], duration_s: float, sampler: PowerSampler) -> pd.DataFrame:
    rows = []
    for bs in batch_sizes:
        # warmup
        run_forward_loop(model, bs, hidden_size, seq_len, device, duration_s=1.0)
        torch.cuda.synchronize()
        sampler.start()
        t0 = time.time()
        tokens = run_forward_loop(model, bs, hidden_size, seq_len, device, duration_s=duration_s)
        wall = time.time() - t0
        energy_j, sampled_duration, mean_power_w = sampler.stop()
        rows.append({
            "batch_size": bs,
            "seq_len": seq_len,
            "tokens_processed": tokens,
            "wall_time_s": wall,
            "mean_power_w": mean_power_w,
            "energy_j": energy_j,
            "throughput_tokens_per_s": tokens / max(wall, 1e-9),
            "energy_per_token_mj": 1000.0 * energy_j / max(tokens, 1),
        })
        print(f"  batch={bs}: throughput={rows[-1]['throughput_tokens_per_s']:.1f} tok/s, "
              f"mean_power={mean_power_w:.1f}W, energy/token={rows[-1]['energy_per_token_mj']:.4f}mJ")
    return pd.DataFrame(rows)


def measure_fp8_vs_bf16(model, hidden_size: int, device: str, duration_s: float, sampler: PowerSampler,
                         batch_tokens: int = 4096) -> pd.DataFrame:
    """Real FP8 tensor-core GEMM (torch._scaled_mm) vs real bf16 GEMM, at a
    matched matrix size equal to one expert's gate_proj matmul. Isolates the
    real compute-energy effect of precision, decoupled from the routing loop
    (which is identical either way)."""
    layer0 = model.model.layers[0]
    moe = layer0.mlp if hasattr(layer0, "mlp") and hasattr(layer0.mlp, "experts") else layer0.block_sparse_moe
    w = moe.experts[0].gate_proj.weight.detach()  # [intermediate, hidden]
    out_dim, in_dim = w.shape
    x_bf16 = torch.randn(batch_tokens, in_dim, dtype=torch.bfloat16, device=device)
    w_bf16 = w.to(torch.bfloat16).t()  # [in, out], column-major view (no .contiguous())

    has_scaled_mm = hasattr(torch, "_scaled_mm")
    rows = []

    def bf16_matmul():
        return x_bf16 @ w_bf16

    def run_loop(fn, dur):
        n = 0
        t0 = time.time()
        with torch.no_grad():
            while time.time() - t0 < dur:
                fn()
                n += 1
        torch.cuda.synchronize()
        return n

    run_loop(bf16_matmul, 1.0)
    sampler.start()
    t0 = time.time()
    n_bf16 = run_loop(bf16_matmul, duration_s)
    wall_bf16 = time.time() - t0
    energy_bf16, _, power_bf16 = sampler.stop()
    rows.append({"precision": "bf16", "matmuls": n_bf16, "wall_time_s": wall_bf16,
                 "mean_power_w": power_bf16, "energy_j": energy_bf16,
                 "matmuls_per_s": n_bf16 / max(wall_bf16, 1e-9),
                 "energy_per_matmul_mj": 1000.0 * energy_bf16 / max(n_bf16, 1)})

    if has_scaled_mm:
        try:
            x_scale = torch.tensor(1.0, device=device)
            w_scale = torch.tensor(1.0, device=device)
            x_fp8 = x_bf16.to(torch.float8_e4m3fn).contiguous()  # [batch, in], row-major
            w_fp8 = w.to(torch.float8_e4m3fn).t()  # [in, out], column-major view of row-major [out, in]

            def fp8_matmul():
                return torch._scaled_mm(x_fp8, w_fp8, scale_a=x_scale, scale_b=w_scale, out_dtype=torch.bfloat16)

            run_loop(fp8_matmul, 1.0)
            sampler.start()
            t0 = time.time()
            n_fp8 = run_loop(fp8_matmul, duration_s)
            wall_fp8 = time.time() - t0
            energy_fp8, _, power_fp8 = sampler.stop()
            rows.append({"precision": "fp8_e4m3_real_tensor_core", "matmuls": n_fp8, "wall_time_s": wall_fp8,
                         "mean_power_w": power_fp8, "energy_j": energy_fp8,
                         "matmuls_per_s": n_fp8 / max(wall_fp8, 1e-9),
                         "energy_per_matmul_mj": 1000.0 * energy_fp8 / max(n_fp8, 1)})
        except Exception as exc:  # pragma: no cover - hardware/kernel dependent
            rows.append({"precision": "fp8_e4m3_real_tensor_core", "matmuls": 0, "wall_time_s": 0.0,
                         "mean_power_w": 0.0, "energy_j": 0.0, "matmuls_per_s": 0.0,
                         "energy_per_matmul_mj": 0.0, "error": str(exc)})
    else:
        rows.append({"precision": "fp8_e4m3_real_tensor_core", "matmuls": 0, "error": "torch._scaled_mm unavailable"})

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--model-key", default="olmoe")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--duration-s", type=float, default=4.0)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{args.model_key}] loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).to(device)
    model.eval()
    hidden_size = int(model.config.hidden_size)

    sampler = PowerSampler()

    print(f"[{args.model_key}] measuring batch-size vs energy-per-token curve...")
    batch_curve = measure_batch_energy_curve(model, hidden_size, args.seq_len, device,
                                              args.batch_sizes, args.duration_s, sampler)
    batch_curve.insert(0, "model", args.model_key)
    batch_curve.to_csv(out / f"{args.model_key}_batch_energy_curve.csv", index=False)

    print(f"[{args.model_key}] measuring real FP8 tensor-core vs bf16 GEMM energy...")
    fp8_curve = measure_fp8_vs_bf16(model, hidden_size, device, args.duration_s, sampler)
    fp8_curve.insert(0, "model", args.model_key)
    fp8_curve.to_csv(out / f"{args.model_key}_fp8_vs_bf16.csv", index=False)

    lines = [f"# Energy-SLO P0: Real GPU Power Measurement ({args.model_key})", "",
             "## Batch size vs energy-per-token (real nvidia-smi power draw)", ""]
    cols1 = ["batch_size", "seq_len", "throughput_tokens_per_s", "mean_power_w", "energy_per_token_mj"]
    lines.append("| " + " | ".join(cols1) + " |")
    lines.append("|" + "|".join(["---"] * len(cols1)) + "|")
    for _, row in batch_curve.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols1]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Real FP8 tensor-core GEMM vs real bf16 GEMM (matched expert gate_proj matmul size)")
    lines.append("")
    cols2 = ["precision", "matmuls_per_s", "mean_power_w", "energy_per_matmul_mj"]
    lines.append("| " + " | ".join(cols2) + " |")
    lines.append("|" + "|".join(["---"] * len(cols2)) + "|")
    for _, row in fp8_curve.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row.get(c), float) else str(row.get(c, "")) for c in cols2]
        lines.append("| " + " | ".join(vals) + " |")
    (out / f"{args.model_key}_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()

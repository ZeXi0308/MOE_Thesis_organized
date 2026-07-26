#!/usr/bin/env python3
"""Decompose BF16 linear, dynamic FP8, FP8 quantization, and prequantized GEMM."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import statistics

import torch

from route_row_policy import DualResidentFP8Linear, RuntimeCounters, _fp8_quantize_per_tensor, require_cuda_fp8

EXPERT_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\..*experts\.(\d+)$")


def ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--projection", choices=("gate", "down"), required=True)
    p.add_argument("--rows", type=ints, default=(1,2,4,8,16,32,64,128,256,512,1024,2048,4096))
    p.add_argument("--blocks", type=int, default=30)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def expert(model: object, layer: int, expert_id: int) -> object:
    found=[]
    for name,module in model.named_modules():
        m=EXPERT_PATH.search(name)
        if m and (int(m.group(1)),int(m.group(2)))==(layer,expert_id):
            if hasattr(module,"gate_proj") or hasattr(module,"w1"):
                found.append(module)
    if len(found)!=1:
        raise RuntimeError(f"expected one expert, found {len(found)}")
    return found[0]


def timed(operation: object) -> float:
    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    start.record(); output=operation(); end.record(); end.synchronize()
    if output is None:
        raise RuntimeError("operation returned None")
    return float(start.elapsed_time(end))*1000.0


def main() -> None:
    a=args()
    if a.output_dir.exists():
        raise RuntimeError("refusing to overwrite output")
    require_cuda_fp8("cuda:0",probe_kernel=True)
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(str(a.model_path),torch_dtype=torch.bfloat16,device_map="cuda:0",local_files_only=True).eval()
    e=expert(model,a.layer,a.expert)
    if a.projection=="gate":
        original=getattr(e,"gate_proj",getattr(e,"w1",None))
    else:
        original=getattr(e,"down_proj",getattr(e,"w2",None))
    if not isinstance(original,torch.nn.Linear):
        raise RuntimeError("projection is not Linear")
    counters=RuntimeCounters(); dual=DualResidentFP8Linear(original,counters)
    gen=torch.Generator(device="cuda"); gen.manual_seed(a.seed)
    inputs={r:torch.randn(r,original.in_features,generator=gen,device="cuda",dtype=torch.bfloat16) for r in a.rows}
    prequant={r:_fp8_quantize_per_tensor(inputs[r]) for r in a.rows}

    def operations(r: int) -> dict[str,object]:
        x=inputs[r]; q,scale=prequant[r]
        return {
            "bf16_linear":lambda: original(x),
            "quant_only":lambda: _fp8_quantize_per_tensor(x),
            "prequant_fp8_mm":lambda: torch._scaled_mm(q,dual.weight_fp8_t,scale_a=scale,scale_b=dual.weight_scale,out_dtype=torch.bfloat16,use_fast_accum=True),
            "dynamic_fp8_linear":lambda: dual.forward_fp8(x),
        }
    with torch.inference_mode():
        for r in a.rows:
            for op in operations(r).values():
                for _ in range(a.warmups): op()
        torch.cuda.synchronize()
        raw=[]; rng=random.Random(a.seed+1)
        for block in range(a.blocks):
            row_order=list(a.rows); rng.shuffle(row_order)
            for r in row_order:
                names=list(operations(r)); rng.shuffle(names); ops=operations(r)
                values={name:timed(ops[name]) for name in names}
                raw.append({"block":block,"rows":r,**{f"{name}_us":values[name] for name in sorted(values)}})
    cells=[]
    for r in a.rows:
        sample=[row for row in raw if row["rows"]==r]
        means={name:statistics.mean(float(row[f"{name}_us"]) for row in sample) for name in operations(r)}
        cells.append({"rows":r,**means,
            "prequant_speedup_vs_bf16":(means["bf16_linear"]-means["prequant_fp8_mm"])/means["bf16_linear"],
            "dynamic_speedup_vs_bf16":(means["bf16_linear"]-means["dynamic_fp8_linear"])/means["bf16_linear"],
            "quant_fraction_dynamic":means["quant_only"]/means["dynamic_fp8_linear"]})
    a.output_dir.mkdir(parents=True)
    with (a.output_dir/"timings.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0])); w.writeheader(); w.writerows(raw)
    summary={"model_path":str(a.model_path),"layer":a.layer,"expert":a.expert,"projection":a.projection,"gpu":torch.cuda.get_device_name(0),"cells":cells,"counters":counters.snapshot(),"boundary":"REAL_LINEAR_NATIVE_FP8_CONTROLLED_ACTIVATIONS_NOT_FULL_EXPERT"}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()

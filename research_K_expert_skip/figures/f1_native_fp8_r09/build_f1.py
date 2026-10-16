#!/usr/bin/env python3
"""Reproduce r09 F1 locally from completed native traces; no GPU imports.

Run with a Python environment containing Matplotlib. This wrapper reuses the
existing analyzer/plotter and changes only this figure's right-hand x-axis.
"""
from argparse import Namespace
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/olmoe_fp8_matrix_r09/native_traces"
sys.path.insert(0, str(ROOT / "code"))
import analyze_routes
import plot_occupancy
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

BATCHES = [1, 2, 4, 8, 16, 32, 64]
MODEL = "OLMoE FP8 W8A8"


def read(path):
    return json.loads(path.read_text())


def source_record(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    cfg = read(SOURCE / "config.json")
    install = read(SOURCE / "installation.json")
    prompts = read(SOURCE / "prompts.json")
    assert read(SOURCE / "status.json")["status"] == "COMPLETE"
    assert cfg["args"]["quantization"] == "fp8"
    assert cfg["args"]["prompt_tokens"] == 128
    assert cfg["args"]["trace_steps"] == 16
    assert len(install["layers"]) == 16
    assert {x["weight_bytes_per_expert"] for x in install["layers"].values()} == {6291464}
    records = prompts["records"]
    summary, sources = [], [source_record(SOURCE / n) for n in
                            ["config.json", "status.json", "installation.json", "prompts.json"]]
    boundary = ("Native OLMoE FP8 W8A8; one 16-step capture per B, 16 layers, "
                "128-token prefixes; fixed output length. Occupancy diagnostic only; "
                "fitted inclusion probabilities are in sample, not held-out validation.")
    for b in BATCHES:
        trace_path = SOURCE / f"trace_native_b{b}/snapshot.json"
        output_path = trace_path.with_name("outputs.json")
        snap, outputs = read(trace_path), read(output_path)
        assert snap["phase_steps"] == {"prefill": 1, "decode": 16}
        trace = snap["trace"]
        assert len(trace) == 256
        assert len({r["layer"] for r in trace}) == 16
        assert len({r["step"] for r in trace}) == 16
        assert set(Counter(r["layer"] for r in trace).values()) == {16}
        assert set(Counter(r["step"] for r in trace).values()) == {16}
        assert all(r["batch_size"] == b and r["phase"] == "decode" and
                   r["forward_completed"] and not r["policy_applied"] and
                   r["topk_ids"] == r["post_topk_ids"] for r in trace)
        assert outputs["batch_size"] == b and len(outputs["requests"]) == b
        assert outputs["decode_output_tokens"] == 16 * b
        assert all(r["output_tokens"] == 17 and r["finished"] for r in outputs["requests"])
        assert len({r["prompt_token_sha256"] for r in records[:b]}) == b
        analyze_routes.analyze(Namespace(
            trace=trace_path, input_format="vllm-json", run_id=f"r09_fp8_native_b{b}",
            model=MODEL, experts=64, top_k=8, mode="actual-decode",
            batch_sizes=[b], evidence_boundary=boundary, output_dir=OUT / f"b{b}"))
        with (OUT / f"b{b}/summary.csv").open() as f:
            rows = list(csv.DictReader(f))
        slots = sum(int(r["slots"]) for r in rows)
        singles = sum(int(r["singleton_slots"]) for r in rows)
        assert slots == 16 * 16 * b * 8
        summary.append(dict(batch=b, captures=1, decode_steps=16, routed_layers=16,
            layer_step_observations=256, all_output_tokens_per_request=17,
            decode_output_tokens=16*b, tokens_across_layers=256*b, slots=slots,
            singleton_slots=singles, singleton_fraction=singles/slots,
            fitted_iid_q_fraction=sum(float(r["iid_fitted_q_singleton_fraction"])*int(r["slots"])
                                      for r in rows)/slots,
            uniform_iid_fraction=(1-8/64)**(b-1),
            mean_distinct_experts_per_layer_step=sum(float(r["distinct_experts_mean"])
                                                     for r in rows)/len(rows),
            unique_prefixes=b,
            source_articles=len({r["source_record"]["source_article_id"] for r in records[:b]})))
        sources.extend([source_record(trace_path), source_record(output_path)])
    analyze_routes.write_csv(OUT / "measurement_summary.csv", summary)
    old = read(ROOT / "figures/f1_native_r05/measurement_summary.json")
    comparison = []
    for new, previous in zip(summary, old["rows"]):
        assert new["batch"] == previous["batch"]
        comparison.append(dict(batch=new["batch"],
            r09_fp8_singleton_fraction=new["singleton_fraction"],
            r05_bf16_singleton_fraction=previous["singleton_fraction"],
            difference_percentage_points=100*(new["singleton_fraction"]-previous["singleton_fraction"]),
            r09_prefix_tokens=128, r05_prefix_tokens=32,
            r09_decode_steps=16, r05_decode_steps=8,
            interpretation="unpaired realized trajectories; not quantization causal effect"))
    analyze_routes.write_csv(OUT / "r05_bf16_context_comparison.csv", comparison)
    meta = dict(model="OLMoE", weight_format="FP8 W8A8",
        nonquantized_dtype=cfg["args"]["dtype"], quantization="fp8", execution="native eager",
        E=64, k=8, prefix_tokens=128, weight_bytes_per_expert_including_scales=6291464,
        rows=summary, sources=sources,
        boundaries=[boundary,
            "64 distinct nested WikiText validation prompt windows come from 60 source articles.",
            "Each q_e is a per-layer expert inclusion frequency fitted on the same 16*B tokens.",
            "The iid model assumes independent tokens; generated tokens/cohorts may be correlated.",
            "Observed-versus-fitted differences do not validate/reject iid assumptions; no independent-run CI.",
            "Occupancy is not measured DRAM traffic, performance, quality or energy.",
            "r05 BF16 used 32 prefix tokens and 8 decode steps; cross-run differences are not a paired quantization effect."],
        plot_adaptation="Existing plot_occupancy.main; only right x-axis uses log2 with integer labels.")
    (OUT / "measurement_summary.json").write_text(json.dumps(meta, indent=2)+"\n")
    savefig = Figure.savefig
    def save_log_b(self, *args, **kwargs):
        ax = self.axes[1]
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        return savefig(self, *args, **kwargs)
    Figure.savefig = save_log_b
    sys.argv = ["plot_occupancy", "--summary", *[str(OUT/f"b{b}/summary.csv") for b in BATCHES],
                "--output-dir", str(OUT), "--note",
                "OLMoE FP8 W8A8: 128-token prefixes, 16 decode steps x 16 layers per B; one short capture per batch."]
    try:
        plot_occupancy.main()
    finally:
        Figure.savefig = savefig
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

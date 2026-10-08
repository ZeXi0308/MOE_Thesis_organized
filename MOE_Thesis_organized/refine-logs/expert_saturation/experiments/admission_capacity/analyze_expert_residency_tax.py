"""Quantify the MoE expert-residency tax on KV capacity and admission ceiling.

This is a structural accounting analysis over *measured* memory telemetry. It
introduces no new execution and no policy. It answers one question that the
existing regime-qualification runs record the inputs for but do not compute:

    how much serving concurrency does a MoE engine give up simply by keeping all
    expert weights resident, and where does that land on the measured per-step
    cost staircase?

Why this is MoE-specific rather than a generic memory observation: a dense model
of equal per-token compute stores only the weights it uses. A top-k MoE stores
every expert but activates k of E per token, so a fixed fraction of HBM is held
for parameters that are idle at any instant. That held fraction is not available
to KV, and KV capacity is what bounds concurrency, which in turn selects the
CUDA-graph capture bucket that sets per-request decode cost.

Accounting rules, stated because misuse here is easy:
  * `expert_w13_w2_storage_bytes` is a SUBSET of `parameter_storage_bytes`.
    They are never summed. Non-expert parameters are the difference.
  * `kv_storage_bytes` is the allocator's KV region, not tokens in use.
  * The counterfactual is deliberately *not* presented as an achievable speedup.
    Releasing idle experts is not free: it requires paging with real transfer
    cost, which this analysis does not model and explicitly refuses to net out.
    The number produced is a capacity headroom bound, nothing more.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

GIB = 2 ** 30
# vLLM V1 capture sizes at or below max_num_seqs=32, confirmed twice: recovered
# from sealed timing reconstruction and printed by the engine itself.
CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)
# Measured pure-decode median step cost by width, from the 100-episode
# reconstruction in 20260908_step_cost_surface_r01/analysis/.
MEASURED_STEP_MS = {1: 2.806, 2: 4.027, 4: 5.200, 8: 6.751, 16: 8.374, 24: 9.344, 32: 9.964}


def ceil_capture(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def step_cost_ms(width):
    """Measured cost at the capture point this width actually executes on."""
    return MEASURED_STEP_MS.get(ceil_capture(max(1, int(width))))


def load_cell(directory):
    directory = Path(directory)
    memory = json.loads((directory / "memory-after-init.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    engine = json.loads((directory / "engine_args.json").read_text())
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    status = json.loads((directory / "status.json").read_text())
    return dict(name=directory.name, memory=memory, config=config, engine=engine,
                metrics=metrics, status=status)


def analyse(cell, kv_bytes_per_token, experts_total, experts_per_token):
    memory = cell["memory"]
    parameters = memory["parameter_storage_bytes"]
    experts = memory["expert_w13_w2_storage_bytes"]
    kv = memory["kv_storage_bytes"]
    if experts > parameters:
        raise ValueError("expert storage must be a subset of parameter storage")
    non_expert = parameters - experts
    active_fraction = experts_per_token / experts_total
    # Bytes held for experts that are idle for any given token. This is a
    # residency property, not a claim that they are never used.
    idle_expert_bytes = experts * (1.0 - active_fraction)

    engine = cell["engine"]
    context = engine.get("max_model_len")
    max_seqs = engine.get("max_num_seqs")
    kv_tokens = kv / kv_bytes_per_token
    concurrency_now = kv_tokens / context if context else None
    # Counterfactual capacity if the idle expert bytes were KV instead.
    kv_if_released = kv + idle_expert_bytes
    concurrency_if_released = (kv_if_released / kv_bytes_per_token) / context if context else None

    admission_ceiling_now = min(max_seqs, int(concurrency_now)) if concurrency_now else None
    admission_ceiling_released = (min(max_seqs, int(concurrency_if_released))
                                  if concurrency_if_released else None)
    cost_now = step_cost_ms(admission_ceiling_now) if admission_ceiling_now else None
    cost_released = step_cost_ms(admission_ceiling_released) if admission_ceiling_released else None

    result = dict(
        cell=cell["name"], status=cell["status"].get("status"),
        max_model_len=context, engine_max_num_seqs=max_seqs,
        gpu_memory_utilization=engine.get("gpu_memory_utilization"),
        parameter_gib=parameters / GIB, expert_gib=experts / GIB,
        non_expert_parameter_gib=non_expert / GIB,
        expert_share_of_parameters=experts / parameters,
        kv_region_gib=kv / GIB,
        expert_gib_per_kv_gib=experts / kv,
        experts_total=experts_total, experts_per_token=experts_per_token,
        active_expert_fraction=active_fraction,
        idle_expert_gib=idle_expert_bytes / GIB,
        idle_expert_bytes_as_share_of_kv=idle_expert_bytes / kv,
        kv_tokens_now=kv_tokens,
        concurrent_full_context_requests_now=concurrency_now,
        concurrent_full_context_requests_if_idle_experts_released=concurrency_if_released,
        concurrency_multiplier=(concurrency_if_released / concurrency_now
                                if concurrency_now else None),
        admission_ceiling_now=admission_ceiling_now,
        admission_ceiling_if_released=admission_ceiling_released,
        capture_bucket_now=ceil_capture(admission_ceiling_now) if admission_ceiling_now else None,
        capture_bucket_if_released=(ceil_capture(admission_ceiling_released)
                                    if admission_ceiling_released else None),
        step_ms_at_ceiling_now=cost_now, step_ms_at_ceiling_if_released=cost_released,
        per_request_ms_now=(cost_now / admission_ceiling_now) if cost_now else None,
        per_request_ms_if_released=((cost_released / admission_ceiling_released)
                                    if cost_released else None))
    if result["per_request_ms_now"] and result["per_request_ms_if_released"]:
        result["per_request_cost_ratio"] = (result["per_request_ms_if_released"]
                                            / result["per_request_ms_now"])
    metrics = cell["metrics"]
    if metrics:
        latency = metrics.get("latency_s", {})
        result.update(
            measured_goodput_rps=metrics.get("goodput_rps"),
            measured_throughput_rps=metrics.get("throughput_rps"),
            measured_slo_attainment=metrics.get("slo_attainment"),
            measured_n_completed=metrics.get("n_completed"),
            measured_ttft_p50_ms=(latency.get("ttft", {}).get("p50") or 0) * 1000 or None,
            measured_tpot_p50_ms=(latency.get("tpot", {}).get("p50") or 0) * 1000 or None,
            measured_itl_p50_ms=(latency.get("itl", {}).get("p50") or 0) * 1000 or None,
            observation_duration_s=metrics.get("observation_duration_s"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", required=True,
                        help="a results cell directory containing memory-after-init.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, default=131072,
                        help="16 layers * 16 kv heads * 128 head_dim * 2 (K,V) * 2 bytes")
    parser.add_argument("--experts-total", type=int, default=64)
    parser.add_argument("--experts-per-token", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)

    rows = [analyse(load_cell(path), args.kv_bytes_per_token,
                    args.experts_total, args.experts_per_token) for path in args.cell]

    report = dict(
        evidence_type="STRUCTURAL_ACCOUNTING_OVER_MEASURED_MEMORY_TELEMETRY_NO_NEW_EXECUTION",
        claim_boundary=(
            "expert/KV byte figures are measured; the released-expert column is a capacity "
            "headroom bound, NOT an achievable speedup: it excludes paging transfer cost, "
            "latency and any effect on route or quality"),
        accounting_rules=[
            "expert storage is a subset of parameter storage; the two are never summed",
            "kv_storage_bytes is the allocator KV region, not tokens in use",
            "step costs are measured medians at the capture point a width executes on",
        ],
        kv_bytes_per_token=args.kv_bytes_per_token,
        experts_total=args.experts_total, experts_per_token=args.experts_per_token,
        measured_step_ms_by_capture_point=MEASURED_STEP_MS, cells=rows)
    (out / "residency_tax.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# MoE expert-residency tax on KV capacity and admission ceiling", "",
             "Structural accounting over measured memory telemetry. No new execution.",
             "The released-expert column is a **capacity bound, not a speedup**: paging cost",
             "is deliberately not netted out.", "",
             "## Measured memory split", "",
             "| cell | params GiB | **experts GiB** | expert share | non-expert GiB | "
             "KV region GiB | expert GiB per KV GiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f'| {r["cell"]} | {r["parameter_gib"]:.2f} | **{r["expert_gib"]:.2f}** | '
                     f'{r["expert_share_of_parameters"]:.3f} | {r["non_expert_parameter_gib"]:.2f} | '
                     f'{r["kv_region_gib"]:.2f} | {r["expert_gib_per_kv_gib"]:.3f} |')
    lines += ["", "## Residency tax and admission ceiling", "",
              f'top-k = {args.experts_per_token}/{args.experts_total} '
              f'({100*args.experts_per_token/args.experts_total:.1f}% of experts active per token)', "",
              "| cell | context | idle expert GiB | idle/KV | concurrency now | if released | "
              "x | ceiling now | ceiling if released | bucket now -> if released |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f'| {r["cell"]} | {r["max_model_len"]} | {r["idle_expert_gib"]:.2f} | '
            f'{r["idle_expert_bytes_as_share_of_kv"]:.3f} | '
            f'{r["concurrent_full_context_requests_now"]:.1f} | '
            f'{r["concurrent_full_context_requests_if_idle_experts_released"]:.1f} | '
            f'{r["concurrency_multiplier"]:.2f} | {r["admission_ceiling_now"]} | '
            f'{r["admission_ceiling_if_released"]} | '
            f'{r["capture_bucket_now"]} -> {r["capture_bucket_if_released"]} |')
    lines += ["", "## Where the ceiling lands on the measured cost staircase", "",
              "| cell | step ms at ceiling now | if released | per-request ms now | "
              "if released | ratio |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r.get("per_request_cost_ratio") is None:
            continue
        lines.append(f'| {r["cell"]} | {r["step_ms_at_ceiling_now"]:.3f} | '
                     f'{r["step_ms_at_ceiling_if_released"]:.3f} | '
                     f'{r["per_request_ms_now"]:.4f} | {r["per_request_ms_if_released"]:.4f} | '
                     f'{r["per_request_cost_ratio"]:.4f} |')
    measured = [r for r in rows if r.get("measured_goodput_rps") is not None]
    if measured:
        lines += ["", "## Measured request-level outcomes in the same cells", "",
                  "| cell | completed | SLO attainment | goodput rps | throughput rps | "
                  "TTFT p50 ms | TPOT p50 ms | ITL p50 ms | duration s |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in measured:
            lines.append(
                f'| {r["cell"]} | {r["measured_n_completed"]} | {r["measured_slo_attainment"]:.3f} | '
                f'{r["measured_goodput_rps"]:.4f} | {r["measured_throughput_rps"]:.4f} | '
                f'{r["measured_ttft_p50_ms"]:.1f} | {r["measured_tpot_p50_ms"]:.2f} | '
                f'{r["measured_itl_p50_ms"]:.2f} | {r["observation_duration_s"]:.2f} |')
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"cells {len(rows)}; wrote {out}")
    for r in rows:
        print(f'  {r["cell"]}: experts {r["expert_gib"]:.2f} GiB '
              f'({r["expert_share_of_parameters"]:.1%} of params), KV {r["kv_region_gib"]:.2f} GiB, '
              f'idle-expert {r["idle_expert_gib"]:.2f} GiB = {r["idle_expert_bytes_as_share_of_kv"]:.1%} of KV, '
              f'concurrency {r["concurrent_full_context_requests_now"]:.1f} -> '
              f'{r["concurrent_full_context_requests_if_idle_experts_released"]:.1f}')


if __name__ == "__main__":
    main()

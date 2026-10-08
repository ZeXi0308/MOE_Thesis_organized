"""Summarize the two frozen formal CPU replays; never run an experiment.

Only formal_medium/results.json and formal_high/results.json are accepted.
Pilot and profiling runs are intentionally excluded because their harness was
not the final measurement configuration. Two repetitions do not justify a CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("native", "fixed1", "fixed8", "cost")
LOADS = ("medium", "high")
METRICS = (
    "light_delivery_p99_ms", "light_planned_delivery_p99_ms",
    "heavy_delivery_p99_ms", "heavy_planned_delivery_p99_ms",
    "light_completion_p99_s", "heavy_completion_p99_s",
    "light_gap_p99_ms", "heavy_gap_p99_ms",
    "tokens_s", "req_s", "elapsed_s",
    "server_cpu_pct", "client_cpu_pct", "server_cpu_s", "client_cpu_s",
    "server_peak_rss_mib", "client_peak_rss_mib",
    "light_drain_max_ms", "heavy_drain_max_ms", "producer_lateness_p99_ms",
)


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def change_pct(value, baseline):
    """Positive means an increase; caller supplies the direction of benefit."""
    if baseline == 0:
        return None
    return 100.0 * (value / baseline - 1.0)


def describe(values):
    return {"values": values, "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def check_number(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{label}: expected a finite nonnegative number, got {value!r}")


def analyze_load(path, load):
    trace_path = path.parent.parent / "inputs" / f"{load}.json"
    trace_raw = trace_path.read_bytes()
    trace = json.loads(trace_raw)
    source_times = [batch["time_s"] for batch in trace["batches"]]
    intervals_ms = [(right - left) * 1000.0
                    for left, right in zip(source_times, source_times[1:])]
    if not intervals_ms or any(value <= 0 for value in intervals_ms):
        raise ValueError(f"{trace_path}: require strictly increasing batch times")
    pacing_threshold_ms = statistics.median(intervals_ms)
    raw = path.read_bytes()
    runs = json.loads(raw)
    if not isinstance(runs, list) or len(runs) != 8:
        raise ValueError(f"{path}: require exactly eight formal runs (four arms × two repetitions)")
    groups = {arm: [] for arm in ARMS}
    reference_semantics = None
    reference_work = None
    summarized_runs = []
    for index, run in enumerate(runs):
        arm = run.get("arm")
        if arm not in groups or run.get("trace") != load:
            raise ValueError(f"{path} run {index}: unexpected arm/trace")
        if run.get("profile") is not False:
            raise ValueError(f"{path} run {index}: profiling or unspecified profiling status is excluded")
        for metric in METRICS:
            check_number(run.get(metric), f"{path} run {index} {metric}")
        for metric in ("tokens_s", "req_s", "elapsed_s", "light_delivery_p99_ms"):
            if run[metric] <= 0:
                raise ValueError(f"{path} run {index}: {metric} must be positive")
        semantics = run.get("semantics")
        if not isinstance(semantics, dict) or not semantics:
            raise ValueError(f"{path} run {index}: missing per-request semantic hashes")
        for rid, record in semantics.items():
            for field in ("text_sha256", "logprobs_sha256"):
                value = record.get(field)
                if (not isinstance(value, str) or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)):
                    raise ValueError(f"{path} run {index} {rid}: invalid {field}")
            count = record.get("logprobs_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{path} run {index} {rid}: invalid logprobs_count")
        if run.get("requests") != len(semantics):
            raise ValueError(f"{path} run {index}: request count differs from semantic ledger")
        work = (run.get("requests"), run.get("tokens"))
        if reference_semantics is None:
            reference_semantics, reference_work = semantics, work
        if semantics != reference_semantics:
            raise ValueError(f"{path} run {index}: semantic hashes/counts differ across formal arms")
        if work != reference_work:
            raise ValueError(f"{path} run {index}: request/token workload changed")
        if not isinstance(run.get("all_completed"), bool):
            raise ValueError(f"{path} run {index}: missing all_completed boolean")
        record = {k: v for k, v in run.items() if k != "semantics"}
        pacing_warning = run["producer_lateness_p99_ms"] > pacing_threshold_ms
        record.update(run_index=index, repetition_index=len(groups[arm]),
                      semantics_sha256=canonical_hash(semantics),
                      source_pacing_warning=pacing_warning,
                      source_pacing_threshold_ms=pacing_threshold_ms,
                      source_pacing_warning_reason=(
                          "Producer P99 lateness exceeds the trace median batch interval; source timing differs materially."
                          if pacing_warning else None))
        summarized_runs.append(record)
        groups[arm].append(record)
    for arm, group in groups.items():
        if len(group) != 2:
            raise ValueError(f"{path}: arm {arm} requires exactly two repetitions")

    by_arm = {}
    for arm, group in groups.items():
        by_arm[arm] = {
            "run_indices": [r["run_index"] for r in group],
            "repetitions": len(group),
            "metrics": {key: describe([r[key] for r in group]) for key in METRICS},
            "all_completed": all(r["all_completed"] for r in group),
            "completed_runs": sum(r["all_completed"] for r in group),
            "semantics_equal": True,
            "source_pacing_warning": [r["source_pacing_warning"] for r in group],
            "source_pacing_valid_runs": sum(not r["source_pacing_warning"] for r in group),
        }
    # Choose the strongest tested simple baseline once per workload by the
    # prespecified primary metric, then use that arm for both repetition pairs.
    best = min(("fixed1", "fixed8"),
               key=lambda arm: by_arm[arm]["metrics"]["light_delivery_p99_ms"]["median"])
    base, candidate = by_arm[best]["metrics"], by_arm["cost"]["metrics"]
    comparisons = {}
    for key in METRICS:
        comparisons[key] = {
            "median_change_pct": change_pct(candidate[key]["median"], base[key]["median"]),
            "paired_change_pct": [change_pct(c[key], b[key])
                                  for c, b in zip(groups["cost"], groups[best])],
        }
    light = comparisons["light_delivery_p99_ms"]
    light_improvement = -light["median_change_pct"]
    paired_improvement = [-v for v in light["paired_change_pct"]]
    throughput_change = comparisons["tokens_s"]["median_change_pct"]
    req_throughput_change = comparisons["req_s"]["median_change_pct"]
    completion_valid = all(record["all_completed"] for record in by_arm.values())
    numeric_gate = (light_improvement >= 20.0 and throughput_change >= -5.0
                    and req_throughput_change >= -5.0 and completion_valid)
    pacing_warning_indices = [r["run_index"] for r in summarized_runs
                              if r["source_pacing_warning"]]
    # Keep all numerical comparisons, but do not treat an altered source pace
    # as causal evidence for or against a method. No runs are removed/replaced.
    pacing_valid = not pacing_warning_indices
    heavy_keys = ("heavy_delivery_p99_ms", "heavy_completion_p99_s", "heavy_drain_max_ms")
    return {
        "load": load, "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "run_order": [r["arm"] for r in runs],
        "requests": reference_work[0], "tokens": reference_work[1],
        "source_pacing_quality": {
            "trace_path": str(trace_path),
            "trace_sha256": hashlib.sha256(trace_raw).hexdigest(),
            "median_source_batch_interval_ms": pacing_threshold_ms,
            "warning_rule": "producer_lateness_p99_ms > median original batch interval in ms",
            "threshold_status": "Post-run diagnostic quality flag, not a preregistered effect threshold; applied identically to all runs.",
            "warning_run_indices": pacing_warning_indices,
            "all_runs_retained": True,
            "causal_comparison_supported_by_pacing_check": pacing_valid,
            "interpretation": (
                "No warning under this diagnostic; absence of a flag alone does not establish causal validity."
                if pacing_valid else
                "Source pacing is compromised in at least one run. All raw values and medians remain visible, but this workload comparison is descriptive only; do not attribute its negative or positive changes to the candidate or use its screening gate as causal evidence."
            ),
        },
        "semantics_validation": {
            "all_eight_runs_equal": True,
            "per_request_fields": ["text_sha256", "logprobs_sha256", "logprobs_count"],
            "requests_checked_per_run": len(reference_semantics),
            "canonical_sha256": canonical_hash(reference_semantics),
            "scope": "Within workload across all arms/repetitions; workloads differ. Does not independently prove original expected output correctness.",
        },
        "arms": by_arm, "runs": summarized_runs,
        "candidate_vs_strongest_simple": {
            "candidate": "cost", "baseline": best,
            "baseline_selection": "Minimum median light_delivery_p99_ms among fixed1 and fixed8",
            "pairing": "First occurrence versus first occurrence, second versus second; same chosen baseline for both pairs",
            "light_p99_median_improvement_pct": light_improvement,
            "light_p99_paired_improvement_pct": paired_improvement,
            "tokens_s_median_change_pct": throughput_change,
            "req_s_median_change_pct": req_throughput_change,
            "heavy_median_regressions": {
                key: comparisons[key]["median_change_pct"] for key in heavy_keys
                if comparisons[key]["median_change_pct"] is not None
                and comparisons[key]["median_change_pct"] > 0
            },
            "all_metric_changes": comparisons,
            "comparison_interpretation": "descriptive_only_source_pacing_warning" if not pacing_valid else "no_source_pacing_warning",
            "screening_data": {
                "light_improvement_at_least_20pct": light_improvement >= 20.0,
                "tokens_throughput_loss_at_most_5pct": throughput_change >= -5.0,
                "requests_throughput_loss_at_most_5pct": req_throughput_change >= -5.0,
                "all_arms_all_requests_completed": completion_valid,
                "both_light_pairs_improve": all(v > 0 for v in paired_improvement),
                "both_light_pairs_meet_20pct": all(v >= 20.0 for v in paired_improvement),
                "raw_numeric_gate_only": numeric_gate,
                "source_pacing_valid": pacing_valid,
                "numeric_gate_only": numeric_gate if pacing_valid else None,
                "gate_status": ("indeterminate_source_pacing" if not pacing_valid
                                else "pass_numeric_only" if numeric_gate else "fail_numeric_only"),
                "heavy_side_effect_review_required": True,
                "paper_decision": "Not made by this script. Heavy regressions, starvation, replay validity, method value and two-repetition noise need explicit review.",
            },
        },
    }


def fmt(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def render(summary):
    lines = [
        "# F 正式 CPU 回放主结果",
        "",
        "仅使用 `formal_medium/results.json` 与 `formal_high/results.json`；排除 diagnostic/profile/pilot/pilot_gc。每组每方法两次，完整列出主指标重复值；不估计置信区间，不据两次重复声称统计显著。",
        "",
        "轻/重交付为 token 产出到客户端完成 SSE JSON 解析的 P99。除轻请求重复值和 RSS/排空最大值外，表中为两次运行的中位数。CPU 为单个绑定逻辑核的利用率；吞吐计入全部请求完成。",
        "",
        "† 标记 producer P99 迟到超过该轨迹原始批间隔中位数的运行。这是事后统一应用的时序质量诊断，不能当作预注册阈值。所有原始数值、重复和中位数照常保留，不删除、不补跑；出现标记的负载比较仅作描述，门槛不可判，其收益或退化不得归因候选方法。",
        "",
        "| 负载 | 方法 | 轻交付 P99 两次 → 中位数 ms | 重交付 P99 ms / 完成 P99 s | tokens/s / requests/s | CPU server/client % | 峰 RSS server/client MiB | 最大排空 light/heavy ms | 完成运行数 | producer 迟到 P99 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for load in LOADS:
        data = summary["workloads"][load]
        for arm in ARMS:
            item = data["arms"][arm]
            metrics = item["metrics"]
            med = lambda key: fmt(metrics[key]["median"])
            peak = lambda key: fmt(metrics[key]["max"])
            light = metrics["light_delivery_p99_ms"]
            light_text = ", ".join(fmt(v) + ("†" if warning else "")
                                   for v, warning in zip(light["values"], item["source_pacing_warning"]))
            light_text += " → " + fmt(light["median"])
            cells = [load, arm, light_text,
                     med("heavy_delivery_p99_ms") + " / " + med("heavy_completion_p99_s"),
                     med("tokens_s") + " / " + med("req_s"),
                     med("server_cpu_pct") + " / " + med("client_cpu_pct"),
                     peak("server_peak_rss_mib") + " / " + peak("client_peak_rss_mib"),
                     peak("light_drain_max_ms") + " / " + peak("heavy_drain_max_ms"),
                     f'{item["completed_runs"]}/2', med("producer_lateness_p99_ms")]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "", "候选 `cost` 只与各负载中主指标中位数最小的简单方案比较；选择一次后保留同一基线做两次配对。正改善表示轻延迟下降；吞吐和重延迟的正变化表示对应数值上升。", "",
        "| 负载 | 最强简单基线 | 轻 P99 中位数改善 % | 配对 0 / 1 改善 % | tokens/s 变化 % | requests/s 变化 % | 重交付 P99 变化 % | 重完成 P99 变化 % | 数值门槛 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for load in LOADS:
        comp = summary["workloads"][load]["candidate_vs_strongest_simple"]
        changes = comp["all_metric_changes"]
        gate = comp["screening_data"]["numeric_gate_only"]
        gate_text = ("不可判（source pacing）" if gate is None else
                     "通过（仍需副作用审查）" if gate else "未通过")
        cells = [load, comp["baseline"], fmt(comp["light_p99_median_improvement_pct"]),
                 " / ".join(fmt(v) for v in comp["light_p99_paired_improvement_pct"]),
                 fmt(comp["tokens_s_median_change_pct"]), fmt(comp["req_s_median_change_pct"]),
                 fmt(changes["heavy_delivery_p99_ms"]["median_change_pct"]),
                 fmt(changes["heavy_completion_p99_s"]["median_change_pct"]),
                 gate_text]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "", "数值门槛：相对最强简单方案，轻请求交付 P99 改善至少 20%，全部完成吞吐损失不超过 5%，且所有请求完成。重请求无饥饿、显著退化必须披露；没有预先规定重延迟显著退化的数值阈值，脚本不补设阈值，也不自动给出论文继续/停止判决。", "",
        "每负载各八次运行的全部请求 text/logprobs 哈希与 logprobs 数量一致。这里只验证跨方法一致性，原始完整输出正确性仍依赖采集与独立语义检查。原始运行顺序、全部指标重复值、配对变化、语义哈希与输入文件 SHA-256 均在 `summary.json`。", "",
        "内存口径：server 峰 RSS 包含已加载原生组件和预构建回放数据；client ru_maxrss 是同一驱动进程的累计高水位，不能把各方法 client RSS 的先后差异解释为方法导致的独立峰值差异。全部完成仅排除观察窗口内未完成，不能单独证明普遍无饥饿。", "",
        "产出时序质量（全部 16 次运行，不筛选）：", "",
        "| 负载 | 运行 index / 方法 / repetition | producer 迟到 P99 ms | 原始批间隔中位数 ms | source_pacing_warning |",
        "|---|---|---:|---:|---|",
    ]
    for load in LOADS:
        for run in summary["workloads"][load]["runs"]:
            cells = [load, f'{run["run_index"]:02d} / {run["arm"]} / {run["repetition_index"]}',
                     fmt(run["producer_lateness_p99_ms"]), fmt(run["source_pacing_threshold_ms"]),
                     "true †" if run["source_pacing_warning"] else "false"]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "", "主指标仍以实际 producer emit 为起点；planned-ready 指标保留对原始时间计划的偏差。重新从 emit 计时并不能恢复被迟到改变的批次间隔，因此标记运行的两个指标都保留、都不单独支持方法归因。", "",
        "复核数据：", "",
        "| 负载 | 方法 | 轻 planned-ready P99 ms | 重最大排空两次 ms | producer P99 迟到两次 ms |",
        "|---|---|---:|---:|---:|",
    ]
    for load in LOADS:
        for arm in ARMS:
            metrics = summary["workloads"][load]["arms"][arm]["metrics"]
            lists = [", ".join(fmt(v) for v in metrics[key]["values"])
                     for key in ("light_planned_delivery_p99_ms", "heavy_drain_max_ms", "producer_lateness_p99_ms")]
            lines.append("| " + " | ".join([load, arm, *lists]) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Directory containing formal_medium and formal_high (default: this script's directory)")
    parser.add_argument("--markdown", type=Path,
                        help="Markdown output path (default: ROOT/main_results.md)")
    parser.add_argument("--summary", type=Path,
                        help="JSON output path (default: ROOT/summary.json)")
    args = parser.parse_args()
    root = args.root.resolve()
    summary = {
        "schema": 2,
        "scope": "Formal CPU replay only; no GPU end-to-end claim",
        "excluded": ["diagnostic", "profile", "pilot", "pilot_gc"],
        "repetitions_per_arm": 2,
        "uncertainty": "Two observed repetitions, descriptive paired changes only; no confidence interval or statistical significance claim.",
        "workloads": {load: analyze_load(root / f"formal_{load}" / "results.json", load)
                      for load in LOADS},
    }
    # Both input sets must validate before either deliverable is written.
    markdown = render(summary)
    summary_path = args.summary or root / "summary.json"
    markdown_path = args.markdown or root / "main_results.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n")
    markdown_path.write_text(markdown)
    print(json.dumps({"summary": str(summary_path), "markdown": str(markdown_path),
                      "formal_runs": 16, "all_semantic_hashes_match": True},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

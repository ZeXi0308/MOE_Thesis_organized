"""Analyze the frozen exploratory 2x2 CPU endpoint experiment (stdlib only).

Run units, not tokens, are the independent repetitions. Token-level tail
decomposition is descriptive and uses the same tail membership for all stages.
No experiment, parameter tuning, data exclusion, or significance test is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_SHA = "cb813b551ed43b8aafbfedda87d7f3bba57fbd9cb4e6d993d17d5a63a728a32b"
ORDER = ["fixed1", "priority", "cache", "cache_priority",
         "cache_priority", "cache", "priority", "fixed1"]
LABELS = {"fixed1": "A", "priority": "P", "cache": "D", "cache_priority": "DP"}
STAGES = ("emit_receive_ms", "receive_collector_ms", "collector_client_ms")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q(xs, fraction):
    values = sorted(xs)
    if not values:
        return None
    pos = (len(values) - 1) * fraction
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)


def same_float(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(f"{label}: recomputed {actual}, recorded {expected}")


def stats(xs):
    return {"n": len(xs), "mean": statistics.mean(xs),
            "p50": q(xs, .5), "p95": q(xs, .95), "p99": q(xs, .99), "max": max(xs)}


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def token_batches(trace):
    mapping = {r["request_id"]: [None] * len(r["output_token_ids"])
               for r in trace["requests"]}
    requests = {r["request_id"]: r for r in trace["requests"]}
    for index, batch in enumerate(trace["batches"]):
        for out in batch["outputs"]:
            rid = out["request_id"]
            start = out.get("token_start", out.get("position", 0))
            end = out.get("token_end", start + 1)
            for pos in range(start, end):
                assert mapping[rid][pos] is None, (rid, pos, "duplicate trace token")
                assert requests[rid]["token_times_s"][pos] == batch["time_s"]
                mapping[rid][pos] = index
    assert all(index is not None for indices in mapping.values() for index in indices)
    return mapping


def execution_identity(manifest, trace):
    paths = [HERE / "endpoint_server.py", HERE / "run_endpoint.py",
             HERE.parent / "native_adapter.py", HERE.parent / "bootstrap.py", trace]
    hashes = {str(p.relative_to(HERE.parent)): sha(p) for p in paths}
    result = {"local_sha256": hashes, "manifest": str(manifest),
              "status": "未验证远程执行源码身份；本地SHA仅作当前复现记录"}
    if not manifest.is_file():
        return result
    recorded = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower())
        recorded.setdefault(Path(filename.lstrip("*")).name, set()).add(digest.lower())
    checked = {}
    for name, digest in hashes.items():
        candidates = recorded.get(Path(name).name, set())
        checked[name] = bool(candidates) and candidates == {digest}
    result.update(recorded_names={k: sorted(v) for k, v in recorded.items()},
                  matches=checked, all_match=all(checked.values()))
    result["status"] = ("已匹配保存的远程源码/输入SHA清单（清单本身的采集时点另见执行记录）"
                        if result["all_match"] else "源码/输入SHA清单缺项或不一致；不能确认执行身份")
    return result


def analyze_run(index, recorded, runpath, trace, mapping, expected_semantics, interval_median):
    result = read(runpath / "result.json")
    assert result == recorded, (runpath, "aggregate/per-run result mismatch")
    server, client, producer = [read(runpath / (name + ".json"))
                                for name in ("server", "client", "producer")]
    assert result["arm"] == server["arm"] == ORDER[index]
    assert result["semantics"] == expected_semantics, (runpath, "cross-arm content mismatch")
    requests = {r["request_id"]: r for r in trace["requests"]}
    assert set(client) == set(server["ledger"]) == set(requests)
    assert result["all_completed"] and result["requests"] == len(requests)
    assert server["affinity"] == [2] and producer["affinity"] == [6]
    assert server["native_default_chunk_size"] == 1
    assert not result["profile"]
    epoch = producer["epoch_ns"]
    assert epoch == server["epoch_ns"]
    received = dict(server["source_received"])
    events = producer["events"]
    assert [e[0] for e in events] == list(range(len(trace["batches"])))
    assert [x[0] for x in server["source_received"]] == list(range(len(events)))
    for event, batch in zip(events, trace["batches"]):
        assert event[1] == epoch + int(batch["time_s"] * 1e9)
        assert event[3] >= event[2]
    rows = {kind: [] for kind in ("light", "heavy")}
    coalescing = {kind: Counter() for kind in rows}
    for rid, request in requests.items():
        ledger, item = server["ledger"][rid], client[rid]
        kind = "heavy" if request["heavy"] else "light"
        assert len(ledger) == len(item["visible_ns"]) == item["chunks"]
        assert item["finish_reason"] == "length"
        assert item["usage"]["completion_tokens"] == len(mapping[rid])
        assert item["logprobs_count"] == (len(mapping[rid]) if request["heavy"] else 0)
        assert {key: item[key] for key in expected_semantics[rid]} == expected_semantics[rid]
        assert item["visible_ns"] == sorted(item["visible_ns"])
        assert item["complete_ns"] >= max(item["visible_ns"])
        position = 0
        for consumption, visible in zip(ledger, item["visible_ns"]):
            count, consumed = consumption["token_count"], consumption["consumed_ns"]
            coalescing[kind]["events"] += 1
            coalescing[kind]["merged_events"] += count > 1
            coalescing[kind]["tokens"] += count
            for batch_index in mapping[rid][position:position + count]:
                planned, emitted = events[batch_index][1:3]
                source_received = received[batch_index]
                assert emitted <= source_received <= consumed <= visible
                rows[kind].append({"delivery_ms": (visible - emitted) / 1e6,
                    "planned_delivery_ms": (visible - planned) / 1e6,
                    "emit_receive_ms": (source_received - emitted) / 1e6,
                    "receive_collector_ms": (consumed - source_received) / 1e6,
                    "collector_client_ms": (visible - consumed) / 1e6})
            position += count
        assert position == len(mapping[rid])
    assert sum(len(v) for v in rows.values()) == result["tokens"]
    decomposition = {}
    for kind, samples in rows.items():
        total = [x["delivery_ms"] for x in samples]
        threshold = q(total, .99)
        same_float(threshold, result[f"{kind}_delivery_p99_ms"], f"{runpath}/{kind} delivery")
        same_float(q([x["planned_delivery_ms"] for x in samples], .99),
                   result[f"{kind}_planned_delivery_p99_ms"], f"{runpath}/{kind} planned")
        tail = [x for x in samples if x["delivery_ms"] >= threshold]
        tail_means = {name: statistics.mean(x[name] for x in tail)
                      for name in (*STAGES, "delivery_ms")}
        same_float(sum(tail_means[name] for name in STAGES), tail_means["delivery_ms"],
                   f"{runpath}/{kind} same-tail identity")
        decomposition[kind] = {
            "tokens": len(samples), "stage_ms": {key: stats([x[key] for x in samples]) for key in STAGES},
            "tail": {"rule": "delivery_ms >= this run/class empirical P99", "threshold_ms": threshold,
                     "tokens": len(tail), "conditional_mean_ms": tail_means},
            "coalescing": dict(coalescing[kind])}
    late = [(e[2] - e[1]) / 1e6 for e in events]
    interval_error = [abs(((y[2] - x[2]) - (y[1] - x[1])) / 1e6) for x, y in zip(events, events[1:])]
    same_float(q(late, .99), result["producer_lateness_p99_ms"], f"{runpath}/producer lateness")
    pacing = {"lateness_ms": stats(late), "absolute_interbatch_error_ms": stats(interval_error),
              "frozen_warning_threshold_ms": interval_median,
              "warning": q(late, .99) > interval_median}
    cache = server.get("endpoint", {}).get("cache")
    if cache:
        assert cache["maxsize"] == 4096 and cache["currsize"] <= 4096
        cache = dict(cache, hit_fraction=cache["hits"] / (cache["hits"] + cache["misses"]))
    return {"index": index, "repetition": 0 if index < 4 else 1,
            "run": runpath.name, "arm": result["arm"],
            "metrics": {k: v for k, v in result.items() if k != "semantics"},
            "decomposition": decomposition, "pacing": pacing, "cache": cache,
            "versions": server["versions"], "checks": {
                "aggregate_matches_per_run": True, "complete_request_token_ledger": True,
                "cross_arm_recorded_semantic_hashes": True, "finish_usage_logprobs_counts": True,
                "source_plan_batch_ids_and_order": True, "causal_timestamp_order": True,
                "recomputed_delivery_metrics": True, "server_producer_affinity": True}}


def build_report(summary):
    runs, comparisons = summary["runs"], summary["comparisons"]
    out = ["# 冻结 2×2 CPU 端点诊断结果", "",
        "本组是探索性动作诊断，不是论文独立确认；每臂两次独立运行，完整保留正反顺序的8次结果。不设百分比成功判据、不计算伪独立token置信区间。A=fixed1 FIFO，P=稳定light-first，D=4096项在线解码LRU，DP=D+P。", "",
        "主指标为轻token实际source emit→客户端完整SSE解析P99；同时报告原计划时间口径。吞吐为固定开放环产出、全部完成且排空后的吞吐，不能解释为最大可承载能力。", "",
        "## 全部独立运行", ""]
    out += markdown_table(["运行/臂/重复", "light P99/计划 P99 ms", "heavy P99 ms", "gap P99 L/H ms", "完成 P99 L/H s", "最大排空 L/H ms", "tokens/s · req/s", "完成请求"], [
        [f"{r['index']:02d}/{LABELS[r['arm']]}/{r['repetition']}",
         f"{fmt(m['light_delivery_p99_ms'])}/{fmt(m['light_planned_delivery_p99_ms'])}", fmt(m['heavy_delivery_p99_ms']),
         f"{fmt(m['light_gap_p99_ms'])}/{fmt(m['heavy_gap_p99_ms'])}",
         f"{fmt(m['light_completion_p99_s'])}/{fmt(m['heavy_completion_p99_s'])}",
         f"{fmt(m['light_drain_max_ms'])}/{fmt(m['heavy_drain_max_ms'])}",
         f"{fmt(m['tokens_s'])} · {fmt(m['req_s'])}", f"{m['requests']}/{summary['requests']}"]
        for r in runs for m in [r["metrics"]]])
    out += ["", "资源与缓存（client RSS是同一驱动进程累计高水位，不能作为独立方法内存差异）：", ""]
    out += markdown_table(["运行/臂", "CPU server/client %", "CPU server/client s", "峰RSS server/client MiB", "OutputProcessor CPU s", "cache hits/misses/命中率", "source迟到 P99/max ms", "批间隔绝对误差 P99/max ms", "时序标记"], [
        [f"{r['index']:02d}/{LABELS[r['arm']]}", f"{fmt(m['server_cpu_pct'])}/{fmt(m['client_cpu_pct'])}",
         f"{fmt(m['server_cpu_s'])}/{fmt(m['client_cpu_s'])}",
         f"{fmt(m['server_peak_rss_mib'])}/{fmt(m['client_peak_rss_mib'])}", fmt(m['output_processor']['cpu_ns'] / 1e9),
         "—" if not r['cache'] else f"{r['cache']['hits']}/{r['cache']['misses']}/{fmt(100*r['cache']['hit_fraction'])}%",
         f"{fmt(p['lateness_ms']['p99'])}/{fmt(p['lateness_ms']['max'])}",
         f"{fmt(p['absolute_interbatch_error_ms']['p99'])}/{fmt(p['absolute_interbatch_error_ms']['max'])}", "失配，保留但不归因" if p['warning'] else "未触发"]
        for r in runs for m, p in [(r['metrics'], r['pacing'])]])
    out += ["", f"预注册时序警报阈值为原batch间隔中位数 {fmt(summary['original_interbatch_median_ms'])} ms。P99未触发不等于全部批次严格匹配，max及间隔误差照常披露。", "",
            "## 四个动作对比", "", "每对按正序/反序分别配对；负Δ表示实验臂数值下降。light改善%采用正值=下降；其余变化%采用正值=上升。两次配对仍不是统计显著性证明。", ""]
    out += markdown_table(["对比/重复 (实验-参照)", "light Δms/改善%", "heavy Δms/变化%", "tokens/s变化%", "server CPU变化%", "公平时序可解释"], [
        [f"{c['name']}/{c['repetition']} ({c['experimental_run']}-{c['reference_run']})",
         f"{fmt(c['light_delta_ms'])}/{fmt(c['light_improvement_pct'])}%",
         f"{fmt(c['heavy_delta_ms'])}/{fmt(c['heavy_change_pct'])}%",
         f"{fmt(c['tokens_s_change_pct'])}%", f"{fmt(c['server_cpu_change_pct'])}%",
         "未触发时序警报" if c['pacing_comparable'] else "否；保留描述数值"] for c in comparisons])
    out += ["", "## 等待分解", "", "逐token按同一请求游标接合原始批次、collector与client记录。阶段P99独立给出，**不能相加**；collector→client含原生chat对象、序列化、I/O及客户端解析，不能全归为网络。", ""]
    out += markdown_table(["运行/臂/类", "emit→receive P99 ms", "receive→collector P99 ms", "collector→client P99 ms", "合并事件/总事件"], [
        [f"{r['index']:02d}/{LABELS[r['arm']]}/{kind}", *[fmt(d['stage_ms'][key]['p99']) for key in STAGES],
         f"{d['coalescing'].get('merged_events', 0)}/{d['coalescing']['events']}"]
        for r in runs for kind, d in r['decomposition'].items()])
    out += ["", "以下先按**同run、同类总delivery ≥ 该类P99**选出尾token，再报告三个阶段的条件均值；这三项均值可以相加成同组总均值，不是P99相加，也不是将尾token当作独立实验。", ""]
    out += markdown_table(["运行/臂/类", "尾token数", "选取阈值 ms", "尾条件均值 source/server/post ms", "尾总均值 ms"], [
        [f"{r['index']:02d}/{LABELS[r['arm']]}/{kind}", t['tokens'], fmt(t['threshold_ms']),
         "/".join(fmt(t['conditional_mean_ms'][key]) for key in STAGES), fmt(t['conditional_mean_ms']['delivery_ms'])]
        for r in runs for kind, d in r['decomposition'].items() for t in [d['tail']]])
    out += ["", "## 完整性、身份与证据边界", "",
        f"输入SHA-256：`{summary['trace_sha256']}`，匹配冻结输入。全部{len(runs)}运行的记录语义hash跨臂一致；与旧fixed1记录一致：{summary['historical_semantic_hash_match']}。逐请求token计数、collector/client事件接合、length终止与usage、logprobs计数、全部请求完成及时间戳先后已核验。",
        "", "语义hash来自运行时保存的完整输出摘要；分析脚本核验摘要一致和账目完整，不重建已未保存的原始SSE正文。DONE由驱动在写出result前断言，独立结果中未另存DONE字段，不能宣称离线重新逐字验证。",
        "", summary['execution_identity']['status'] + "。详细SHA及运行版本保存在summary.json。",
        "", "原生单token缓存使用已知词表、当前合成top-20分布；命中率与收益不能外推真实候选token分布。无有效工作预热，冷启动首项保留；CPU/RSS包含原生组件与预建回放存储。仅CPU输出回放，未运行真实混合logprobs GPU服务；不存在已验证的GPU端到端收益。",
        "", "所有端点都是强简单策略/近邻组件适配。任何性能改善本身都不证明新的cost-isolation状态或方法有独立贡献；当前判断须结合四个动作对比及heavy代价。"]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "endpoint_high")
    parser.add_argument("--trace", type=Path, default=HERE.parent / "inputs/high.json")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--execution-hashes", type=Path, default=HERE / "execution_hashes.sha256")
    args = parser.parse_args()
    trace = read(args.trace)
    assert sha(args.trace) == EXPECTED_SHA, "frozen trace SHA mismatch"
    results = read(args.runs / "results.json")
    assert len(results) == 8 and [r['arm'] for r in results] == ORDER
    command = read(args.runs / "command.json")
    assert [command[k] for k in ('server_core', 'client_core', 'producer_core')] == [2, 4, 6]
    assert command['arms'].split(',') == ORDER
    interval_median = statistics.median((y['time_s'] - x['time_s']) * 1000
        for x, y in zip(trace['batches'], trace['batches'][1:]))
    mapping = token_batches(trace)
    baseline_semantics = results[0]['semantics']
    runs = [analyze_run(i, recorded, args.runs / f"{i:02d}_{recorded['arm']}",
                       trace, mapping, baseline_semantics, interval_median)
            for i, recorded in enumerate(results)]
    assert all(r['versions'] == runs[0]['versions'] for r in runs)
    comparisons = []
    by_arm_rep = {(r['arm'], r['repetition']): r for r in runs}
    for name, experimental, reference in [('P/A', 'priority', 'fixed1'), ('D/A', 'cache', 'fixed1'),
                                          ('DP/D', 'cache_priority', 'cache'), ('DP/P', 'cache_priority', 'priority')]:
        for repetition in (0, 1):
            e, b = by_arm_rep[experimental, repetition], by_arm_rep[reference, repetition]
            em, bm = e['metrics'], b['metrics']
            change = lambda field: 100 * (em[field] / bm[field] - 1)
            comparisons.append({'name': name, 'repetition': repetition,
                'experimental_run': e['run'], 'reference_run': b['run'],
                'light_delta_ms': em['light_delivery_p99_ms'] - bm['light_delivery_p99_ms'],
                'light_improvement_pct': -change('light_delivery_p99_ms'),
                'heavy_delta_ms': em['heavy_delivery_p99_ms'] - bm['heavy_delivery_p99_ms'],
                'heavy_change_pct': change('heavy_delivery_p99_ms'),
                'tokens_s_change_pct': change('tokens_s'), 'server_cpu_change_pct': change('server_cpu_s'),
                'pacing_comparable': not (e['pacing']['warning'] or b['pacing']['warning'])})
    historical = HERE.parent / 'formal_high/01_fixed1/result.json'
    historical_match = (read(historical)['semantics'] == baseline_semantics) if historical.is_file() else None
    summary = {'evidence_status': '探索性2x2动作诊断；不是独立论文确认', 'independent_unit': 'run',
        'trace_sha256': sha(args.trace), 'requests': len(trace['requests']),
        'tokens': sum(len(r['output_token_ids']) for r in trace['requests']),
        'original_interbatch_median_ms': interval_median, 'all_runs_retained': True,
        'execution_identity': execution_identity(args.execution_hashes, args.trace),
        'historical_semantic_hash_match': historical_match, 'runs': runs, 'comparisons': comparisons}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'endpoint_results.md').write_text(build_report(summary))
    print(json.dumps({'runs': len(runs), 'all_completed': all(r['metrics']['all_completed'] for r in runs),
        'pacing_warnings': [r['run'] for r in runs if r['pacing']['warning']],
        'historical_semantic_hash_match': historical_match,
        'execution_identity': summary['execution_identity']['status'], 'comparisons': comparisons}, ensure_ascii=False))


if __name__ == '__main__':
    main()

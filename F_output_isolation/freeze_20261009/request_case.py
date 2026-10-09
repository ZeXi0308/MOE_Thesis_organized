"""Freeze one existing high-load L/S case; offline stdlib, no new experiments.

Reuses materialization/analyze.py for the four selected runs. Writes only
request_case.json and request_case.md next to this script; preserves old data.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FROOT = HERE.parent
OLD = FROOT / "materialization"
PAIR_INDICES = ((5, 6), (8, 7))  # (S, L), original forward and reverse order
LABELS = {"lower_both": "两对均降低", "higher_both": "两对均升高",
          "mixed": "一降一升", "equal_both": "两对均相等", "with_tie": "含一次相等"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rel(path):
    return str(path.relative_to(FROOT))


def classify(deltas):
    if all(d < 0 for d in deltas):
        return "lower_both"
    if all(d > 0 for d in deltas):
        return "higher_both"
    if all(d == 0 for d in deltas):
        return "equal_both"
    return "mixed" if deltas[0] * deltas[1] < 0 else "with_tie"


def main():
    spec = importlib.util.spec_from_file_location("materialization_analysis", OLD / "analyze.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    old_summary = read(OLD / "summary.json")
    frozen = old_summary["loads"]["high"]
    assert sha(OLD / "analyze.py") == old_summary["analysis_sha256"]
    paths = [OLD / "analyze.py", OLD / "summary.json", OLD / "high" / "results.json",
             OLD / "high" / "command.json", FROOT / "inputs" / "high.json"]
    assert sha(paths[2]) == frozen["results_sha256"]
    assert sha(paths[3]) == frozen["command_sha256"]
    assert sha(paths[4]) == frozen["trace_sha256"] == analysis.SHAS["high"]
    trace, results = read(paths[4]), read(paths[2])
    mapping = analysis.mapping_for(trace)
    runs = {}
    for index in sorted({i for pair in PAIR_INDICES for i in pair}):
        expected = frozen["runs"][index]
        folder = OLD / "high" / expected["run"]
        for filename, expected_hash in expected["input_sha256"].items():
            path = folder / filename
            assert sha(path) == expected_hash, str(path)
            paths.append(path)
        run = analysis.analyze_run(index, results[index], folder, trace, mapping,
                                   results[0]["semantics"], results[0]["materialization"]["final_states"])
        # JSON stringifies the original integer chunk-histogram keys.
        assert json.loads(json.dumps(run)) == expected, (folder, "recomputed run differs from saved summary")
        runs[index] = run
    comparisons = [analysis.compare(runs[l], runs[s]) for s, l in PAIR_INDICES]
    old_comparisons = [c for c in frozen["comparisons"]
                       if c["candidate"] == "lazy_static" and c["baseline"] == "class_static"]
    assert comparisons == old_comparisons
    assert all(not c["source_pacing_warning"] for c in comparisons)
    requests = []
    for rid in sorted(comparisons[0]["requests"]):
        rows = [c["requests"][rid] for c in comparisons]
        assert all((r["kind"], r["group"], r["tokens"]) ==
                   (rows[0]["kind"], rows[0]["group"], rows[0]["tokens"]) for r in rows)
        pairs = [r["delivery_p99_ms"] for r in rows]
        requests.append({"request_id": rid,
                         **{k: rows[0][k] for k in ("kind", "group", "tokens")},
                         "pairs": pairs,
                         "classification": classify([p["delta"] for p in pairs])})
    assert len(requests) == 160
    cohorts = []
    for kind in ("light", "heavy"):
        for group in ("all", "background", "long"):
            members = [r for r in requests if r["kind"] == kind and
                       (group == "all" or r["group"] == group)]
            cohorts.append({"kind": kind, "group": group, "requests": len(members),
                            "tokens": sum(r["tokens"] for r in members),
                            "classification_counts": dict(Counter(r["classification"] for r in members)),
                            "classification_members": {label: [r["request_id"] for r in members
                                                                if r["classification"] == label]
                                                       for label in LABELS},
                            "pair_sign_counts": [dict(Counter("lower" if r["pairs"][i]["delta"] < 0
                                                              else "higher" if r["pairs"][i]["delta"] > 0
                                                              else "equal" for r in members))
                                                 for i in range(2)]})
    report = {
        "schema": 1,
        "scope": "Frozen historical materialization high L versus S; not direct encoding evidence",
        "analysis": "Post-hoc cross-pair classification of existing runs; no new experiment or tuned threshold",
        "unit": "ms",
        "metric": "Per-request token delivery P99, (client visible ns - producer emitted ns) / 1e6",
        "quantile": "Sort n values; linear interpolation at (n - 1) * 0.99 (existing analyze.q)",
        "classification": "Exact sign of L minus S in each pair; lower is metric benefit, not proven service utility",
        "independence": "Two run comparisons; requests and tokens within each run are dependent observations",
        "pairs": [{"pair": i, "baseline_run": c["baseline_run"], "candidate_run": c["candidate_run"],
                   "execution_order": [c["baseline_run"], c["candidate_run"]] if i == 0 else
                                      [c["candidate_run"], c["baseline_run"]],
                   "pooled_token_p99_ms": {kind: c["metrics"][f"{kind}_delivery_p99_ms"]
                                           for kind in ("light", "heavy")},
                   "source_pacing_warning": c["source_pacing_warning"]}
                  for i, c in enumerate(comparisons)],
        "cohorts": cohorts,
        "requests": requests,
        "input_sha256": {rel(p): sha(p) for p in paths},
        "analysis_source_sha256": sha(Path(__file__)),
        "verified": {"selected_raw_runs_recomputed_equal_existing_summary": True,
                     "full_semantic_hashes_and_completion_checked_by_existing_analyzer": True,
                     "all_160_requests_in_both_pairs": True,
                     "all_four_runs_source_pacing_passed": True},
    }
    (HERE / "request_case.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    lines = ["# 固定逐请求案例：旧物化 L 相对 S 的收益与损失", "",
        "本案例只复算已经保存的 high 开放环 CPU 回放：S=`class_static`，L=`lazy_static`。"
        "两者沿用相同释放规则（light 立即、heavy 4 token 或 75 ms；首 token 与完成立即），"
        "L 另延迟 heavy logprob 容器物化。这里的 75 ms 是历史实验参数，不是应用 SLO。"
        "**此案例属于旧物化路线，与直接编码分支的效果无关。**", "",
        "本轮未重跑服务、未修改候选/负载/指标。已有 `materialization/analyze.py` 已覆盖单对逐请求分析；"
        "新增脚本只复用它核验四次原始记录，并补齐跨两对受益/受损分类及完整清单。", "",
        "## 指标、配对与统计单位", "",
        "交付延迟为 `(客户端看到所属 SSE chunk 的时间 − producer 实际产出该 token 的时间) / 1e6`，"
        "单位 ms；一个 chunk 内全部 token 使用同一可见时间。这里的 engine 由 CPU 回放 producer 替代。"
        "总体 light/heavy P99 分别在该类全部 token 上计算；逐请求 P99 只在该请求 token 上计算。"
        "P99 使用原脚本的线性插值：排序后在 `(n−1)×0.99` 位置取值，既不是每请求 P99 的平均，"
        "也不是请求等权的整体指标。", "",
        "对 0 为 `06_lazy_static − 05_class_static`（执行 S→L）；对 1 为"
        " `07_lazy_static − 08_class_static`（执行 L→S）。相同请求 ID、输入内容与到达轨迹配对，"
        "Δ 均为 L−S。Δ<0 称指标受益，Δ>0 称指标受损，仅按原始数值符号分类，未设事后容差；"
        "不据此断言小差异超过噪声或实际用户效用改变。", "",
        "只有两对运行比较。160 个请求及其 token 共享各次运行的资源、定时器和环境，"
        "不是 160 个独立系统重复；本案例不计算伪独立置信区间或宣称统计显著。", "",
        "## 总体 P99 与完整群体分解", "",
        "| 对 | 类别 | S P99 ms | L P99 ms | Δ ms | Δ% |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for pair in report["pairs"]:
        for kind, metric in pair["pooled_token_p99_ms"].items():
            lines.append(f"| {pair['pair']} | {kind} | {metric['baseline']:.6f} | "
                         f"{metric['candidate']:.6f} | {metric['delta']:+.6f} | {metric['delta_pct']:+.3f}% |")
    lines += ["", "| 类别/输入组 | 请求/token 数 | 两对均受益 | 两对均受损 | 一降一升 | 含相等 | 对 0 降/升/平 | 对 1 降/升/平 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for c in cohorts:
        counts = c["classification_counts"]
        signs = ["/".join(str(d.get(k, 0)) for k in ("lower", "higher", "equal"))
                 for d in c["pair_sign_counts"]]
        lines.append(f"| {c['kind']}/{c['group']} | {c['requests']}/{c['tokens']} | "
                     f"{counts.get('lower_both', 0)} | {counts.get('higher_both', 0)} | "
                     f"{counts.get('mixed', 0)} | {counts.get('equal_both', 0) + counts.get('with_tie', 0)} | "
                     f"{signs[0]} | {signs[1]} |")
    lines += ["", "`all` 包含该类别的 `background` 与 `long`，各行不能相加。background/long 是既有输入组名，"
        "分别为 384/512 个输出 token，不代表请求优先级或允许等待时间。", "",
        "**总体轻流 P99 的两次改善伴随集中退化。** 120 个轻请求中，67 个两对均降低、50 个两对均升高、"
        "3 个方向不一致；40 个重请求中，0 个两对均降低、34 个两对均升高、6 个方向不一致。"
        "轻 background 的 18/24 个请求两对均受损，轻 long 的 32/96 个也两对均受损。"
        "轻 long 占轻类 token 的 84.21%，因此 pooled token P99 不为各请求提供同等代表性，"
        "更不保证少数或较短请求不受损。", "",
        "这是一项可观测的交付取舍：总体指标变好，同时部分群体和许多请求变差。"
        "它支持“不能仅凭总体 P99 认定隔离改善”的判断；两次运行不能证明一个请求的每毫秒收益"
        "恰由另一个请求承担，也不能排除共享主机噪声。没有独立交付需求，不能由轻/重类别推导"
        "这种等待调整合理。两对比较也没有证明 L 被所有简单方案支配。", "",
        "## 完整逐请求清单", "",
        "下表覆盖全部 160 个请求，无删选。表中数值仅显示到 6 位小数；"
        "[request_case.json](request_case.json) 保留两对 S/L 原值、原始浮点差值、完整分组成员及输入 SHA256。"
        "各组按请求 ID 排列；分类依据未四舍五入的差值。", ""]
    for kind in ("light", "heavy"):
        lines += [f"### {kind}", "", "| 请求 | 输入组 | token | S₀ | L₀ | Δ₀ ms | S₁ | L₁ | Δ₁ ms | 两对分类 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in requests:
            if r["kind"] != kind:
                continue
            a, b = r["pairs"]
            lines.append(f"| {r['request_id']} | {r['group']} | {r['tokens']} | "
                         f"{a['baseline']:.6f} | {a['candidate']:.6f} | {a['delta']:+.6f} | "
                         f"{b['baseline']:.6f} | {b['candidate']:.6f} | {b['delta']:+.6f} | "
                         f"{LABELS[r['classification']]} |")
        lines.append("")
    lines += ["## 复算与证据边界", "", "在工作区根目录执行（Python 标准库，仅离线）：", "", "```sh",
        "python3 F_output_isolation/freeze_20261009/request_case.py", "```", "",
        "脚本只改写本目录的 `request_case.md` 与 `request_case.json`。读取冻结 high 输入、"
        "四次运行的 `client/server/producer/result.json` 及现有汇总，核验 SHA256，"
        "通过原分析器重新检查完整 token 账目、时间因果顺序、完成/usage、语义摘要、最终状态、"
        "无 pending 积压和原 source pacing 标准，并断言重算值与已有 summary 完全一致。"
        "完整输入相对路径与 SHA256 见 JSON 的 `input_sha256`；代码身份见 `analysis_source_sha256`。", "",
        "四次记录都完整结束，没有为改善 P99 删减重请求。原始客户端正文已在历史摘要后释放，"
        "复算核验的是保存的内容哈希与账目，不能声称再次逐字核对正文。top-20 候选为合成数据，"
        "回放省略真实 engine IPC/GPU；源批内顺序与共享主机仍有限制。"
        "本案例既不提供直接编码分支性能证据，也不把旧 L 的代价转移为对新分支的负面判决。", ""]
    (HERE / "request_case.md").write_text("\n".join(lines))
    print(json.dumps({"requests": len(requests), "pairs": len(comparisons),
                      "outputs": ["request_case.md", "request_case.json"],
                      "all_four_raw_runs_match_saved_analysis": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()

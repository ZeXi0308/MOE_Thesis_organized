"""Summarize the six actual equal-pool cells, without fixed-KV eligibility rules."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "execution_partition_41307_r01/results_partition_r01"
report = json.loads((RESULTS / "partition_metrics.json").read_text())
cells = report["cells"]
groups = {cap: [n for n, c in cells.items() if c["config"]["expert_cap"] == cap]
          for cap in (24, 23, 20)}
fields = {
    "drained_s": lambda m: m["kv_service_cost"]["drained_s"],
    "token_rate": lambda m: m["drained_output_tokens_per_s"],
    "output_tokens": lambda m: m["output_tokens"],
    "flow_s": lambda m: m["flow"]["mean_s"],
    "ttft_s": lambda m: m["ttft"]["mean_s"],
    "max_gap_s": lambda m: m["inter_chunk_gap"]["max_s"],
    "expert_gb": lambda m: m["weight_copy_bytes"] / 1e9,
    "correct": lambda m: m["correct_requests"],
}
values = {n: {k: f(c["metrics"]) for k, f in fields.items()} for n, c in cells.items()}
means = {str(cap): {k: mean(values[n][k] for n in ns) for k in fields} for cap, ns in groups.items()}
pairs = []
for cap in (23, 20):
    for n, b in zip(groups[cap], groups[24]):
        pairs.append(dict(candidate=n, baseline=b,
                          percent={k: 100 * (values[n][k] / values[b][k] - 1) for k in fields}))
outputs = {}
for cap, names in groups.items():
    a, b = (cells[n]["metrics"]["per_request"] for n in names)
    outputs[str(cap)] = dict(identical_repeat_sequences=sum(x["output_token_ids"] == y["output_token_ids"] for x, y in zip(a, b)))
    reference = cells[groups[24][0]]["metrics"]["per_request"]
    outputs[str(cap)]["identical_to_24_sequences"] = sum(x["output_token_ids"] == y["output_token_ids"] for x, y in zip(a, reference))
    outputs[str(cap)]["quality_flips_vs_24"] = [dict(request=x["request_id"], score_24=y["score"], score_candidate=x["score"])
                                                for x, y in zip(a, reference) if x["score"] != y["score"]]
resources = {}
for n, c in cells.items():
    assert c["analysis_status"] == "COMPLETE" and c["metrics"]["finish_counts"] == {"stop": 16}
    p = json.loads((RESULTS / n / "partition_resources.json").read_text())
    assert p["equal_total_bytes"] and p["actual_total_bytes"] == 5905580032
    p["cuda_after_measurement"] = json.loads((RESULTS / n / "cuda_memory.json").read_text())["after_measurement"]
    resources[n] = p
summary = dict(evidence_type="ACTUAL_STATIC_EXPERT_KV_PARTITION", means=means, pairs=pairs,
               outputs=outputs, resources=resources, cells=values,
               archive_sha256="605ad41a5c29cfe5be6b40abdba865a567f7a87aa3fae631c06ea19ff1626d51")
(RESULTS / "partition_r01_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
lines = ["# 等专家＋KV总池静态划分：六格实测", "",
    "结论：expert23/KV1.1875GiB 相对 expert24/KV1GiB，两次完整时间分别降低 %.2f%%/%.2f%%，输出率提高 %.2f%%/%.2f%%。均值完整时间降低 %.2f%%，输出率提高 %.2f%%，但输出量增加 %.2f%%。expert20/KV1.75GiB 消除抢占、缩短首输出等待，却使完整服务变慢。静态曲线存在可测的容量取舍，不能将静态划分本身称为新方法。" % (
        -pairs[0]["percent"]["drained_s"], -pairs[1]["percent"]["drained_s"], pairs[0]["percent"]["token_rate"], pairs[1]["percent"]["token_rate"],
        100*(1-means['23']['drained_s']/means['24']['drained_s']), 100*(means['23']['token_rate']/means['24']['token_rate']-1), 100*(means['23']['output_tokens']/means['24']['output_tokens']-1)), "",
    "全部6格COMPLETE，各16请求自然EOS、无长度截断。顺序24/23/20/20/23/24，不丢弃任何点；同臂两次输出序列各16/16一致。独立engine、相同Q2048、cap16、CPU KV1GiB、原revision/32–47开发输入与原warmup；详见 PARTITION_EXPERIMENT.md。n=2仅描述性重复，不提供置信区间。", "",
    "| 格 | drained s | token/s | 输出 | mean flow s | mean TTFT s | max host gap s | 正确 | 专家GB |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for n, v in values.items():
    lines.append(f"| {n} | {v['drained_s']:.4f} | {v['token_rate']:.4f} | {v['output_tokens']} | {v['flow_s']:.4f} | {v['ttft_s']:.4f} | {v['max_gap_s']:.4f} | {v['correct']}/16 | {v['expert_gb']:.4f} |")
lines += ["", "| 容量 | 平均drained s | token/s | flow s | TTFT s | max gap s |", "|---|---:|---:|---:|---:|---:|"]
for cap, v in means.items():
    lines.append(f"| {cap} | {v['drained_s']:.4f} | {v['token_rate']:.4f} | {v['flow_s']:.4f} | {v['ttft_s']:.4f} | {v['max_gap_s']:.4f} |")
lines += ["", "完整成本包含所有prefill、mixed、decode、CPU KV恢复和最终drain；CUDA/host子时间不再相加。这里为AR单token receipt，host间隔可分辨，但仍是host可见输出节律。", "",
    "| 容量 | 抢占次数/不同请求 | KV load MiB | store MiB | mixed步/秒 | decode步/秒 | 重复调度位置 | 专家group |",
    "|---|---:|---:|---:|---:|---:|---:|---:|"]
for cap, names in groups.items():
    m = cells[names[0]]["metrics"]; s = m['stages']['by_kind']; pre=m['preemption']; tr=m['completed_transfer_bytes']
    mixed=mean(cells[n]['metrics']['stages']['by_kind']['mixed']['total_engine_wall_s'] for n in names)
    dec=mean(cells[n]['metrics']['stages']['by_kind']['pure_decode']['total_engine_wall_s'] for n in names)
    lines.append(f"| {cap} | {pre['total_preemption_events']}/{pre['distinct_preempted_requests']} | {tr['load']/2**20:g} | {tr['store']/2**20:g} | {s['mixed']['engine_calls']}/{mixed:.4f} | {s['pure_decode']['engine_calls']}/{dec:.4f} | {m['scheduled_position_overlap']['repeated_scheduled_positions']} | {m['group_count']} |")
lines += ["", "24→23少18次engine调用、205个expert group，但专家字节增加1.56%；24→20少mixed执行却增加decode轮次和总group。更大的KV允许不同批次与请求推进，不能用同一未来轨迹或只用miss字节解释完整时间；抢占为零也不保证吞吐更高。", "",
    "质量：23相对24仅8/16完整序列相同，两者同为8/16正确且同题正确；20相对24仅7/16序列相同，9/16正确。20新增正确题与全部quality flips记录于partition_r01_summary.json。输出变化可能来自不同batch/专家分组的有限精度执行，尚无端到端位等价证明；不把相同准确率称为等价，也不以输出变少解释加速。", "",
    "显存：六格真实底层storage去重的专家scratch＋GPU KV均为5,905,580,032B（5.5GiB）；24/23/20分别为512/608/896个KV块。六格measurement peak allocated均7,044,517,376B，peak reserved均7,501,512,704B。5.5GiB仅两工作池，未包含模型其余张量/workspace/allocator；本卡可全驻留OLMoE，因此仍是人工容量压力。", "",
    "后续裁决已改变：直接查新发现VAMP已覆盖分配失败→成本比较→专家转KV的中心动作，见C_PRIOR_ART.md。停止普通grow-only诊断/控制器开发，live_grow_src仅保留未执行草稿。当前静态均值时间差仅%.4fs，也未证明能支付快照、barrier、重建和专家再热。继续原静态六格的48–63独立输入确认，不调整配置、不称新方法。" % (means['24']['drained_s']-means['23']['drained_s']), "",
    "归档：partition_41307_r01.tar.gz，SHA256 `"+summary['archive_sha256']+"`。原始目录 `execution_partition_41307_r01/results_partition_r01`；逐请求及阶段原始分析 `partition_metrics.json`，简表与配对 `partition_r01_summary.json`。", ""]
(ROOT / "PARTITION_RESULTS_R01.md").write_text("\n".join(lines))
print(json.dumps(dict(means=means, pairs=pairs, outputs=outputs), ensure_ascii=False))

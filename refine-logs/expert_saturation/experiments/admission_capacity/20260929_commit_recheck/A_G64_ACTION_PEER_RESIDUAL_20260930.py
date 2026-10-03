#!/usr/bin/env python3
"""Reproduce the action and peer-service residual of the frozen G64 LTR cell.

This is a diagnostic, not a performance contrast.  The large raw JSON is read
once to verify its hash, then only its request records are streamed.  The
published V4 lifecycle audit supplies the strict native store/load chain check.
The script never writes into the frozen archive and refuses to overwrite output.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "moe-a-ltr-g64-session-system-copy-r02-20260930/cell-00-ltr_r02_g64/archive"
AUDIT = HERE / "A_G64_LTR_LIFECYCLE_AUDIT_SYSTEM_COPY_R02_20260930_V4.json"
AUDIT_SHA256 = "4f528cc121064898e1cc071d10a6e5eb1ea568c799cc6b7911bd15903603d646"
OUTPUT = HERE / "A_G64_ACTION_PEER_RESIDUAL_20260930"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def distribution(values: list[float | int]) -> dict[str, float | int]:
    require(bool(values) and all(finite(x) for x in values), "empty/nonfinite distribution")
    ordered = sorted(values)

    def percentile(q: float) -> float:
        index = (len(ordered) - 1) * q
        lo = math.floor(index)
        hi = math.ceil(index)
        return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))

    return dict(n=len(ordered), min=float(ordered[0]), p50=percentile(.5),
                p95=percentile(.95), max=float(ordered[-1]))


def read_request_prefix(raw_path: Path) -> tuple[float, dict[str, dict[str, Any]]]:
    """Read the 64 request objects before raw.json's very large output log.

    This parser intentionally accepts only the verified frozen pretty-print
    layout.  A changed layout or incomplete object fails instead of guessing.
    """
    origin: float | None = None
    rows: list[dict[str, Any]] = []
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith('  "measurement_origin_perf_counter_s": '):
                require(origin is None, "duplicate measurement origin")
                origin = json.loads(line.split(": ", 1)[1].rstrip().rstrip(","))
            if line == '  "requests": [\n':
                break
        else:
            raise ValueError("raw request array is absent")
        require(finite(origin), "invalid measurement clock origin")
        current: list[str] = []
        for line in stream:
            if not current and line == '  ],\n':
                break
            if not current:
                require(line == '    {\n', "unexpected raw request opening")
            current.append(line)
            require(sum(map(len, current)) < 2_000_000, "raw request object too large")
            if line in ('    },\n', '    }\n'):
                record = json.loads("".join(current).strip().rstrip(","))
                require(isinstance(record, dict), "raw request is not an object")
                rows.append(record)
                current = []
        else:
            raise ValueError("raw request array is unterminated")
        require(not current and len(rows) == 64, "raw request count/layout differs")

    requests: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    arrivals: list[float] = []
    for row in rows:
        internal, source = row.get("internal_request_id"), row.get("request_id")
        times = row.get("token_times_s")
        require(isinstance(internal, str) and internal.startswith("measured/")
                and isinstance(source, str) and internal.split("/", 1)[1].rsplit("-", 1)[0] == source,
                "internal/source request identity differs")
        require(internal not in requests and source not in source_ids, "duplicate request identity")
        require(isinstance(times, list) and bool(times) and all(finite(x) for x in times)
                and all(left < right for left, right in zip(times, times[1:])),
                f"missing/unordered returned token times: {source}")
        require(isinstance(row.get("output_token_ids"), list)
                and len(row["output_token_ids"]) == len(times),
                f"output IDs and token times differ: {source}")
        require(finite(row.get("arrival_s")) and finite(row.get("completion_s"))
                and row["completion_s"] >= times[-1] >= row["arrival_s"],
                f"request clock invalid: {source}")
        require(row.get("stop_reason") in ("stop", "length"),
                f"request stop reason invalid: {source}")
        requests[internal] = dict(source=source, arrival_s=row["arrival_s"],
                                  completion_s=row["completion_s"], times=times,
                                  stop_reason=row["stop_reason"])
        source_ids.add(source)
        arrivals.append(row["arrival_s"])
    require(arrivals == [i * .2 for i in range(64)],
            "frozen 0.2-second source-order arrivals differ")
    return float(origin), requests


def analyze(archive: Path, audit_path: Path) -> dict[str, Any]:
    archive, audit_path = archive.resolve(), audit_path.resolve()
    require(archive == ARCHIVE.resolve() and audit_path == AUDIT.resolve(),
            "only the frozen G64 archive and V4 audit are accepted")
    require(digest(audit_path) == AUDIT_SHA256, "V4 lifecycle audit SHA-256 differs")
    audit = json.loads(audit_path.read_text())
    require(audit.get("verdict") == "OBSERVED_CHAIN" and audit.get("issues") == []
            and audit.get("result_dir") == str(archive)
            and all(audit.get("checks", {}).get(k) is True for k in
                    ("identity", "physical_resources", "runtime_source")),
            "V4 lifecycle audit does not certify this result directory")
    required_files = ("raw.json", "selective-store.json", "offload-events.json")
    hashes = {}
    for name in required_files:
        path = archive / name
        expected = audit.get("files", {}).get(name, {})
        require(path.is_file() and path.stat().st_size == expected.get("bytes"),
                f"archive file size differs: {name}")
        hashes[name] = digest(path)
        require(hashes[name] == expected.get("sha256"), f"archive file hash differs: {name}")

    origin, requests = read_request_prefix(archive / "raw.json")
    selective = json.loads((archive / "selective-store.json").read_text())
    offload = json.loads((archive / "offload-events.json").read_text())
    events = selective.get("events")
    require(isinstance(events, list), "selective events absent")
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        require(isinstance(row, dict) and isinstance(row.get("event"), str)
                and type(row.get("step")) is int and finite(row.get("host_perf_counter_s")),
                "malformed policy event")
        by_kind[row["event"]].append(row)
    accepts = sorted(by_kind["accept"], key=lambda x: x["step"])
    prepares = {x["step"]: x for x in by_kind["prepare"]}
    commits = {x["step"]: x for x in by_kind["commit_check"] if x.get("reason") == "READY"}
    releases = sorted(by_kind["release"], key=lambda x: x["step"])
    markers = by_kind["target_new_output"]
    allocations = {x["step"]: x for x in by_kind["allocation"]}
    metadata = by_kind["metadata"]
    require(len(accepts) == 69 and len(prepares) == len(commits) == 53
            and len(releases) == 69 and len(markers) == 69
            and len(allocations) == 2707, "frozen policy event counts differ")
    require(len({x["step"] for x in accepts}) == 69
            and len({x["step"] for x in releases}) == 69
            and len(allocations) == len(by_kind["allocation"]),
            "duplicate action, release, or allocation step")
    require(Counter(x.get("action") for x in accepts)
            == {"PREPARE_SELECTED": 53, "PRIORITIZE_WAITING": 16},
            "frozen accepted action mix differs")
    require(Counter(x.get("reason") for x in releases)
            == {"quantum_expired": 67, "terminal": 2},
            "frozen release mix differs")
    chains = audit.get("observed_chains")
    require(isinstance(chains, list) and len(chains) == 42,
            "V4 strict lifecycle chain count differs")
    joined_steps = {x.get("prepare_step") for x in chains}
    require(len(joined_steps) == 42 and joined_steps <= prepares.keys(),
            "V4 chain identities do not match the action log")

    epochs: list[dict[str, Any]] = []
    paired_release_steps: set[int] = set()
    last_arrival = max(row["arrival_s"] for row in requests.values())
    for index, action in enumerate(accepts):
        step = action["step"]
        next_step = accepts[index + 1]["step"] if index + 1 < len(accepts) else len(allocations) + 1
        target = action.get("target_id")
        require(target in requests and step < next_step, "accepted target/step identity differs")
        candidate_releases = [x for x in releases if step < x["step"] <= next_step]
        require(len(candidate_releases) == 1 and candidate_releases[0].get("target") == target,
                f"action has no unique matching release: step {step}")
        release = candidate_releases[0]
        paired_release_steps.add(release["step"])
        candidate_markers = [x for x in markers if step <= x["step"] < release["step"]
                             and x.get("target") == target]
        require(len(candidate_markers) == 1, f"action has no unique new-output marker: step {step}")
        marker = candidate_markers[0]
        times = requests[target]["times"]
        start_s = action["host_perf_counter_s"] - origin
        end_s = release["host_perf_counter_s"] - origin
        first_output_index = bisect.bisect_right(times, start_s)
        last_output_index = bisect.bisect_right(times, end_s)
        require(0 <= first_output_index < last_output_index <= len(times),
                f"no returned output during accepted epoch: step {step}")
        scheduled = []
        for turn in range(step, release["step"]):
            row = allocations.get(turn)
            require(row is not None and isinstance(row.get("scheduled"), dict),
                    f"missing native allocation: step {turn}")
            amount = row["scheduled"].get(target, 0)
            require(type(amount) is int and amount >= 0, "invalid target scheduled tokens")
            scheduled.append(amount)
        positive_turns = sum(amount > 0 for amount in scheduled)
        result: dict[str, Any] = dict(
            accept_step=step, release_step=release["step"], action=action["action"],
            target=requests[target]["source"], release_reason=release["reason"],
            accepted_at_s=start_s, released_at_s=end_s,
            accepted_to_first_returned_output_s=times[first_output_index] - start_s,
            returned_outputs_during_epoch=last_output_index - first_output_index,
            positive_scheduled_turns=positive_turns,
            scheduled_tokens_during_epoch=sum(scheduled),
        )
        if action["action"] == "PREPARE_SELECTED":
            prepare, commit = prepares.get(step), commits.get(step + 1)
            victim = action.get("victim_id")
            require(prepare is not None and commit is not None and victim in requests
                    and prepare.get("victim") == commit.get("victim") == victim
                    and prepare.get("target") == commit.get("target") == target,
                    f"selected prepare/READY victim or target mismatch: step {step}")
            require(prepare.get("slot_required") is False
                    and prepare.get("kv_funding_required") is True,
                    f"selected action does not show KV-only deficit: step {step}")
            require(positive_turns == 10 and 7 <= result["returned_outputs_during_epoch"] <= 10,
                    f"selected service quantum/output count differs: step {step}")
            native_target_loads = [job.get("job_id") for meta in metadata
                                   if step < meta["step"] < release["step"]
                                   for job in meta.get("jobs", [])
                                   if isinstance(job, dict) and job.get("request") == target
                                   and job.get("is_store") is False
                                   and job.get("job_id") in meta.get("loads", [])]
            require((step in joined_steps) == (len(native_target_loads) == 1),
                    f"V4 chain versus observed target load differs: step {step}")
            victim_times = requests[victim]["times"]
            commit_s = commit["host_perf_counter_s"] - origin
            next_victim_index = bisect.bisect_right(victim_times, commit_s)
            require(0 < next_victim_index < len(victim_times),
                    f"cannot bracket victim returned-output gap: step {step}")
            next_victim_output_s = victim_times[next_victim_index]
            require(next_victim_output_s > end_s,
                    f"victim output preceded target release: step {step}")
            result.update(
                victim=requests[victim]["source"], commit_step=commit["step"],
                committed_at_s=commit_s,
                strict_load_joined=step in joined_steps,
                native_target_load_jobs=native_target_loads,
                selected_victim_saved_tokens=prepare.get("saved_tokens"),
                victim_commit_to_next_output_s=next_victim_output_s - commit_s,
                victim_release_to_next_output_s=next_victim_output_s - end_s,
                victim_returned_output_gap_s=next_victim_output_s - victim_times[next_victim_index - 1],
            )
        else:
            require(step not in prepares and positive_turns in (2, 9, 10),
                    f"direct priority epoch/quantum differs: step {step}")
        epochs.append(result)
    require(len(paired_release_steps) == len(releases), "unpaired policy release")

    forced = [x for x in epochs if x["action"] == "PREPARE_SELECTED"]
    direct = [x for x in epochs if x["action"] == "PRIORITIZE_WAITING"]
    joined = [x for x in forced if x["strict_load_joined"]]
    no_load = [x for x in forced if not x["strict_load_joined"]]
    require(len(joined) == 42 and len(no_load) == 11, "load-joined/fallback split differs")
    no_load_targets = Counter(x["target"] for x in no_load)
    require(all(x["target"] not in {y["victim"] for y in forced} for x in no_load),
            "a no-target-load request was also a selected-save victim")
    eos = [x for x in requests.values() if x["stop_reason"] == "stop"]
    length = [x for x in requests.values() if x["stop_reason"] == "length"]
    require(len(eos) == 5 and len(length) == 59, "frozen EOS/length stop counts differ")
    require(max(x["completion_s"] for x in eos) < min(x["accepted_at_s"] for x in epochs),
            "natural EOS did not all precede the first recovery action")
    terminal = [x for x in epochs if x["release_reason"] == "terminal"]
    require(len(terminal) == 2 and all(requests[next(k for k, v in requests.items()
            if v["source"] == x["target"])]["stop_reason"] == "length" for x in terminal),
            "terminal release has an unqualified natural EOS")
    transfers = offload.get("transfers")
    require(isinstance(transfers, list) and len(transfers) == 106,
            "offload aggregate transfer count differs")
    transfer_bytes = {}
    for kind in ("store", "load"):
        values = [x.get(kind, {}).get("bytes") for x in transfers if isinstance(x, dict)]
        require(len(values) == 106 and all(type(v) is int and v >= 0 for v in values),
                f"offload {kind} bytes malformed")
        transfer_bytes[kind] = sum(values)
    counts = dict(
        requests=len(requests), accepted_epochs=len(epochs),
        forced_selected_epochs=len(forced), direct_priority_epochs=len(direct),
        selected_kv_only_deficits=len(forced), distinct_targets=len({x["target"] for x in epochs}),
        distinct_selected_victims=len({x["victim"] for x in forced}),
        strict_store_load_output_service_chains=len(joined),
        selected_epochs_without_target_load=len(no_load),
        selected_no_load_distinct_targets=len(no_load_targets),
        accepted_before_last_arrival=sum(x["accepted_at_s"] < last_arrival for x in epochs),
        selected_before_last_arrival=sum(x["accepted_at_s"] < last_arrival for x in forced),
        quantum_expirations=sum(x["release_reason"] == "quantum_expired" for x in epochs),
        terminal_length_releases=len(terminal), natural_eos=len(eos), length_stops=len(length),
    )
    require(all(counts[k] == audit["counts"][v] for k, v in (
            ("accepted_epochs", "accepted_intents"),
            ("forced_selected_epochs", "prepared"),
            ("strict_store_load_output_service_chains", "observed_chains"),
            ("natural_eos", "natural_eos"), ("length_stops", "length_stops"))),
            "action/result counts disagree with V4")
    return dict(
        schema_version=1, verdict="OBSERVED_ACTION_PEER_RESIDUAL_DIAGNOSTIC",
        evidence_ceiling="NATIVE_SERVING_INPROCESS_DIAGNOSTIC_NO_PERFORMANCE_CONTRAST",
        source=dict(archive=str(archive), lifecycle_audit=str(audit_path),
                    lifecycle_audit_sha256=AUDIT_SHA256,
                    analyzer_sha256=digest(Path(__file__)), archive_file_sha256=hashes),
        counts=counts,
        time_boundaries_s=dict(last_arrival=last_arrival,
                               first_accepted=min(x["accepted_at_s"] for x in epochs),
                               last_accepted=max(x["accepted_at_s"] for x in epochs),
                               last_natural_eos=max(x["completion_s"] for x in eos)),
        transfer_bytes_aggregate=transfer_bytes,
        distributions=dict(
            selected_victim_commit_to_next_returned_output_s=distribution(
                [x["victim_commit_to_next_output_s"] for x in forced]),
            selected_victim_release_to_next_returned_output_s=distribution(
                [x["victim_release_to_next_output_s"] for x in forced]),
            selected_target_accept_to_next_returned_output_s=distribution(
                [x["accepted_to_first_returned_output_s"] for x in forced]),
            selected_returned_outputs_during_epoch=distribution(
                [x["returned_outputs_during_epoch"] for x in forced]),
            joined_scheduled_tokens_during_epoch=distribution(
                [x["scheduled_tokens_during_epoch"] for x in joined]),
            no_target_load_scheduled_tokens_during_epoch=distribution(
                [x["scheduled_tokens_during_epoch"] for x in no_load]),
        ),
        no_target_load_target_epoch_counts=dict(sorted(no_load_targets.items())),
        interpretation=[
            "All 53 selected actions were KV-funding deficits with a free sequence slot; each preempted one running peer.",
            "The chosen peer's returned-output wait is action-associated, not an incremental delay versus a counterfactual scheduler.",
            "Eleven selected epochs have no native load of their target before release. Their large scheduled-token counts are consistent with recomputation, but exact recomputation work and FLOPs are not independently measured.",
            "The ten-turn quantum counts positive native scheduling turns, not equal compute or returned-output cost.",
        ],
        limits=[
            "This diagnostic capture has observer overhead and is on a different GPU from older G current/eager/full timing data; do not calculate performance effects across them.",
            "The five natural EOS requests finished before the first recovery action; recovery-phase natural EOS remains unobserved.",
            "The V4 audit establishes 42 strict physical store/load/output/service chains, not tensor-value equality, production service, or a novel mechanism claim.",
            "Aggregate transfer bytes do not identify bytes for each native job or recovery epoch.",
        ],
        epochs=epochs,
    )


def report_markdown(result: dict[str, Any], json_name: str) -> str:
    c, d, t = result["counts"], result["distributions"], result["time_boundaries_s"]
    p = d["selected_victim_commit_to_next_returned_output_s"]
    after = d["selected_victim_release_to_next_returned_output_s"]
    joined = d["joined_scheduled_tokens_during_epoch"]
    missed = d["no_target_load_scheduled_tokens_during_epoch"]
    return f"""# G64 LTR 动作与 peer 服务残差（只读复算）

**证据层级：原生进程内单格诊断；无性能对照。** [逐 episode 数据]({json_name}) 基于冻结 G64/T30/Q10 归档和 V4 `OBSERVED_CHAIN` 生命周期审计，脚本拒绝身份不符或覆盖旧输出。

- 64/64 请求完成；{c['accepted_epochs']} 次恢复动作涉及 {c['distinct_targets']} 个目标。{c['forced_selected_epochs']} 次选中保存均为 KV 空间不足、序列槽仍可用，抢占 {c['distinct_selected_victims']} 个不同运行中请求；{c['direct_priority_epochs']} 次直接优先恢复无需抢占。
- {c['strict_store_load_output_service_chains']} 次严格串起选中保存、目标原生加载、新输出及后续正调度。另 {c['selected_epochs_without_target_load']} 次选中动作在活动期没有目标加载，涉及 {c['selected_no_load_distinct_targets']} 个目标。前者十次正调度累计 token 中位 {joined['p50']:.0f}（范围 {joined['min']:.0f}–{joined['max']:.0f}），后者中位 {missed['p50']:.0f}（范围 {missed['min']:.0f}–{missed['max']:.0f}）；后者符合重算路径，但调度 token 不是 FLOPs 或独立重算计量。
- {c['forced_selected_epochs']} 次被选 victim 在 READY commit 至下一返回输出等待中位 {p['p50']:.3f} s（范围 {p['min']:.3f}–{p['max']:.3f} s），在目标量子释放后仍等待中位 {after['p50']:.3f} s。它是本运行中的动作关联间隔，**不能**当作相对其他调度器的增量损失。
- {c['quantum_expirations']} 次量子耗尽、{c['terminal_length_releases']} 次长度上限终止。所有 {c['forced_selected_epochs']} 次选中恢复活动期返回 7–10 个输出；没有 0–2 输出的选中恢复短段。{c['accepted_before_last_arrival']} 次动作发生在最后到达 {t['last_arrival']:.1f} s 之前。
- {c['natural_eos']} 个自然 EOS 最晚 {t['last_natural_eos']:.3f} s 完成，早于首个动作 {t['first_accepted']:.3f} s；恢复中的自然 EOS 未覆盖。

**下一道实证门：** 先在同一 GPU、G64 同输入和轻量采集下比较 LTR、eager、native full，校准后才把固定点迁移到 H128；共同报告目标停顿、被抢占 peer 间隔、完成时间与吞吐。旧 G 轻量格与本诊断跨 GPU/观测负担，不能算性能差。selected 范围的 11 次无目标加载也可能被 native full 消除，故现阶段不提出新方法收益主张。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--out-prefix", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prefix = args.out_prefix.resolve()
    require(not prefix.is_relative_to(args.archive.resolve()), "output cannot be inside archive")
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    require(not json_path.exists() and not markdown_path.exists(), "output already exists")
    result = analyze(args.archive, args.audit)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(report_markdown(result, json_path.name))
    print(json.dumps(dict(verdict=result["verdict"], counts=result["counts"],
                          json=str(json_path), report=str(markdown_path)), ensure_ascii=False))


if __name__ == "__main__":
    main()

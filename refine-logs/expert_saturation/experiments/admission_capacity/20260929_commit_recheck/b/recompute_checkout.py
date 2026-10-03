#!/usr/bin/env python3
"""Read-only arithmetic and evidence-availability check for commit recheck.

The D/E numbers come from the accepted ledger. An optional isolated checkout
cross-checks the old derived CPU REPORT/JSON and actual G diagnostic snapshots;
it does not execute H1 GPU actions. G light counts come from RESULTS.md and H
counts from report_tables.json. Git blob checks disable lazy fetch, so this
command never uses the network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
REL = Path("refine-logs/expert_saturation")
OUT = REL / "outputs/admission_capacity"
LEDGER = ROOT / REL / "experiments/admission_capacity/RESULT_LEDGER.md"
G = OUT / "20260915_natural_recovery_cadence_r01"
H = OUT / "20260915_natural_cadence_holdout_r02"
A = OUT / "20260915_natural_native_full_gate_r01"


def must_match(pattern: str, text: str, source: Path) -> tuple[str, ...]:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Expected published field missing or changed: {source}: {pattern}")
    return match.groups()


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    return subprocess.run(
        ["git", *args], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def blob_state(relative_path: Path) -> str:
    path = relative_path.as_posix()
    tree = run_git("ls-tree", "HEAD", "--", path)
    if tree.returncode:
        raise RuntimeError(tree.stderr)
    if not tree.stdout.strip():
        return "NOT_IN_HEAD"
    rows = tree.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError(f"Expected one Git blob for {path}, found {len(rows)}")
    fields = rows[0].split("\t", 1)[0].split()
    if len(fields) != 3 or fields[1] != "blob":
        raise ValueError(f"Expected blob for {path}: {rows[0]}")
    local = run_git("cat-file", "-e", fields[2]).returncode == 0
    materialized = (ROOT / relative_path).is_file()
    if materialized:
        return "WORKTREE_PRESENT"
    return "LOCAL_BLOB_ONLY" if local else "TRACKED_BLOB_MISSING_LOCAL"


def historical_recheck(checkout: Path, de: dict) -> dict:
    """Cross-check the isolated, historical CPU outputs without raw replay."""
    folder = checkout / A / "commit_recheck"
    report = (folder / "REPORT.md").read_text()
    if "26/22个running" not in report:
        raise ValueError("Historical report does not state D/E running counts")
    result = {}
    for name, event, expected_running, protocol in (
        ("selected", "D859", 26, OUT / "20260915_natural_saved_recovery_gate_r01/README.md"),
        ("full", "E1399", 22, OUT / "20260915_natural_native_full_gate_r01/README.md"),
    ):
        data = json.loads((folder / f"{name}.json").read_text())
        if data["checks"] != 25 or data["changed"] != 1 or len(data["rows"]) != 25:
            raise ValueError(f"Unexpected historical commit coverage: {name}")
        changed = [row for row in data["rows"] if row.get("decision") == "RESUME_WITHOUT_VICTIM"]
        if len(changed) != 1 or changed[0]["step"] != int(event[1:]):
            raise ValueError(f"Unexpected historical action: {name}")
        row = changed[0]
        cap = int(must_match(r"Running cap(\d+)", (ROOT / protocol).read_text(), protocol)[0])
        expected = de[event]
        if (row["original_reason"] != "READY"
            or row["free"] != expected["free"]
            or row["target_remaining_blocks"] != expected["need"]
            or row["retained_victim_blocks"] != expected["old_victim_blocks"]
            or row["margin_after_current_peer_decode"] != expected["reported_one_peer_step_margin"]
            or row["running_count"] != expected_running
            or not row["all_running_pure_decode"]
            or row["pending_load_count"] != 0
            or not row["conditional_all_peers_and_target_fit"]
            or row["running_count"] >= cap):
            raise ValueError(f"Historical CPU row disagrees with ledger/protocol: {name}")
        result[event] = {
            "historical_cpu_checks": data["checks"],
            "historical_cpu_changed": data["changed"],
            "running_count": row["running_count"],
            "running_cap_from_protocol": cap,
            "sequence_slots_free_at_snapshot": cap-row["running_count"],
            "all_running_pure_decode": row["all_running_pure_decode"],
            "current_peer_one_step_growth": row["current_peer_one_step_growth"],
            "pending_load_count": row["pending_load_count"],
            "conditional_all_peers_and_target_fit": row["conditional_all_peers_and_target_fit"],
            "layer": "HISTORICAL_DERIVED_CPU_SNAPSHOT_NOT_RAW_OR_GPU_EXECUTION",
        }
    cost_path = checkout / A / "commit_recheck_cost/analysis.json"
    cost = json.loads(cost_path.read_text())
    if cost["status"] != "FACTUAL_COST_CONTEXT_NOT_ACTION_VALUE" or len(cost["rows"]) != 2:
        raise ValueError("Unexpected historical cost analysis scope")
    for row in cost["rows"]:
        event = f"{'D' if row['arm'] == 'selected' else 'E'}{row['commit_step']}"
        if event not in result:
            raise ValueError(f"Unexpected historical cost event: {event}")
        expected = de[event]
        if (row["candidate_commits_evaluated"] != 25
            or row["free_blocks"] != expected["free"]
            or row["target_required_blocks"] != expected["need"]
            or row["retained_victim_blocks"] != expected["old_victim_blocks"]
            or row["victim_confirmed_recompute_tokens"] != expected["victim_next_recomputed_positions"]
            or round(row["victim_reconstructed_load_bytes"] / 2**20) != expected["victim_next_load_mib"]
            or abs(row["victim_factual_output_gap_s"]-expected["victim_next_gap_s"]) > 1e-6
            or row["victim_event_is_largest"]
            or row["counterfactual_output_gap_s"] is not None
            or row["net_load_bytes_saved"] is not None):
            raise ValueError(f"Historical cost row disagrees with ledger/causal boundary: {event}")
        result[event]["factual_victim_cost_checked"] = True
    return result


def g_diagnostic_recheck(checkout: Path, published_light_counts: dict) -> dict:
    """Count H1's necessary KV condition at actual G READY commits."""
    folder = checkout / G / "execution_weste_26862/readback"
    adapter = (folder / "pkg/staged_store_rotation.py").read_text()
    contract = (folder / "pkg/staged_save_contract.py").read_text()
    if ("if block_size!=16" not in adapter
        or "ceil((self.prompt + self.output) / 16) - len(self.blocks)" not in contract
        or "data['eligibility_snapshots'].append(_eligibility_snapshot(" not in adapter
        or "pool.get_num_free_blocks(),step,cohort,plan,protected" not in adapter
        or "reason=commit_reason(plan,step,view(victim),view(target),pool.get_num_free_blocks(),None,save_enabled=False)" not in adapter):
        raise ValueError("G package block-size/need/snapshot source differs")
    snapshot_call = adapter.index("data['eligibility_snapshots'].append(_eligibility_snapshot(")
    commit_call = adapter.index("reason=commit_reason(plan,step,view(victim),view(target)", snapshot_call)
    between = adapter[snapshot_call:commit_call]
    if "_preempt_request" in between or "allocate_slots" in between:
        raise ValueError("G snapshot and commit check have intervening pool mutation")
    raw = json.loads((folder / "results/diagnostic-eager/selective-store.json").read_text())
    if (raw["status"] != "DRAINED" or raw["store_scope"] != "selected"
        or not raw["diagnostic"] or raw["population_mode"] != "open"
        or raw["eligibility_logging"] != "ENABLED"):
        raise ValueError("Unexpected G diagnostic identity")
    snapshots = raw["eligibility_snapshots"]
    by_step = {row["step"]: row for row in snapshots}
    if (len(snapshots) != raw["schedule_calls"] or len(by_step) != len(snapshots)
        or set(by_step) != set(range(raw["schedule_calls"]))):
        raise ValueError("G snapshots do not align one-to-one with schedule steps")
    events = raw["events"]
    prepares = {row["step"]: row for row in events if row["event"] == "prepare"}
    commits = [row for row in events if row["event"] == "commit_check"]
    if (len(prepares) != 54 or len(commits) != 54
        or raw["applied_rotations"] != len(commits)):
        raise ValueError("G prepare/commit/applied counts differ")
    deficits = []
    free_values = []
    need_values = []
    running_values = []
    target_held_values = []
    directly_funded = 0
    for event in commits:
        step = event["step"]
        snapshot = by_step[step]
        prepare = prepares.get(step-1)
        target_id = event["target"]
        victim_id = event["victim"]
        if (event["reason"] != "READY" or prepare is None
            or prepare["target"] != target_id or prepare["victim"] != victim_id
            or snapshot["plan_target"] != target_id or snapshot["plan_victim"] != victim_id
            or target_id not in snapshot["waiting_ids"]
            or target_id in snapshot["running_ids"] or target_id in snapshot["skipped_ids"]
            or victim_id not in snapshot["running_ids"] or snapshot["skipped_ids"]):
            raise ValueError(f"G commit identity/queue mismatch at step {step}")
        target = snapshot["requests"][target_id]
        victim = snapshot["requests"][victim_id]
        if target["status"] != "PREEMPTED" or victim["status"] != "RUNNING":
            raise ValueError(f"G commit phase mismatch at step {step}")
        free = snapshot["free_blocks"]
        held = target["held_blocks"]
        prompt = target["prompt"]
        output = target["output"]
        if any(type(value) is not int or value < 0 for value in (free, held, prompt, output)):
            raise ValueError(f"G commit invalid block/token count at step {step}")
        need = max(0, (prompt + output + 15)//16 - held)
        directly_funded += free >= need
        deficits.append(need-free)
        free_values.append(free)
        need_values.append(need)
        running_values.append(len(snapshot["running_ids"]))
        target_held_values.append(held)
    if directly_funded != 0 or min(deficits) <= 0:
        raise ValueError("G diagnostic has a direct-KV candidate; revise evidence")
    light = {}
    for cell, published_count in published_light_counts.items():
        cell_data = json.loads((folder / f"results/{cell}/selective-store.json").read_text())
        checks = [event for event in cell_data["events"] if event["event"] == "commit_check"]
        ready = sum(event["reason"] == "READY" for event in checks)
        if (cell_data["diagnostic"] is not False
            or cell_data["eligibility_logging"] != "DISABLED"
            or cell_data.get("eligibility_snapshots")
            or ready != published_count
            or cell_data["applied_rotations"] != published_count):
            raise ValueError(f"G light count or snapshot mode differs: {cell}")
        light[cell] = dict(commit_checks=len(checks), ready_commits=ready,
                           cancelled_commits=len(checks)-ready, eligibility_snapshots=0)
    return {
        "schedule_calls": raw["schedule_calls"],
        "one_to_one_eligibility_snapshots": len(snapshots),
        "ready_commits": len(commits),
        "applied_rotations": raw["applied_rotations"],
        "direct_kv_funded_commits": directly_funded,
        "min_deficit_blocks": min(deficits),
        "max_deficit_blocks": max(deficits),
        "free_blocks_range": [min(free_values), max(free_values)],
        "native_need_blocks_range": [min(need_values), max(need_values)],
        "running_count_range": [min(running_values), max(running_values)],
        "all_target_held_blocks_zero": all(value == 0 for value in target_held_values),
        "G_light_raw_event_counts_without_snapshots": light,
        "layer": "ACTUAL_G_DIAGNOSTIC_PRECOMMIT_SNAPSHOTS_OFFLINE_NECESSARY_CONDITION_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-checkout", type=Path,
                        help="Read isolated old commit_recheck REPORT and selected/full JSON")
    args = parser.parse_args()
    ledger = LEDGER.read_text()
    raw = must_match(
        r"D859 free(\d+)/need(\d+)仍驱逐(\d+)块，E1399 free(\d+)/need(\d+)仍驱逐(\d+)块；"
        r"保留全部当前peer一步后余量(\d+)/(\d+)，无pending",
        ledger, LEDGER,
    )
    dfree, dneed, dvictim, efree, eneed, evictim, dpeer, epeer = map(int, raw)
    costs = must_match(
        r"D859 victim(\d+)实际下一load(\d+)MiB/(\d+)重算/gap([0-9.]+)s，"
        r"E1399 victim(\d+)为(\d+)MiB/(\d+)/gap([0-9.]+)s",
        ledger, LEDGER,
    )
    de = {
        "D859": dict(free=dfree, need=dneed, immediate_kv_margin=dfree-dneed,
                     reported_one_peer_step_margin=dpeer, old_victim_blocks=dvictim,
                     victim_id=costs[0], victim_next_load_mib=int(costs[1]),
                     victim_next_recomputed_positions=int(costs[2]),
                     victim_next_gap_s=float(costs[3])),
        "E1399": dict(free=efree, need=eneed, immediate_kv_margin=efree-eneed,
                      reported_one_peer_step_margin=epeer, old_victim_blocks=evictim,
                      victim_id=costs[4], victim_next_load_mib=int(costs[5]),
                      victim_next_recomputed_positions=int(costs[6]),
                      victim_next_gap_s=float(costs[7])),
    }

    g_results = (ROOT / G / "RESULTS.md").read_text()
    g_diagnostic = int(must_match(r"诊断完成(\d+)次实际轮转", g_results, G / "RESULTS.md")[0])
    g_light = list(map(int, must_match(
        r"current/eager主动轮转为(\d+)/(\d+)/(\d+)/(\d+)",
        g_results, G / "RESULTS.md")))
    g_counts = dict(zip(("block0-current", "block0-eager", "block1-eager", "block1-current"), g_light))

    h_table = json.loads((ROOT / H / "execution_weste_26862/report_tables.json").read_text())
    h_counts = {}
    h_native_preempts = {}
    for cell, row in h_table["cells"].items():
        if row["status"] != "COMPLETE":
            raise ValueError(f"Unexpected H cell status {cell}: {row['status']}")
        if "native_full_native" in cell:
            if row["applied_rotations"] is not None:
                raise ValueError(f"Native cell unexpectedly has a rotation count: {cell}")
            h_native_preempts[cell] = row["sparse_counts"]["successful_preempt_methods"]
        else:
            h_counts[cell] = row["applied_rotations"]
    result = {
        "claim_layers": {
            "D_E": "ACCEPTED_LEDGER_TRANSCRIPTION_AND_ARITHMETIC_ONLY",
            "G": "PUBLISHED_RESULTS_COUNT_ONLY",
            "H": "MATERIALIZED_REPORT_TABLE_COUNT_ONLY",
        },
        "D_E": de,
        "G_selected_rotations": {"diagnostic_eager": g_diagnostic, **g_counts},
        "G_selected_total": g_diagnostic + sum(g_counts.values()),
        "H_selected_rotations": h_counts,
        "H_selected_total": sum(h_counts.values()),
        "H_native_preemptions_not_H1_commits": h_native_preempts,
        "GH_H1_opportunity_identification": {
            "lower_bound": 0,
            "upper_bound_selected_forced_commits": g_diagnostic + sum(g_counts.values()) + sum(h_counts.values()),
            "reason": "The available G/H summaries omit per-commit free, need, slots, pending and ownership."
        },
        "source_availability_primary_checkout": {
            name: blob_state(path) for name, path in {
                "old_commit_report": A / "commit_recheck/REPORT.md",
                "old_commit_selected": A / "commit_recheck/selected.json",
                "old_commit_full": A / "commit_recheck/full.json",
                "old_cost_analysis": A / "commit_recheck_cost/analysis.json",
                "G_analysis": G / "execution_weste_26862/analysis.json",
                "G_diagnostic_events": G / "execution_weste_26862/readback/results/diagnostic-eager/selective-store.json",
                "H_analysis": H / "execution_weste_26862/analysis.json",
                "H_current_events": H / "execution_weste_26862/readback/results/block0-current/selective-store.json",
                "H_report_tables": H / "execution_weste_26862/report_tables.json",
            }.items()
        },
    }
    if args.historical_checkout is not None:
        result["historical_cpu_recheck"] = historical_recheck(args.historical_checkout, de)
        result["claim_layers"]["D_E"] = "HISTORICAL_DERIVED_CPU_JSON_CROSSCHECK_NOT_RAW_OR_GPU"
        result["G_diagnostic_raw_recheck"] = g_diagnostic_recheck(args.historical_checkout, g_counts)
        result["claim_layers"]["G"] = "ACTUAL_DIAGNOSTIC_PRECOMMIT_SNAPSHOTS_AND_LIGHT_COMMIT_EVENTS"
        result["GH_H1_opportunity_identification"] = {
            "lower_bound": 0,
            "upper_bound_selected_forced_commits": sum(g_counts.values()) + sum(h_counts.values()),
            "G_diagnostic_exact_direct_kv_funded_commits": 0,
            "G_light_identification": [0, sum(g_counts.values())],
            "H_light_identification": [0, sum(h_counts.values())],
            "reason": "G diagnostic has 0/54 free>=need at aligned READY commits; G/H light cells lack per-commit snapshots.",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

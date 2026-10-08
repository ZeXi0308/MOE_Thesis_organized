"""Read-only qualification of salted native CPU KV recovery captures."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def qualify(root, expected_cpu_episodes=2, expected_episodes=6, require_cpu_load=True):
    arms, salt_owners = [], {}
    for path in sorted(root.glob("*/raw.json")):
        d, raw = path.parent, read(path)
        rows, mapping = raw["requests"], raw["internal_to_source"]
        before = read(d / "measurement_resources.json")["kv_offload"]
        after = read(d / "kv_resources_after_measurement.json")
        events, reset = read(d / "kv_offload_events.json"), read(d / "warmup_kv_reset.json")
        cpu = before["connector"] == "OffloadingConnector"
        first, preempted = {}, set()
        for step in raw["scheduler_steps"]:
            preempted.update(step["preempted_request_ids"])
            for row in step["scheduled"]:
                first.setdefault(row["request_id"], row["scheduled_start_computed"])
        positive = [x for x in events["lookup"] if x["matched"] is not None and x["matched"] > 0]
        positive_ids = {x["request"] for x in positive}
        positive_sources = {mapping.get(rid, "UNKNOWN:" + rid) for rid in positive_ids}
        counts = {name: sum(x[name]["bytes"] for x in events["transfers"]) for name in ("load", "store")}
        for row in rows:
            salt_owners.setdefault(row.get("cache_salt"), set()).add(row["external_request_id"])
        checks = dict(
            complete_16=raw["status"] == "COMPLETE" and len(rows) == 16 and all(r["status"] == "completed" for r in rows),
            salts_unique_within_episode=len({r.get("cache_salt") for r in rows}) == len(rows),
            salt_formula_matches=all(r.get("cache_salt") == "kv-recovery-r02/" + r["external_request_id"] for r in rows),
            internal_source_mapping_matches=all(mapping.get(r["internal_request_id"]) == r["request_id"] for r in rows),
            every_first_computed_position_zero=all(first.get(r["request_id"]) == 0 for r in rows),
            warmup_reset_successful=reset["success"] is True,
            premeasurement_cpu_empty=(all(before[k] == 0 for k in
                ("cpu_valid_entries", "cpu_pending_store_entries", "pending_transfer_jobs"))
                and before["pending_push_work"] is False) if cpu else before["cpu_kv_unique_bytes"] == 0,
            positive_lookup_ids_known=positive_ids <= mapping.keys(),
            positive_lookup_only_preempted_sources=positive_sources <= preempted,
            completed_transfer_summary_matches=counts == events["completed_transfer_bytes"],
            cpu_transfer_requirement=(counts['store'] > 0 and (counts['load'] > 0 or not require_cpu_load)) if cpu else counts == {"load": 0, "store": 0},
            final_no_pending=(after["cpu_pending_store_entries"] == after["pending_transfer_jobs"] == 0
                and after["pending_push_work"] is False) if cpu else after["connector"] == "NoneType",
        )
        arms.append(dict(arm=d.name, qualified=all(checks.values()), checks=checks,
            request_count=len(rows), cpu_connector=cpu, first_scheduled_positions=first,
            actual_preempted_sources=sorted(preempted), positive_lookup_sources=sorted(positive_sources),
            positive_lookup_internal_ids=sorted(positive_ids), positive_lookup_observations=len(positive),
            completed_transfer_bytes=counts, premeasurement_resources=before, final_resources=after))
    checks = dict(expected_episodes=len(arms) == expected_episodes, expected_cpu_episodes=sum(a["cpu_connector"] for a in arms) == expected_cpu_episodes,
                  salt_never_assigned_to_different_external_ids=all(len(v) == 1 for v in salt_owners.values()),
                  all_episode_checks_pass=all(a["qualified"] for a in arms))
    return dict(status="QUALIFIED" if all(checks.values()) else "FAILED", checks=checks, arms=arms,
        request_observations=sum(a["request_count"] for a in arms),
        distinct_salts_across_independent_engines=len(salt_owners),
        salt_scope="Unique within each episode; the same external request ID intentionally has the same salt across independent engine runs.",
        lookup_scope="Positive offers are checked by internal-ID/source identity, never summed as restored tokens. Transfer bytes are completed native increments.",
        qualification_scope="Measurement request isolation, first-compute boundary, observed preemption membership and clean transfer boundaries; not performance, quality, or equal-output qualification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results_kv_recovery_r02")
    parser.add_argument("--cpu-episodes", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--allow-zero-load", action="store_true", help="For non-recovery interventions that may eliminate preemptions; still check stores, IDs and complete transfer accounting")
    args = parser.parse_args()
    result = qualify(args.results, args.cpu_episodes, args.episodes, not args.allow_zero_load)
    with (args.results / "isolation_qualification.json").open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], checks=result["checks"], request_observations=result["request_observations"])))

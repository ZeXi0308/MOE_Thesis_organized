"""Locate the observed source of free blocks before qualified H1 direct actions.

Read only. No alternative policy is replayed and no saved cost is inferred.
"""
import json
from pathlib import Path
import ijson

from audit_h128_guarded_transfer_r02 import read, require, sha_file

BASE = Path(__file__).resolve().parent
QUAL = BASE / "A_H1_GUARD_QUALIFICATION_AUDIT_R02_20261001.json"
ARCHIVE = BASE / "moe-a-h1-guard-qual-session-r02-20260930/cell-00-eager_diagnostic_on/archive"


def analyze():
    require(sha_file(QUAL) == "7a3b323673ac7b42d666f758de2109ad21ed625df9957bed8cfd149a75437f08",
            "qualification changed")
    qual = read(QUAL)
    hashes = read(ARCHIVE.parent / "output_sha256.json")
    for name in ("raw.json", "selective-store.json"):
        require(sha_file(ARCHIVE / name) == hashes[name], f"original changed: {name}")
    installed = read(ARCHIVE / "selective-store.json")
    snapshots = {x["step"]: x for x in installed["eligibility_snapshots"]}
    chains = qual["direct_action"]["chains"]
    wanted = {n for chain in chains for n in (chain["step"] - 1, chain["step"])}
    steps, identities = {}, {}
    # Request and scheduler arrays precede cumulative output events in this raw.
    with (ARCHIVE / "raw.json").open("rb") as stream:
        for row in ijson.items(stream, "requests.item", use_float=True):
            identities[row["internal_request_id"]] = row["request_id"]
            if len(identities) == 128:
                break
    with (ARCHIVE / "raw.json").open("rb") as stream:
        for row in ijson.items(stream, "scheduler_steps.item", use_float=True):
            if row["step"] in wanted:
                steps[row["step"]] = row
            if row["step"] >= max(wanted):
                break
    require(set(steps) == wanted, "transition steps missing")
    results = []
    for chain in chains:
        step = chain["step"]
        before, after = snapshots[step - 1], snapshots[step]
        left, right = set(before["running_ids"]), set(after["running_ids"])
        departed, entered, common = left - right, right - left, left & right
        native_preempted = steps[step - 1]["preempted_request_ids"]
        require(entered == set() and len(departed) == 1 and
                {identities[r] for r in departed} == set(native_preempted),
                "the free-block transition has another source")
        released = sum(before["requests"][r]["held_blocks"] for r in departed)
        growth = sum(after["requests"][r]["held_blocks"] - before["requests"][r]["held_blocks"]
                     for r in common)
        require(after["free_blocks"] - before["free_blocks"] == released - growth,
                "observed held/free block balance does not close")
        store_events = [e for e in installed["events"] if
                        e.get("step") == step - 1 and e.get("event") == "store_delta"]
        prepares = [e for e in installed["events"] if e.get("step") == step - 1 and
                    e.get("event") == "prepare" and e.get("target") == chain["target_internal_id"]]
        require(len(prepares) == 1 and len(store_events) == 1,
                "missing unique preceding prepare/store receipt")
        results.append({"direct_step": step, "target_request_id": chain["target_request_id"],
            "planned_victim_request_id": chain["planned_victim_request_id"],
            "preceding_native_preempted_ids": native_preempted,
            "preceding_forced_commit_events": sum(e.get("event") == "commit" and
                e.get("step") == step - 1 for e in installed["events"]),
            "free_blocks_before_prepare": before["free_blocks"],
            "free_blocks_before_commit": after["free_blocks"],
            "departed_held_blocks": released, "surviving_running_block_growth": growth,
            "already_prepared_new_store_blocks": store_events[0]["new_store_blocks"],
            "already_prepared_store_job_ids": [j["job"] for j in store_events[0]["jobs"]],
            "direct_step_preempted_ids": steps[step]["preempted_request_ids"]})
    return {"status": "OBSERVED_PRECEDING_NATIVE_PREEMPTION_FUNDS_DIRECT", "chains": results,
            "qualification_sha256": sha_file(QUAL),
            "source_sha256": {k: hashes[k] for k in ("raw.json", "selective-store.json")},
            "scope": "Observed transition and held/free block accounting only. Direct preserves the planned victim after a different native preemption; preceding preparation stores are already incurred. No off-policy saved transfer, peer benefit or performance result."}


if __name__ == "__main__":
    output = BASE / "A_H1_OBSERVED_RESOURCE_TRANSITIONS_20261001.json"
    require(not output.exists(), "output must be new")
    result = analyze()
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))

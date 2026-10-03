"""Cache-demand diagnostic on two fixed X traces; not a G policy rerun.

Run from anywhere: python3 /path/to/this/global_lru_diagnostic.py
The only output is adjacent global_lru_diagnostic.json, created exclusively.
"""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
P = HERE.parent / "shared_pool_performance"
REPEAT = "repeat_0_retention_none_early"
CAP, EXPERT_BYTES = 384, 12582912
PHASES = ["initialization", "warmup", "warmup_injection_none_early_release8",
          "warmup_injection_frequency_early_release8",
          "warmup_injection_frequency_late_release8",
          "warmup_injection_decode_late_release8", "measurement"]


def read(path):
    return json.loads(path.read_text())


class GlobalLRU:
    def __init__(self):
        self.slots = [None] * CAP
        self.ticks = [0] * CAP
        self.mapping = {}
        self.clock = 0

    def ensure(self, layer, experts):
        active = {(layer, e) for e in experts}
        assert len(active) <= 64 <= CAP
        missing = sorted(active - self.mapping.keys())
        self.clock += 1
        for key in active & self.mapping.keys():
            self.ticks[self.mapping[key]] = self.clock
        # Pin ALL current experts, including hits and misses. Distinct victims
        # use oldest last-call tick, then lowest physical slot; empty tick is 0.
        victims = sorted((s for s, key in enumerate(self.slots) if key not in active),
                         key=lambda s: (self.ticks[s], s))[:len(missing)]
        assert len(victims) == len(missing)
        for key, slot in zip(missing, victims):
            old = self.slots[slot]
            if old is not None:
                assert old not in active
                del self.mapping[old]
            self.slots[slot] = key
            self.mapping[key] = slot
            self.ticks[slot] = self.clock
        assert active <= self.mapping.keys()
        assert len(self.mapping) <= CAP
        assert len(set(self.mapping.values())) == len(self.mapping)
        assert all(self.slots[s] == key for key, s in self.mapping.items())
        return len(missing)


def bucket():
    return dict(calls=0, active_expert_demands=0, global_entry_misses=0,
                actual_x_h2d_loads=0, actual_x_d2d_bytes=0)


def add(dst, active, global_miss, actual_miss, d2d):
    dst["calls"] += 1
    dst["active_expert_demands"] += active
    dst["global_entry_misses"] += global_miss
    dst["actual_x_h2d_loads"] += actual_miss
    dst["actual_x_d2d_bytes"] += d2d


def finish(dst):
    g, x = dst["global_entry_misses"], dst["actual_x_h2d_loads"]
    dst.update(global_h2d_payload_bytes=g * EXPERT_BYTES,
               actual_x_h2d_payload_bytes=x * EXPERT_BYTES,
               global_minus_x_loads=g - x,
               global_minus_x_h2d_bytes=(g - x) * EXPERT_BYTES,
               global_minus_x_h2d_pct=100 * (g - x) / x if x else None)
    return dst


def analyze(label):
    d = P / "results" / label
    reset_path = d / REPEAT / "reset.json"
    reset = read(reset_path)
    assert len(reset["cache_after"]) == 16 and reset["allocations_unchanged"]
    for c in reset["cache_after"].values():
        assert not c["expert_to_slot"] and set(c["slot_to_expert"]) == {-1}
        assert set(c["expert_map_device"]) == {-1}
        assert set(c["lru_tick"]) == {0} and c["lru_clock"] == 0
    records, digest, next_call = [], hashlib.sha256(), None
    with (d / "pager/calls.jsonl").open("rb") as stream:
        for line in stream:
            r = json.loads(line)
            phase = r["context"]["phase"]
            if phase != "initialization" and not phase.startswith(REPEAT + "/"):
                next_call = dict(call_id=r["call_id"], phase=phase)
                break
            digest.update(line)
            assert r["call_id"] == len(records) and r["status"] == "complete"
            records.append(r)
    assert next_call == dict(call_id=len(records), phase="repeat_1_retention_none_early/warmup")
    assert len(records) % 16 == 0
    for i in range(0, len(records), 16):
        group = records[i:i + 16]
        assert [r["layer_name"] for r in group] == [f"model.layers.{j}.mlp.experts" for j in range(16)]
        assert len({(r["context"]["phase"], r["context"]["step_id"]) for r in group}) == 1
    g = GlobalLRU()
    phases, total, previous, reset_at, last_init = {}, bucket(), None, None, {}
    for r in records:
        phase = r["context"]["phase"].removeprefix(REPEAT + "/")
        if previous == "initialization" and phase != previous:
            assert phase == "warmup"
            assert all(last_init[name] == c for name, c in reset["cache_before"].items())
            reset_at = dict(after_call_id=r["call_id"] - 1, before_call_id=r["call_id"],
                            global_resident_slots_before=len(g.mapping), after=0)
            g = GlobalLRU()
        if phase == "initialization":
            last_init[r["layer_name"]] = r["shared_plan"]["final_state"]
        if phase not in phases:
            assert phase == PHASES[len(phases)]
            phases[phase] = dict(first_call_id=r["call_id"], last_call_id=r["call_id"],
                                global_resident_slots_before=len(g.mapping),
                                totals=bucket(), by_layer={str(i): bucket() for i in range(16)})
        rows = r["row_topk_experts"]
        assert len(rows) == r["rows"] and all(len(row) == 8 for row in rows)
        assert all(type(e) is int and 0 <= e < 64 for row in rows for e in row)
        if phase != "initialization":
            ctx = r["context"]
            assert ctx["row_request_order_verified"] and ctx["valid_row_start"] == 0
            assert ctx["valid_row_stop"] == ctx["expected_rows"] == r["rows"] == len(ctx["rows"])
        active = {e for row in rows for e in row}
        assert active == set(r["active_experts"])
        layer = int(r["layer_name"].split(".")[2])
        actual = sum(len(group["loaded_experts"]) for group in r["groups"])
        assert actual == r["miss"]
        assert r["weight_copy_bytes"] == actual * EXPERT_BYTES
        assert r["weight_copy_bytes"] == sum(group["weight_copy_bytes"] for group in r["groups"])
        assert r["measurement"] == (phase == "measurement")
        if phase == "measurement":
            assert actual == len(active - set(r["entry_resident_experts"]))
        gm = g.ensure(layer, active)
        d2d = sum(group.get("d2d_copy_bytes", 0) for group in r["groups"])
        z = phases[phase]
        for dst in (total, z["totals"], z["by_layer"][str(layer)]):
            add(dst, len(active), gm, actual, d2d)
        z["last_call_id"] = r["call_id"]
        z["global_resident_slots_after"] = len(g.mapping)
        previous = phase
    assert list(phases) == PHASES and reset_at is not None
    raw_status = {}
    for phase, z in phases.items():
        finish(z["totals"])
        for amounts in z["by_layer"].values():
            finish(amounts)
        if phase == "initialization":
            continue
        raw = read(d / REPEAT / ("raw.json" if phase == "measurement" else phase + ".json"))
        assert raw["status"] == "COMPLETE"
        raw_status[phase] = dict(status=raw["status"], requests=len(raw["requests"]),
                                 outputs=sum(len(q["output_token_ids"]) for q in raw["requests"]))
        assert all(q["status"] == "completed" and len(q["output_token_ids"]) == q["max_output_tokens"] for q in raw["requests"])
    return dict(label=label, calls_path=str((d / "pager/calls.jsonl").relative_to(HERE.parent)),
                selected_calls=len(records), selected_raw_prefix_sha256=digest.hexdigest(),
                reset_json_sha256=hashlib.sha256(reset_path.read_bytes()).hexdigest(),
                reset_boundary=reset_at, next_excluded_call=next_call, raw_status=raw_status,
                phases=phases, all_selected_totals=finish(total))


if __name__ == "__main__":
    output = dict(status="FIXED_X_TRACE_CACHE_DEMAND_DIAGNOSTIC", capacity=CAP,
        expert_bytes=EXPERT_BYTES, labels=["3_oneshot", "4_oneshot"], repeats=[0],
        algorithm="Key=(layer,expert). Empty engine start; clear at recorded repeat0 reset; no reset among five warmups or before measurement. Pin entire current active set; refresh hits and fills with one call tick; sorted misses; distinct eligible victims by (last_call_tick, ascending physical slot). Each entry miss loads once. No separate stage or D2D is required by this symbolic layout.",
        boundary="Only cache demand on each observed X initialization/warmup/measurement route trace. Protected warmup rows are retained as observed, but G uses only current-call pinning throughout. Future X rows are replay inputs, not available to the simulated admission decision. This is not G's own executed route/output/KV/request trajectory, a latency estimate, a numerical qualification, or a performance result. Actual X H2D means submitted payload bytes; no wire measurement or bandwidth/overlap credit.",
        reset_source="shared_pool_performance/source/run_native_pager.py:86-117,421-459; reset outside warmup loop; shared_pool_performance/source/run_shared_pool_pager.py:20-29 initializes empty private state",
        command="python3 refine-logs/expert_saturation/outputs/admission_capacity/20260912_wisp_olmoe_r01/new_host_20260913/global_lru_diagnostic.py",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cells=[analyze(label) for label in ("3_oneshot", "4_oneshot")])
    target = HERE / "global_lru_diagnostic.json"
    with target.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    for cell in output["cells"]:
        print(cell["label"], json.dumps(cell["phases"]["measurement"]["totals"]))
    print("output", target, "bytes", target.stat().st_size)

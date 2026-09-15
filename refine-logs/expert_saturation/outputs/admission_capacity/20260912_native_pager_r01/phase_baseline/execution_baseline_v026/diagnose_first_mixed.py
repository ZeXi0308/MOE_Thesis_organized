"""Retained first mixed-call group accounting and observed active expert sets."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def analyze(root):
    entries = read(root / "episodes.json")
    trace = [json.loads(line) for line in (root / "pager/calls.jsonl").open() if line.strip()]
    cells = {}
    for entry in entries:
        phase = entry["phase"]; name = phase.split("/")[0]
        raw = read(root / name / "raw.json")
        action = raw["event_actions"][0]
        call = raw["engine_calls"][action["engine_call"]]
        records = [r for r in trace if r["context"]["phase"] == phase
                   and call["scheduler_step_start"] <= r["context"]["step_id"] < call["scheduler_step_stop"]]
        layers = {}
        for record in records:
            active = (record["active_experts"] if record.get("grouping_axis") == "expert"
                      else {e for group in record["groups"] for e in group["required_experts"]})
            layers[record["layer_name"]] = dict(active_experts=sorted(active),
                groups=len(record["groups"]), loads=sum(g["miss"] for g in record["groups"]),
                weight_copy_bytes=sum(g["weight_copy_bytes"] for g in record["groups"]))
        totals = {k: sum(layer[k] for layer in layers.values()) for k in ("groups", "loads", "weight_copy_bytes")}
        cells[name] = dict(execution=entry["execution"], engine_call=call["index"],
            scheduler_step_start=call["scheduler_step_start"], scheduler_step_stop=call["scheduler_step_stop"],
            engine_call_wall_s=call["return_s"] - call["start_s"],
            totals=dict(**totals, weight_copy_gib=totals["weight_copy_bytes"] / 2**30), layers=layers)
    names = list(cells); comparisons = []
    for a, b, kind in ((names[0], names[1], "forward_expert_minus_token"),
                       (names[3], names[2], "reverse_expert_minus_token"),
                       (names[0], names[3], "token_repeat"), (names[1], names[2], "expert_repeat")):
        left, right = cells[a], cells[b]
        layer_names = sorted(set(left["layers"]) | set(right["layers"]))
        equal = {name: (left["layers"].get(name, {}).get("active_experts") ==
                        right["layers"].get(name, {}).get("active_experts")) for name in layer_names}
        comparisons.append(dict(a=a, b=b, kind=kind, active_union_equal_by_layer=equal,
            all_active_unions_equal=all(equal.values()),
            delta_b_minus_a={k: right["totals"][k] - left["totals"][k] for k in left["totals"]}))
    return dict(status="OBSERVED_DIAGNOSTIC", cells=cells, comparisons=comparisons,
        scope="First actual engine call selected by retained event_actions.engine_call. Loads count actual misses, including repeat loads. Active expert union is observed per layer; equal unions do not establish per-row routes, hidden/logit values, or KV tensor bit equivalence.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, type=Path); p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    result = analyze(args.input_dir)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")

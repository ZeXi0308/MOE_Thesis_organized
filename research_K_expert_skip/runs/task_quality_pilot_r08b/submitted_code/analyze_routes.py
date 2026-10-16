#!/usr/bin/env python3
"""Singleton-slot occupancy from real routes, with explicit batch provenance.

CSV inputs: capture_routes_v2 (sample_id,layer,token_position,rank,expert_id,
gate_weight) or the same fields plus run_id,batch_id,phase,request_id,
decode_step. Indices: rank is 1-based; expert/layer are 0-based.

actual-decode uses captured run_id,batch_id and never reconstructs scheduling.
replay-sessions aligns independent request trajectories by decode_step; this is
counterfactual batching, even when the routes came from real cached decoding.
replay-positions packs consecutive text positions and is not decode evidence.

For each layer, q_e = number of tokens including expert e / number of tokens.
With iid tokens, fixed k and unique experts within each token, expected singleton
slot fraction is sum_e q_e(1-q_e)^(B-1)/k. It needs no within-token independence.
The fitted q curve is descriptive in-sample prediction, not held-out validation.
No confidence interval treats events/tokens from one run as independent runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def expected_singleton(q: list[float], batch_size: int, top_k: int) -> float:
    if batch_size < 1 or top_k < 1:
        raise ValueError("B and k must be positive")
    if any(not 0 <= x <= 1 for x in q) or not math.isclose(sum(q), top_k, abs_tol=1e-8):
        raise ValueError("q must be inclusion probabilities in [0,1], summing to k")
    return sum(x * (1 - x) ** (batch_size - 1) for x in q) / top_k


def occupancy(experts: list[list[int]]) -> dict:
    counts = Counter(e for token in experts for e in token)
    slots = sum(map(len, experts))
    singles = sum(n == 1 for n in counts.values())
    return dict(tokens=len(experts), slots=slots, distinct_experts=len(counts),
                singleton_slots=singles, singleton_slot_fraction=singles / slots)


def load_tokens(path: Path, mode: str, num_experts: int, top_k: int) -> list[dict]:
    grouped = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        needed = {"layer", "rank", "expert_id", "gate_weight"}
        needed |= ({"run_id", "batch_id", "request_id", "decode_step", "phase"}
                   if mode == "actual-decode" else
                   {"request_id", "decode_step", "phase"} if mode == "replay-sessions" else
                   {"sample_id", "token_position"})
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"{mode} missing fields: {sorted(needed - set(reader.fieldnames or []))}")
        for row in reader:
            if mode != "replay-positions" and row["phase"] != "decode":
                continue
            run = row.get("run_id", "historical-no-run-id")
            layer = int(row["layer"])
            request = row.get("request_id") or row["sample_id"]
            step = int(row.get("decode_step") or row.get("token_position", 0))
            batch = row.get("batch_id", "")
            key = (run, layer, batch, request, step)
            token = grouped.setdefault(key, dict(run_id=run, layer=layer,
                batch_id=batch, request_id=request, step=step, pairs={}))
            rank, expert, gate = int(row["rank"]), int(row["expert_id"]), float(row["gate_weight"])
            if rank in token["pairs"]:
                raise ValueError(f"duplicate rank in token {key}: {rank}")
            if not 0 <= expert < num_experts or not math.isfinite(gate) or gate < 0:
                raise ValueError(f"invalid expert/gate at token {key}")
            token["pairs"][rank] = (expert, gate)
    tokens = list(grouped.values())
    if not tokens:
        raise ValueError("no eligible tokens")
    for token in tokens:
        pairs = token.pop("pairs")
        if sorted(pairs) != list(range(1, top_k + 1)):
            raise ValueError(f"incomplete or variable top-k at {token}")
        token["experts"] = [pairs[r][0] for r in range(1, top_k + 1)]
        token["weights"] = [pairs[r][1] for r in range(1, top_k + 1)]
        if len(set(token["experts"])) != top_k:
            raise ValueError("duplicate expert within one token")
    return tokens


def load_vllm_tokens(path: Path, num_experts: int, top_k: int, run_id: str) -> list[dict]:
    """Read skip_worker snapshot JSON or one trace record per JSONL line.

Batch membership is recorded directly by the router matrix. Row indices identify
tokens within one step only; no persistent session identity is inferred from them.
"""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            rows = data["trace"] if "trace" in data else [data]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("expected snapshot object or trace list")
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    tokens, seen = [], set()
    for row in rows:
        if row["phase"] != "decode":
            continue
        if row.get("source") != "vllm_actual_select_experts" or row.get("forward_completed") is not True:
            raise ValueError("native trace needs verified source and forward_completed=true")
        if row["num_experts"] != num_experts or row["top_k"] != top_k:
            raise ValueError("native trace E/k does not match CLI")
        layer, step = str(row["layer"]), int(row["step"])
        key = (layer, step)
        if key in seen:
            raise ValueError("duplicate layer/step: use one policy/run snapshot per input")
        seen.add(key)
        ids, weights = row["topk_ids"], row["topk_weights"]
        if len(ids) != row["batch_size"] or len(weights) != len(ids):
            raise ValueError("native trace batch dimension mismatch")
        for i,(expert_ids,gates) in enumerate(zip(ids,weights)):
            if len(expert_ids)!=top_k or len(gates)!=top_k or len(set(expert_ids))!=top_k:
                raise ValueError("native pre-policy ids must have unique fixed top-k experts")
            if any(not isinstance(e,int) or not 0<=e<num_experts for e in expert_ids):
                raise ValueError("invalid native pre-policy expert id")
            if any(not math.isfinite(g) or g<0 for g in gates):
                raise ValueError("invalid native gate weight")
            tokens.append(dict(run_id=run_id,layer=layer,batch_id=str(step),
                request_id=f"step-{step}-row-{i}",step=step,experts=expert_ids,weights=gates))
    if not tokens:
        raise ValueError("no completed native decode routes")
    return tokens


def make_batches(tokens: list[dict], mode: str, batch_sizes: list[int]):
    pools = defaultdict(list)
    for token in tokens:
        suffix = (token["batch_id"],) if mode == "actual-decode" else (
            (token["step"],) if mode == "replay-sessions" else ())
        pools[(token["run_id"], token["layer"], *suffix)].append(token)
    dropped = Counter()
    for key, pool in sorted(pools.items()):
        pool.sort(key=lambda t: (t["request_id"], t["step"]))
        if mode == "actual-decode":
            if len({t["request_id"] for t in pool}) != len(pool):
                raise ValueError("actual decode batch includes multiple tokens from one request")
            yield key, len(pool), str(key[2]), pool, dropped
        else:
            if mode == "replay-sessions" and len({t["request_id"] for t in pool}) != len(pool):
                raise ValueError("session replay has duplicate request at same step")
            for b in batch_sizes:
                dropped[b] += len(pool) % b
                for start in range(0, len(pool) - b + 1, b):
                    name = f"{key[2:] or ('positions',)}:{start // b}"
                    yield key, b, name, pool[start:start+b], dropped


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args) -> dict:
    native = args.input_format == "vllm-json"
    if native and args.mode != "actual-decode":
        raise ValueError("vllm-json matrices already define batches; use actual-decode")
    tokens = (load_vllm_tokens(args.trace,args.experts,args.top_k,args.run_id or args.trace.stem)
              if native else load_tokens(args.trace,args.mode,args.experts,args.top_k))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    layer_tokens = defaultdict(list)
    for t in tokens:
        layer_tokens[(t["run_id"], t["layer"])].append(t)
    q_by_layer, qrows, predictions = {}, [], []
    for key, pool in sorted(layer_tokens.items()):
        counts = Counter(e for t in pool for e in t["experts"])
        q = [counts[e] / len(pool) for e in range(args.experts)]
        q_by_layer[key] = q
        for e, prob in enumerate(q):
            qrows.append(dict(model=args.model, run_id=key[0], layer=key[1],
                              expert_id=e, inclusion_probability=prob,
                              token_count=len(pool), inclusion_count=counts[e]))
        for b in args.batch_sizes:
            predictions.append(dict(model=args.model, run_id=key[0], layer=key[1],
                batch_size=b, experts=args.experts, top_k=args.top_k,
                iid_fitted_q_singleton_fraction=expected_singleton(q,b,args.top_k),
                iid_uniform_singleton_fraction=(1-args.top_k/args.experts)**(b-1),
                q_fit_token_count=len(pool)))
    batch_rows, dropped = [], Counter()
    for key, b, name, pool, dropped in make_batches(tokens, args.mode, args.batch_sizes):
        row = dict(model=args.model, mode=args.mode, run_id=key[0], layer=key[1],
                   batch_size=b, batch_id=name, experts=args.experts, top_k=args.top_k,
                   **occupancy([t["experts"] for t in pool]))
        row["iid_fitted_q_singleton_fraction"] = expected_singleton(q_by_layer[key[:2]], b, args.top_k)
        row["iid_uniform_singleton_fraction"] = (1-args.top_k/args.experts)**(b-1)
        row["gate_mass_mean"] = sum(sum(t["weights"]) for t in pool) / b
        batch_rows.append(row)
    aggregated = defaultdict(list)
    for row in batch_rows:
        aggregated[(row["run_id"],row["layer"],row["batch_size"])].append(row)
    summaries = []
    for key, rows in sorted(aggregated.items()):
        slots = sum(r["slots"] for r in rows)
        summaries.append(dict(model=args.model,mode=args.mode,run_id=key[0],layer=key[1],
            batch_size=key[2],batches=len(rows),slots=slots,
            singleton_slots=sum(r["singleton_slots"] for r in rows),
            singleton_slot_fraction=sum(r["singleton_slots"] for r in rows)/slots,
            distinct_experts_mean=sum(r["distinct_experts"] for r in rows)/len(rows),
            iid_fitted_q_singleton_fraction=rows[0]["iid_fitted_q_singleton_fraction"],
            iid_uniform_singleton_fraction=rows[0]["iid_uniform_singleton_fraction"]))
    write_csv(out/"batches.csv",batch_rows)
    write_csv(out/"summary.csv",summaries)
    write_csv(out/"inclusion_probabilities.csv",qrows)
    write_csv(out/"iid_predictions.csv",predictions)
    metadata = dict(trace=str(args.trace),trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        model=args.model,mode=args.mode,experts=args.experts,top_k=args.top_k,
        requested_batch_sizes=args.batch_sizes,tokens_across_layers=len(tokens),
        request_count=None if native else len({t['request_id'] for t in tokens}),layer_count=len(layer_tokens),
        input_format=args.input_format,
        token_identity="within-step router row only; persistent request identity absent" if native else "request/sample id from CSV",
        partial_tail_tokens_dropped_across_layers=dict(dropped),
        evidence_boundary=args.evidence_boundary,
        prediction_boundary="iid tokens; fixed k; fitted q from this input, no held-out validation",
        uncertainty_boundary="descriptive batches; no token/event pseudo-replicate confidence interval",
        missing_batch_sizes=sorted(set(args.batch_sizes)-{r['batch_size'] for r in batch_rows}))
    (out/"analysis.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--input-format",choices=["csv","vllm-json"],default="csv",
                        help="vllm-json accepts snapshot JSON, trace list JSON, or trace-row JSONL")
    parser.add_argument("--run-id",help="identity for one native trace capture (default: input stem)")
    parser.add_argument("--model",required=True)
    parser.add_argument("--experts",type=int,required=True)
    parser.add_argument("--top-k",type=int,required=True)
    parser.add_argument("--mode",choices=["actual-decode","replay-sessions","replay-positions"],required=True)
    parser.add_argument("--batch-sizes",type=int,nargs="+",default=[1,2,4,8,16,32,64])
    parser.add_argument("--evidence-boundary",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.top_k > args.experts or min(args.experts,args.top_k,*args.batch_sizes)<1:
        parser.error("require 1 <= k <= E and positive batch sizes")
    print(json.dumps(analyze(args),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

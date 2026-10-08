#!/usr/bin/env python3
"""Compare full QMSum unique 16-token prefix work with observed prompt work."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
workload = HERE / "20261002_c_qmsum_inputs_v1/workload.json"
pressure = HERE / "qmsum_pressure_v1.json"
rows = json.loads(workload.read_text())["requests"]
observed = json.loads(pressure.read_text())["summary"]
nodes, tails = [{}], 0
for row in rows:
    tokens = row["prompt_token_ids"]
    reusable, parent = (len(tokens)-1)//16, 0
    tails += len(tokens)-16*reusable
    for start in range(0, 16*reusable, 16):
        key = tuple(tokens[start:start+16])
        children = nodes[parent]
        if key not in children:
            children[key] = len(nodes)
            nodes.append({})
        parent = children[key]
unique = 16*(len(nodes)-1)+tails
result = dict(requests=len(rows), unique_full_blocks=len(nodes)-1,
    private_tail_tokens=tails, theoretical_unique_prompt_work=unique,
    actual_initial_prompt_work=observed["initial_necessary_prompt_compute_tokens"],
    actual_scheduled_prompt_work=observed["classified_prefill_tokens"],
    excess_initial_work=observed["initial_necessary_prompt_compute_tokens"]-unique,
    excess_total_work=observed["classified_prefill_tokens"]-unique,
    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (workload, pressure, Path(__file__))},
    scope="All200 prefix-trie work reference with perfect residency and one private tail per request. No wall-time optimality or decode-cost bound.")
target = HERE / "qmsum_prompt_work_v1.json"
with target.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))

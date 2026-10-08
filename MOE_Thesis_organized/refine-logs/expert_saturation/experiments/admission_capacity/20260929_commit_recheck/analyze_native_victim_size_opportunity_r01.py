#!/usr/bin/env python3
"""Descriptive size-choice opportunity in a completed native tail trajectory."""
import argparse,hashlib,json,statistics
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--archive",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
assert not args.output.exists()
path=args.archive/"selective-store.json"
store=json.loads(path.read_text())
assert store["native_victim_rule"]=="tail"
rows=[]
for decision in store["victim_decisions"]:
 current=next(x for x in decision["candidates"] if x["request"]==decision["failed_request"])
 others=[x for x in decision["candidates"] if x["qualified"] and x["request"]!=decision["failed_request"]]
 tail=next(x for x in decision["candidates"] if x["request"]==decision["selected"])
 if not current["qualified"] or not others:continue
 need=(current["bidkv"]["computed_tokens"]+1+15)//16-current["held_blocks"]-decision["free_blocks"]
 small=min(others,key=lambda x:(x["held_blocks"],-x["index"]))
 large=max(others,key=lambda x:(x["held_blocks"],x["index"]))
 rows.append(dict(step=decision["step"],computed_tokens=current["bidkv"]["computed_tokens"],
  current_held_blocks=current["held_blocks"],needed_for_next_decode=need,free_blocks=decision["free_blocks"],
  tail_blocks=tail["held_blocks"],small_blocks=small["held_blocks"],large_blocks=large["held_blocks"],
  small_changed=small["request"]!=tail["request"],large_changed=large["request"]!=tail["request"],
  failed=decision["failed_request"],small_id=small["request"],large_id=large["request"],tail_id=tail["request"]))
result=dict(status="OBSERVED_SIZE_CHOICE_OPPORTUNITY_NO_COUNTERFACTUAL_BENEFIT",
 source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 observations=len(rows),all_needed_pages=sorted(set(x["needed_for_next_decode"] for x in rows)),
 all_free_blocks=sorted(set(x["free_blocks"] for x in rows)),
 min_changed=sum(x["small_changed"] for x in rows),max_changed=sum(x["large_changed"] for x in rows),
 median_tail_blocks=statistics.median(x["tail_blocks"] for x in rows),
 median_min_blocks=statistics.median(x["small_blocks"] for x in rows),rows=rows,
 limits="Uses recorded qualified pure-decode counters to calculate next-one-token page need. Alternative post-preemption admission/output/recovery costs are unobserved. No host hit or byte-copy completion is inferred.")
args.output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="rows"}))


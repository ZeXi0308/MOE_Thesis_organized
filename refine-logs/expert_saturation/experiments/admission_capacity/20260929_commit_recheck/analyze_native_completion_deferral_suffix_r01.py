#!/usr/bin/env python3
"""Check for a completion-capacity witness in recorded suffix candidates."""
import argparse,hashlib,json
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--session",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
assert not args.output.exists()
summary={}
for arm in ("tail_break","prefix_work","prefix_finish"):
 archive=next(args.session.glob("cell-*-"+arm+"/archive"))
 source=archive/"selective-store.json"
 store=json.loads(source.read_text())
 hits=[];unknown=0;qualified=0
 for decision in store["victim_decisions"]:
  for row in decision["candidates"]:
   if not row["qualified"] or row["request"]==decision["failed_request"]:continue
   qualified+=1;score=row.get("bidkv") or {}
   if score.get("completion") is None:unknown+=1;continue
   remaining=1024-round(score["completion"]*1024)
   if remaining>0 and score["computed_tokens"]+remaining<=row["held_blocks"]*16:
    hits.append(dict(step=decision["step"],request=row["request"],
     failed=decision["failed_request"],remaining=remaining,index=row["index"],
     scheduled_prefix=row.get("scheduled_prefix"),selected=row["request"]==decision["selected"]))
 summary[arm]=dict(pressure_decisions=len(store["victim_decisions"]),
  qualified_noncurrent=qualified,unknown=unknown,capacity_fit_noncurrent_suffix=hits,
  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
args.output.write_text(json.dumps(dict(status="OBSERVED_NUMERIC_RESIDUAL_ONLY",
 scope="Only logged unprocessed suffix candidates at pressure events which actually led to preemption; no hypothetical prefix states or future EOS used.",
 arms=summary),indent=2)+"\n")
print(json.dumps({arm:dict(pressure=v["pressure_decisions"],hits=len(v["capacity_fit_noncurrent_suffix"])) for arm,v in summary.items()}))


#!/usr/bin/env python3
"""Descriptive completion area for the d6 fixed-arrival campaign; no counterfactual."""
import argparse
import json
from pathlib import Path
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('bundle',type=Path)
a=ap.parse_args();p=a.bundle
load=lambda f:json.loads(f.read_text())
w=load(p/'preparation/pkg/inputs_preparation/prepared/long/workload.json')
arrivals=dict(zip([r['request_id'] for r in w['source_requests']],w['arrival_traces_s']['steady']))
rows=[]
for spec in load(p/'STATUS.json')['cells']:
 m=load(p/'readback/results'/spec['label']/'metrics.json')
 assert m['n_completed']==32 and len(m['per_request'])==32
 completed=sorted(arrivals[r['request_id']]+r['request_latency_s'] for r in m['per_request'])
 events=sorted([(t,1) for t in arrivals.values()]+[(t,-1) for t in completed])
 area=0.;active=0;prev=0.
 for t,delta in events:area+=active*(t-prev);active+=delta;prev=t
 latency_sum=sum(r['request_latency_s'] for r in m['per_request'])
 assert active==0 and abs(area-latency_sum)<1e-8
 rows.append(dict(label=spec['label'],first_completion_s=completed[0],completion_16th_s=completed[15],
                  last_completion_s=completed[-1],unfinished_request_seconds=area,
                  mean_completion_s=area/32,completions_by_22s=sum(t<=22 for t in completed)))
with (p/'completion_distribution.json').open('x') as f:json.dump(dict(cells=rows,scope='Post-hoc observed completion distribution and exact area identity; 22s is descriptive, not an SLO or acceptance threshold.'),f,indent=2)
for r in rows:print(r)

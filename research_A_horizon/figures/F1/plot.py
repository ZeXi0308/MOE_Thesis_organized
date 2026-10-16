#!/usr/bin/env python3
"""F1: historical native recovery and controlled peer-growth context (not HORIZON)."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
native=ROOT/'B_recovery_order_20261004/recovery_timeline/reports/heldout-frozen/report.json'
exchange=ROOT/'B_recovery_order_20261004/recovery_capacity_exchange/session-after-failure-20261009-r01/cell-01-cap256-exchange_once/output/recovery-capacity-exchange.json'
d=json.loads(native.read_text());e=json.loads(exchange.read_text())
counts=Counter(x['request_id'] for x in d['episodes'])
rid=max(counts,key=lambda r:(counts[r],r))
episodes=[x for x in d['episodes'] if x['request_id']==rid]
events={x['event_id']:x for x in d['events']}
starts=[events[x['preempt_event']]['time_s'] for x in episodes]
origin=min(starts)
timeline=[]
for x,t in zip(episodes,starts):
    phase=x['phases']['preempt_to_next_output']
    if phase['status']!='observed':continue
    timeline.append(dict(episode=x['episode_id'],start=t-origin,
                         next_output=t-origin+phase['duration_s'],
                         source_request_id=x['source_request_id']))
phase_names=['preempt_to_ready','submit_begin_to_job_completed','job_completed_to_ack_retired','ack_to_first_schedule']
phase_labels=['Preempt → LOAD ready','Submit → host completion','Host completion → ACK','ACK → first schedule']
phases={name:[j['phases'][name]['duration_s'] for j in d['jobs']
              if j.get('episode_association')=='unique' and j['phases'].get(name,{}).get('status')=='observed']
        for name in phase_names}
holds=[x for x in e['events'] if x['kind']=='peer_hold']
last=max(x['step'] for x in holds)
held={x['peer']:x['peer_current_need_blocks'] for x in holds if x['step']==last}
assert len(held)==62 and sum(held.values())==62
result=dict(scope='Historical fixed-output development data; no new HORIZON result',
            native_timeline=timeline,native_load_phase_samples=phases,
            controlled_peer_growth=dict(step=last,unique_peers=len(held),pages=sum(held.values()),
                fluid_reference=250*4/16,fluid_reference_assumptions='n≈250, four steps; explanatory sizing, not a page-wise guarantee'),
            sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (native,exchange)])
(HERE/'data.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,3,figsize=(12,3.7),gridspec_kw={'width_ratios':[1.25,1.6,1]})
for i,row in enumerate(timeline):
    ax[0].broken_barh([(row['start'],row['next_output']-row['start'])],(i-.25,.5),facecolors='#c7643e')
    ax[0].plot(row['next_output'],i,'o',color='#1c687a',ms=5)
ax[0].set(yticks=range(len(timeline)),yticklabels=[r['episode'] for r in timeline],
          xlabel='Seconds since first shown preemption',title='(a) Repeated native recovery')
ax[0].text(.02,.97,timeline[0]['source_request_id'],transform=ax[0].transAxes,fontsize=7,va='top')
for i,name in enumerate(phase_names):
    vals=phases[name]
    ax[1].scatter([v*1000 for v in vals],[i]*len(vals),s=24,color='#1c687a',alpha=.7)
    if vals:ax[1].plot([sum(vals)/len(vals)*1000]*2,[i-.19,i+.19],color='#c7643e',lw=2)
ax[1].set(yticks=range(4),yticklabels=phase_labels,xlabel='Observed interval (ms; log scale)',
          xscale='log',title='(b) Native LOAD intervals, n=3')
ax[1].invert_yaxis()
ax[2].bar(['Observed\n(62 peers)','Fluid\n250 × 4 / 16'],[62,62.5],color=['#1c687a','#aaa'])
for i,v in enumerate([62,62.5]):ax[2].text(i,v+1,str(v),ha='center')
ax[2].set(ylabel='Pages',ylim=(0,78),title='(c) New peer demand')
fig.text(.02,.015,'Historical fixed-output data. Dots: receipts (a), individual LOAD jobs (b); ticks: means (b). Host intervals can overlap. Panel (c): separate controlled run.',fontsize=7)
fig.tight_layout(rect=(0,.08,1,1))
fig.savefig(HERE/'F1.pdf');fig.savefig(HERE/'F1.png',dpi=180)
print(json.dumps(dict(episodes=len(timeline),native_load_jobs=len(d['jobs']),peer_pages=sum(held.values()))))

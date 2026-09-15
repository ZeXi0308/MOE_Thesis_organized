#!/usr/bin/env python3
"""One old, real S on run through the exact reused ledger/resource accounting."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'experiments/admission_capacity'))
import analyze_restore_token_reservation as new

S=ROOT/'outputs/admission_capacity/20260914_restore_completion_r01'
source=S/'preparation/pkg';folder=S/'execution/readback/results/block0-d6-restore-on'
raw=new.read(folder/'raw.json');decisions=new.read(folder/'component-decisions.json')
components=new.module('smoke_real_S_counters',source/'recovery_service_components.py')
helper=new.module('smoke_real_S_helper',source/'restore_obligation.py')
bound=new.inspect_cell.__globals__
assert all(k in bound['component_accounting'].__globals__ for k in ('math','bisect_right'))
expected,ledger=bound['replay_obligations'](raw,decisions,components,helper,True)
result=bound['component_accounting'](raw,decisions,True,components.LTRCounters(200,10),expected)
assert ledger['snapshot']==new.read(folder/'restore-obligations.json')
assert ledger['replayed_steps']==1906 and ledger['overlay_selected_calls']==81
assert result['totals']['victim_events']==27
out=dict(status='PASS',scope='Exact reused accounting invoked through new inspector globals on one complete old S run; not T execution.',
    source_cell=str(folder),raw_sha256=new.sha(folder/'raw.json'),analyzer_sha256=new.sha(Path(new.__file__)),
    reused_restore_analyzer_sha256=new.PRIOR_SHA,replayed_steps=1906,actual_selected_minus2=81,
    actual_preemptions=27,resource_totals=result['totals'],restore_releases=ledger['releases'])
path=HERE/'smoke_resource.json'
assert not path.exists(),'preserve smoke receipt'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('status','replayed_steps','actual_selected_minus2','actual_preemptions')}))

"""Targeted stream-order regression for the actual dispatcher; CPU mock of CUDA queue API."""
import argparse,hashlib,json,sys
from pathlib import Path
from threading import Lock
from types import SimpleNamespace as NS
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.source.resolve()))
import run_shared_pool_pager as m
log=[];layer=NS(layer_name='model.layers.0.mlp.experts')
class Event:
    def __init__(self):self.last=0
    def record(self,current):log.append(['record',current.cuda_stream]);self.last=current.cuda_stream
class Stream:
    def __init__(self,n):self.cuda_stream=n
    def wait_event(self,event):log.append(['wait',self.cuda_stream,event.last])
current=Stream(0)
r=NS(shared_weights=(None,None),shared_call_lock=Lock(),shared_stream=0,shared_last_event=Event(),shared_transitions=[],
     next_call_id=0,context={},torch=NS(cuda=NS(current_stream=lambda:current)),group_retention='none',
     shared_qualification=False,shared_mode='split')
def ordinary(*unused):log.append(['access',current.cuda_stream]);return 7
for n in [0,0,7,7,0]:
    current=Stream(n);assert m.ordered_dispatch(r,ordinary,None,layer,None,None,None)==7
assert log==[['access',0],['record',0],['access',0],['record',0],['wait',7,0],['access',7],['record',7],['access',7],['record',7],['wait',0,7],['access',0],['record',0]]
r.shared_call_lock.acquire()
try:
    try:m.ordered_dispatch(r,ordinary,None,layer,None,None,None)
    except RuntimeError as e:assert 'concurrent' in str(e)
    else:raise AssertionError('concurrent access allowed')
finally:r.shared_call_lock.release()
def broken(*unused):raise ValueError('deliberate')
try:m.ordered_dispatch(r,broken,None,layer,None,None,None)
except ValueError:pass
else:raise AssertionError('failure swallowed')
assert not r.shared_call_lock.locked() and log[-1]==['record',0]
result=dict(status='PASS',checks=['same-stream FIFO','cross-stream wait before access','event recorded after access','concurrent host rejected','failure event and lock release'],
            source_sha256=hashlib.sha256((args.source/'run_shared_pool_pager.py').read_bytes()).hexdigest(),events=log,
            command=[sys.executable,*sys.argv])
with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print('stream dependency targeted regression PASS')

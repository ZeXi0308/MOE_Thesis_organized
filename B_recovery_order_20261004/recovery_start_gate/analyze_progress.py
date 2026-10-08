#!/usr/bin/env python3
"""Three fixed, conditional capacity readouts; no policy/service comparison."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

PURE_SHA = '8ed4aab468763f880f5f8fe3adbf2517ed741f4602d12d882ca4eaf8d9e7b73d'
PURE_PATH = Path(__file__).with_name('progress_capacity.py')
if hashlib.sha256(PURE_PATH.read_bytes()).hexdigest() != PURE_SHA:
    raise RuntimeError('Frozen progress arithmetic changed')
from progress_capacity import estimate

COLUMNS = ('request', 'status', 'history_tokens', 'held_gpu_blocks', 'computed_tokens',
           'num_in_flight_tokens', 'scheduled_tokens_this_step', 'is_running',
           'is_inflight', 'registry_identity', 'plain_unshared_blocks')
CASES = (('current_history', 0, 0), ('extra_one_position', 1, 1), ('extra_sixteen_positions', 16, 16))


def decode(snapshot):
    columns = snapshot['columns']
    if len(columns) != len(COLUMNS) or set(columns) != set(COLUMNS):
        raise ValueError('UNKNOWN_COLUMNS')
    def row(values):
        if len(values) != len(columns):
            raise ValueError('TUPLE_LENGTH_MISMATCH')
        result = dict(zip(columns, values))
        if result['registry_identity'] is not True or result['plain_unshared_blocks'] is not True:
            raise ValueError('UNVERIFIED_IDENTITY_OR_OWNERSHIP')
        if type(result['is_running']) is not bool or type(result['is_inflight']) is not bool:
            raise ValueError('UNKNOWN_MEMBERSHIP')
        return result
    target = row(snapshot['target'])
    requests = [row(values) for values in snapshot['requests']]
    for key, flag in (('running_ids', 'is_running'), ('inflight_ids', 'is_inflight')):
        ids = snapshot[key]
        if len(set(ids)) != len(ids) or set(ids) != {r['request'] for r in requests if r[flag]}:
            raise ValueError('MEMBERSHIP_SET_MISMATCH')
    if target['is_running'] or target['is_inflight']:
        raise ValueError('TARGET_ALREADY_RUNNING_OR_INFLIGHT')
    return dict(target=target, requests=requests, running_ids=snapshot['running_ids'],
        inflight_ids=snapshot['inflight_ids'], native_reserved_blocks=snapshot['native_reserved_blocks'],
        free_gpu_blocks=snapshot['free_gpu_blocks'], block_size=snapshot['block_size'],
        max_model_len=snapshot['max_model_len'], qualified=True)


def analyze_snapshot(snapshot, events, index):
    out = {key: snapshot.get(key) for key in ('step', 'host_perf_s', 'return_host_perf_s',
        'snapshot_s', 'native_reserved_blocks', 'free_gpu_blocks', 'original_break_return',
        'native_continuation_permitted', 'allocation_has_not_executed', 'native_arguments')}
    out.update(index=index, observer_status=snapshot.get('status'),
               observer_reasons=snapshot.get('reasons', []), scenarios={})
    # A readable RID still links raw action/release events when other snapshot
    # fields are UNKNOWN; it does not qualify the capacity calculation.
    try:
        rid=snapshot['target'][snapshot['columns'].index('request')]
        if isinstance(rid,str):out['request']=rid
    except (KeyError,TypeError,ValueError,IndexError):
        pass
    unknown = None
    try:
        if snapshot.get('status') != 'KNOWN':
            raise ValueError('OBSERVER_NOT_KNOWN')
        args = decode(snapshot)
        out['request'] = args['target']['request']
        out['running_count'] = len(args['running_ids'])
        out['inflight_count'] = len(args['inflight_ids'])
        # Reservation consistency is reported even if the pure estimator rejects
        # another field. This is arithmetic on this snapshot, never another time.
        b, limit = args['block_size'], args['max_model_len']
        if type(b) is not int or b <= 0 or type(limit) is not int or limit <= 0:
            raise ValueError('INVALID_BLOCK_OR_CONTEXT_SIZE')
        by_id = {r['request']: r for r in args['requests']}
        if len(by_id) != len(args['requests']):
            raise ValueError('DUPLICATE_REQUEST_ROWS')
        remaining = []
        for rid in args['inflight_ids']:
            n, held = by_id[rid]['history_tokens'], by_id[rid]['held_gpu_blocks']
            if any(type(v) is not int or v < 0 for v in (n, held)):
                raise ValueError('INVALID_RESERVATION_COUNTS')
            remaining.append(max(0, (min(n, limit)+b-1)//b-held))
        out['reservation_consistency'] = dict(native=args['native_reserved_blocks'],
            reconstructed_current_history=sum(remaining),
            matches=sum(remaining) == args['native_reserved_blocks'])
    except (KeyError, TypeError, ValueError) as error:
        unknown = str(error)
        args = None
        out['reservation_consistency'] = dict(native=snapshot.get('native_reserved_blocks'),
            reconstructed_current_history=None, matches=None)
    for label, gt, w in CASES:
        result = (estimate(**args, target_growth_tokens=gt, growth_tokens=w) if args is not None
                  else dict(status='UNKNOWN', reason=unknown, capacity_fit=None))
        # The source contains the per-RID data; avoid triplicating those rows.
        out['scenarios'][label] = {k:v for k,v in result.items() if k != 'per_request'}
        out['scenarios'][label].update(target_growth_tokens=gt, growth_tokens=w)
    out['status'] = 'KNOWN' if all(v['status'] == 'KNOWN' for v in out['scenarios'].values()) else 'UNKNOWN'
    rid = out.get('request')
    stamp = snapshot.get('host_perf_s')
    releases = [(i,e) for i,e in enumerate(events) if e.get('kind') == 'release'
                and e.get('request') == rid and isinstance(stamp, (int,float))
                and isinstance(e.get('host_perf_s'), (int,float)) and e['host_perf_s'] <= stamp]
    latest = releases[-1] if releases else None
    out['preceding_release'] = None if latest is None else dict(index=latest[0],
        **{k:latest[1].get(k) for k in ('reason','step','host_perf_s','cohort_finished','free_gpu_blocks')})
    out['continuation_after_cohort_finish_release'] = bool(
        out['native_continuation_permitted'] is True and latest
        and latest[1].get('reason') == 'COHORT_REQUEST_FINISHED')
    out['actual_break_events'] = [{k:e.get(k) for k in ('step','ordinal','host_perf_s')}
        for e in events if e.get('kind') == 'actual_break_executed'
        and e.get('request') == rid and e.get('step') == snapshot.get('step')]
    return out


def analyze_cell(document):
    obs = document.get('progress_observation')
    result = dict(mode=document.get('mode'), policy_status=document.get('status'),
        action_count=document.get('action_count'), executed_breaks=document.get('executed_breaks'),
        snapshots=[], unknown_reasons={})
    if not isinstance(obs, dict):
        return result | dict(status='UNVERIFIED', reason='MISSING_PROGRESS_OBSERVATION', snapshot_count=0,
                             snapshot_s_sum=None, first=None, first_native_attempt=None)
    rows = [analyze_snapshot(s, document.get('events', []), i) for i,s in enumerate(obs.get('snapshots', []))]
    reasons = Counter()
    for row in rows:
        if row['status'] == 'UNKNOWN':
            reasons.update(set(row['observer_reasons']) | {s.get('reason','UNKNOWN')
                for s in row['scenarios'].values() if s['status'] == 'UNKNOWN'})
    costs = [r['snapshot_s'] for r in rows]
    valid_cost = lambda v: type(v) in (int,float) and math.isfinite(v) and v >= 0
    summary = lambda r: None if r is None else {k:v for k,v in r.items()
        if k not in ('actual_break_events', 'native_arguments', 'observer_reasons')}
    result.update(status='ANALYZED', snapshots=rows, snapshot_count=len(rows),
        observer_snapshot_count=obs.get('snapshot_count'), observer_snapshot_s=obs.get('snapshot_s'),
        snapshot_s_sum=sum(costs) if all(valid_cost(c) for c in costs) else None,
        unknown_cost_count=sum(not valid_cost(c) for c in costs), unknown_reasons=dict(reasons),
        unknown_snapshot_count=sum(r['status']=='UNKNOWN' for r in rows), observer_outcome=obs.get('outcome'),
        first=summary(rows[0]) if rows else None,
        first_native_attempt=summary(next((r for r in rows if r['native_continuation_permitted'] is True), None)))
    return result


def self_check():
    def row(rid,n,h,c,i,q,running=False,inflight=False):
        return (rid,'RUNNING' if running else 'PREEMPTED',n,h,c,i,q,running,inflight,True,True)
    snapshot = dict(step=2,host_perf_s=20.,return_host_perf_s=20.001,snapshot_s=.001,status='KNOWN',reasons=[],
        columns=list(COLUMNS),target=row('target',49,0,0,0,0),
        requests=[row('load',65,2,32,0,0,inflight=True),row('prefill',65,3,32,16,16,True,True),row('decode',32,2,31,0,1,True)],
        running_ids=['prefill','decode'],inflight_ids=['load','prefill'],native_reserved_blocks=5,
        free_gpu_blocks=11,block_size=16,max_model_len=4096,original_break_return=False,
        native_continuation_permitted=True,allocation_has_not_executed=True)
    release=dict(kind='release',request='target',reason='COHORT_REQUEST_FINISHED',step=1,host_perf_s=19.)
    result=analyze_snapshot(snapshot,[release],0)
    assert [s['required_free_blocks'] for s in result['scenarios'].values()]==[9,10,12]
    assert [s['capacity_fit'] for s in result['scenarios'].values()]==[True,True,False]
    assert result['reservation_consistency']['matches'] and result['continuation_after_cohort_finish_release']
    # Completion can precede a failing conditional capacity result.
    assert result['scenarios']['extra_sixteen_positions']['capacity_fit'] is False
    bad=analyze_snapshot(snapshot | dict(native_reserved_blocks=6),[],0)
    assert bad['status']=='UNKNOWN' and bad['reservation_consistency']['matches'] is False
    missing=analyze_snapshot(snapshot | dict(columns=[]),[],0)
    assert missing['status']=='UNKNOWN'
    uncertain=analyze_snapshot(snapshot | dict(status='UNKNOWN',reasons=['NONPLAIN_BLOCK']),[release],0)
    assert uncertain['status']=='UNKNOWN' and uncertain['continuation_after_cohort_finish_release']
    empty=analyze_cell(dict(mode='native',events=[],progress_observation=dict(snapshots=[],snapshot_count=0,snapshot_s=0.)))
    assert empty['snapshots']==[] and empty['first'] is None and empty['first_native_attempt'] is None
    print('PASS: actual tuple names/Q and membership decoder; fixed three explicit assumptions; reservation mismatch UNKNOWN; completion is not fit; missing/no snapshots remain missing. CPU only.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',nargs='?',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        self_check()
    if args.session is None:
        if args.self_check:return 0
        parser.error('session is required unless only --self-check is requested')
    session=args.session.resolve();receipt_path=session/'receipt.json'
    receipt=json.loads(receipt_path.read_text())
    report=dict(session=str(session),receipt_status=receipt.get('status'),receipt_cells=receipt.get('cells',[]),cells=[],
        assumptions=[dict(label=name,target_growth_tokens=gt,growth_tokens=w) for name,gt,w in CASES],
        semantics='Three prespecified CPU arithmetic diagnostics, not a parameter search or strategy outcome. Growth is extra KV token positions, not schedule rounds. No cross-run same-state claim, completion guarantee, future-free credit, or observation-cost subtraction.',
        source_sha256={str(Path(__file__).name):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       PURE_PATH.name:PURE_SHA,'receipt.json':hashlib.sha256(receipt_path.read_bytes()).hexdigest()})
    for directory in sorted(session.glob('cell-*')):
        if not directory.is_dir():continue
        path=directory/'output/recovery-start-gate.json'
        index=int(directory.name.split('-')[1])
        recorded=receipt.get('cells',[])
        row=dict(directory=directory.name,receipt_cell=recorded[index] if index<len(recorded) else None)
        if not path.exists():
            row.update(status='UNVERIFIED',reason='MISSING_POLICY_ARTIFACT',snapshots=[])
        else:
            row.update(analyze_cell(json.loads(path.read_text())))
            row['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        report['cells'].append(row)
    output=args.output or session/'capacity-snapshots.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

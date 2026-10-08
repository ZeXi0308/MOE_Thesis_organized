"""Read-only six-cell context calibration; complete-policy gap curves."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'experiments/admission_capacity'))
from analyze_gap_service_tradeoff import request_metrics, compare, baseline_envelope


def read(p):
    return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    assert not a.output_dir.exists(), 'retain previous analysis'
    readback=a.source/'execution/readback';receipt=read(a.source/'execution/recovery.json')
    assert receipt['status']=='COMPLETE_READBACK' and receipt['all_package_files_match']
    group=read(readback/'group-status.json')
    assert group==receipt['group'] and group['status']=='COMPLETE' and group['returncode']==0
    cells=[];prompt_identity=None
    for block in (0,1):
        for role in ('native','most_output','least_progress'):
            label=f'context-block{block}-{role}';folder=readback/'results'/label
            rawpath=folder/'raw.json';raw=read(rawpath);cfg=read(folder/'config.json')
            assert read(folder/'status.json')['status']=='COMPLETE' and cfg['variant']==role
            assert cfg['fixed_kv_cache_memory_bytes']==13960740864
            assert read(folder/'safe-cap-qualification.json')['usable_blocks']==6656
            decisions=read(folder/'headroom-decisions.json')
            assert len(decisions)==len(raw['scheduler_steps'])
            for d,s in zip(decisions,raw['scheduler_steps']):
                assert d['step']==s['step'] and d['status']=='APPLIED'
                assert d['actual_scheduled']=={x['internal_request_id']:x['scheduled_tokens'] for x in s['scheduled']}
                assert {raw['internal_to_source'][rid] for rid in d['preempted']}==set(s['preempted_request_ids'])
                if role=='native':assert d['mode']=='native' and not d['held']
            rows=request_metrics(raw)
            assert len(rows)==32 and all(len(r['output_token_ids'])==1024 for r in raw['requests'])
            assert Counter(r['prompt_tokens'] for r in raw['requests'])=={2560:16,3072:16}
            identity={r['request_id']:(r['prompt_tokens'],r['prompt_token_ids_sha256'],r['arrival_s']) for r in raw['requests']}
            assert prompt_identity is None or prompt_identity==identity
            prompt_identity=identity;wall=raw['observation_end_s']
            cells.append(dict(label=label,block=block,role=role,wall_s=wall,requests=rows,
                all_completed_rate=len(rows)/wall,mean_completion_s=statistics.mean(r['completion_s'] for r in rows),
                mean_ttft_s=statistics.mean(r['ttft_s'] for r in rows),max_gap_s=max(r['max_gap_s'] for r in rows),
                preemptions=raw['actual_preemption_count'],output_tokens=32768,
                output_sha256={r['request_id']:hashlib.sha256(json.dumps(r['output_token_ids'],separators=(',',':')).encode()).hexdigest() for r in raw['requests']},
                raw_path=str(rawpath),raw_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest()))
            del raw
    by={r['label']:r for r in cells};comparisons=[];envelopes=[]
    for block in (0,1):
        n,m,l=[by[f'context-block{block}-{role}'] for role in ('native','most_output','least_progress')]
        for x,y in ((n,m),(n,l),(m,l)):
            c=compare(x,y);c['same_output_requests']=sum(x['output_sha256'][k]==y['output_sha256'][k] for k in x['output_sha256'])
            comparisons.append(c)
        envelopes.append(baseline_envelope([n,m],l))
    result=dict(status='MEASUREMENT_ONLY',cells=cells,comparisons=comparisons,envelopes=envelopes,
        source_readback_sha256=receipt['archive_sha256'],
        boundary='Existing cohort3 documents with truncated alternating prompts and fixed output1024; not independent documents or unknown EOS. All capture/host costs included. Engine-return gaps are not client receipt; Q(g) has no TTFT/TPOT constraint or within-episode policy switching. Host peak unmeasured. No method/Oracle claim.')
    a.output_dir.mkdir(parents=True);(a.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ('label','wall_s','all_completed_rate','mean_completion_s','max_gap_s','preemptions')} for r in cells],indent=2))


if __name__=='__main__':main()

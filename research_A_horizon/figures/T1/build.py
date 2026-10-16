"""All analyzed runs, including failures and diagnostic protocols, without pooling."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def main():
    rows=[];sources=[]
    for file in sorted((ROOT/'runs').glob('*/metrics.json')):
        directory=file.parent
        receipt_path=directory/'receipt.json'
        if not receipt_path.exists():continue
        receipt=json.loads(receipt_path.read_text())
        sources.append(dict(session=directory.name,metrics_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                            receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest()))
        for metrics in json.loads(file.read_text()):
            name=Path(metrics['cell']).name;index=int(name.split('-')[1])
            cfg=receipt['plan']['cells'][index]
            inp=directory/name/'inputs.json'
            payload=json.loads(inp.read_text()) if inp.exists() else {}
            config=payload.get('config',{})
            executed=next((c for c in receipt['cells'] if c['index']==index),{})
            runner=next((arg for arg in executed.get('command',[]) if arg.endswith('/run_cell.py')),None)
            row=dict(session=directory.name,cell=name,complete=metrics.get('complete',False),
                status=metrics['status']['status'],model_revision=config.get('model_revision'),
                code_snapshot=Path(runner).parent.name if runner else None,
                workload=config.get('workload'),kv_gib=cfg['kv_gib'],arrival_rate=cfg['arrival_rate'],
                policy=cfg['policy'],cap=cfg['cap'],ha=cfg.get('ha'),hr=cfg.get('hr'),
                lease_victim=cfg.get('lease_victim'),stall_order=cfg.get('stall_order'),
                estimate_hr=cfg.get('estimate_hr'),warmup=cfg.get('warmup_profile','legacy'),
                comparison_use='download-overlap diagnostic' if directory.name=='w2-48-warmup-mc-r07' and index<2
                    else 'broad diagnostic' if cfg.get('warmup_profile')=='broad' else 'development',
                planned=metrics.get('n_planned'),not_arrived=metrics.get('n_not_arrived'),
                arrived=metrics.get('n_arrived'),completed=metrics.get('completed'),
                failed=metrics.get('failed'),unfinished=metrics.get('unfinished'))
            for metric,quantiles in [('ttft_s',['p50','p95']),('maxgap_s',['p99']),('flow_s',['mean','p95']),
                                     ('peer_committed_ratio',['p50','p95']),
                                     ('peer_horizon_growth_ratio',['p50','p95'])]:
                for q in quantiles:row[metric+'_'+q]=metrics.get(metric,{}).get(q)
            for metric in ['stall_rate','output_token_s','joint_goodput_rps','preempted_requests',
                           'preempted_request_fraction','preemption_count','thrash_count','scheduler_us_per_step',
                           'controller_seconds','ledger_seconds','observation_s','delivered_output_tokens',
                           'periodic_last512_fraction','unknown_or_censored_gap_count']:
                row[metric]=metrics.get(metric)
            for reason in ['stop','length']:row['finish_'+reason]=metrics.get('finish_reasons',{}).get(reason)
            row['mc_relaxations']=metrics.get('mc_activity',{}).get('successful_relaxations')
            rows.append(row)
    (OUT/'data.json').write_text(json.dumps(dict(rows=rows,sources=sources,
        semantics='Every locally analyzed cell. No pooling across warmup protocols or calibration conditions. '
        'Null denotes unavailable or inapplicable. Observational repetition sanity is not a quality-equivalence test.'),indent=2)+'\n')
    if rows:
        with (OUT/'all-runs.tsv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]),delimiter='\t')
            writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows),complete=sum(r['complete'] for r in rows))))


if __name__=='__main__':main()

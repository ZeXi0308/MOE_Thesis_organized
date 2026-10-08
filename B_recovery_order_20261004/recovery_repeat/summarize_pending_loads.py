#!/usr/bin/env python3
"""Existing post-output re-preemptions joined to native LOAD host lifecycle only."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    causes_path = args.session/'repreempt-causes.json'
    causes = json.loads(causes_path.read_text())
    inputs = {str(causes_path): hashlib.sha256(causes_path.read_bytes()).hexdigest()}
    lifecycles, unknown = {}, {}
    stages = ('job_created', 'ready', 'submit_begin', 'submit_end', 'job_completed', 'ack_retired')
    rows = []
    for action in causes['rows']:
        cell = action['cell']
        if cell not in lifecycles:
            path = args.session/cell/'output/recovery-order.json'
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            events = json.loads(path.read_text())['events']
            jobs, unidentified = {}, []
            for event in events:
                if event['kind'] not in stages:
                    continue
                if event.get('is_store') is None:
                    unidentified.append(event)
                if event.get('is_store') is not False:
                    continue
                job = jobs.setdefault(event['job_id'], dict(job_id=event['job_id'], request=event.get('request')))
                job.setdefault(event['kind'], event['host_perf_s'])
            lifecycles[cell], unknown[cell] = list(jobs.values()), unidentified
        stamp = action['victim_decision']['host_perf_counter_s']
        jobs = lifecycles[cell]
        pending = [job for job in jobs if job.get('job_created', float('inf')) <= stamp
                   and job.get('ack_retired', float('inf')) > stamp]
        incomplete = [job for job in jobs if any(stage not in job for stage in stages)]
        rows.append(dict(cell=cell, action_number=action['action_number'],
            target_source_request=action['target_source_request'], victim_decision_host_perf_s=stamp,
            status='UNVERIFIED' if incomplete or unknown[cell] else 'ANALYZED',
            created_loads_before_decision=sum(job.get('job_created', float('inf')) <= stamp for job in jobs),
            acknowledged_loads_before_decision=sum(job.get('ack_retired', float('inf')) <= stamp for job in jobs),
            pending_created_loads=pending,
            pending_unsubmitted_loads=[job for job in pending if job.get('submit_begin', float('inf')) > stamp],
            incomplete_lifecycles=incomplete, unidentified_stage_event_count=len(unknown[cell])))
    result = dict(rows=rows, summary=dict(decisions=len(rows),
        analyzed=sum(row['status'] == 'ANALYZED' for row in rows),
        decisions_with_unacknowledged_created_load=sum(bool(row['pending_created_loads']) for row in rows),
        decisions_with_unsubmitted_created_load=sum(bool(row['pending_unsubmitted_loads']) for row in rows)),
        load_lifecycles=lifecycles, inputs_sha256=inputs,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        semantics='Within-run joins at the actual victim decision host timestamp. job_created is observed '
            'after native connector metadata construction; job_completed is host polling, not a GPU '
            'absolute timestamp; ack_retired is after native scheduler confirmation removes the job. '
            'No pending observed LOAD does not prove that no uncreated future recovery could be useful. '
            'Completing LOAD does not itself release its resident GPU KV. No alternate execution is inferred.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()

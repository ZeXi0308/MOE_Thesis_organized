"""CPU-only STORE coverage diagnosis; Python standard library, no runtime imports.

python3 -B diagnose_store_coverage.py RAW [RAW ...] --out report.json
Optional --compare compares the first two complete traces (second minus first).
Optional --extra-events JSON accepts explicit EVICT events for a SINGLE raw trace.
See STORE_DIAGNOSTIC.md for the evidence contract and limitations.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_host_history import allocation_link, completion_view


def intervals(parts):
    out = []
    for a, b in sorted(parts):
        if a >= b:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([a, b])
    return out


def intersect(a, b):
    return intervals([[max(x, u), min(y, v)] for x, y in a for u, v in b
                      if max(x, u) < min(y, v)])


def size(parts):
    return sum(b-a for a, b in intervals(parts))


def contains(parts, block):
    return any(a <= block < b for a, b in parts)


def position_work(raw):
    """Actual scheduled positions, not FLOPs, time, or presumed future work."""
    seen, rows = defaultdict(list), defaultdict(list)
    for step, schedule in enumerate(raw.get('scheduler_steps', [])):
        for index, row in enumerate(schedule.get('scheduled', [])):
            rid = row['request_id']; span = [[row['start_computed'], row['end_computed']]]
            repeated = intersect(seen[rid], span)
            rows[rid].append(dict(step=step, time_s=schedule['time_s'], span=span[0],
                repeated=repeated, pointer=f'/scheduler_steps/{step}/scheduled/{index}'))
            seen[rid] = intervals(seen[rid]+span)
    totals = {rid: dict(scheduled_positions=sum(r['span'][1]-r['span'][0] for r in rr),
                       unique_positions=size(seen[rid]),
                       repeated_positions=sum(size(r['repeated']) for r in rr))
              for rid, rr in rows.items()}
    return rows, totals


def common_event(path, kind, rid, pointer, time=None, step=None, blocks=None,
                 job=None, tier=None, **fields):
    return dict(request_id=rid, job_id=job, kind=kind, time=time,
        clock_domain=f'episode_monotonic:{path}' if time is not None else None,
        scheduler_step=step, logical_block_interval=blocks, storage_tier=tier,
        raw_evidence_location=dict(file=str(path), pointer=pointer),
        content_version=None, **fields)


def classify_gap(first_block, followup, jobs, builders_complete,
                 repeated_full_block, evictions=()):
    """Classify evidence at the next lookup, not the unseen residency history.

    Absence of an ACK is never a pending proof. Completion followed by a miss is
    never an eviction proof. EVICT must identify the STORE lifecycle, not a reused
    physical slot. Job completion ordering is supplied by completion_view.
    """
    if followup is None or followup.get('hit_chunks') is None:
        return 'INSUFFICIENT_EVIDENCE', 'No numeric subsequent Host prefix observation.'
    if followup['hit_chunks'] > first_block:
        return None, 'First non-ready block lies within the later observed ready prefix.'
    if followup['hit_chunks'] < first_block:
        return 'INSUFFICIENT_EVIDENCE', 'A preceding hole hides this block at the later prefix lookup.'
    covering = [j for j in jobs if j['mapping_complete'] and
                contains(j['logical_block_intervals'], first_block)]
    if not covering:
        if builders_complete and repeated_full_block:
            return 'NOT_CREATED', 'No STORE for this block in the fully observed recomputation window.'
        return 'INSUFFICIENT_EVIDENCE', 'Missing builder, mapping, or full-block recomputation evidence.'
    # Only the newest creation for this block can explain its current lifecycle.
    job = max(covering, key=lambda j: j['creation_record_index'])
    completion = job['completion']
    acknowledged = completion['acknowledged_before_next_lookup'] is True
    if acknowledged:
        for event in evictions:
            if (event.get('kind') != 'EVICT' or event.get('request_id') != job['request_id'] or
                event.get('job_id') != job['job_id'] or
                event.get('creation_record_index') != job['creation_record_index'] or
                event.get('storage_tier') != 'Host' or
                event.get('clock_domain') != followup['clock_domain'] or
                not contains([event.get('logical_block_interval') or [0, 0]], first_block)):
                continue
            if (event.get('content_version') is not None and job.get('content_version') is not None
                    and event['content_version'] != job['content_version']):
                continue
            # End of ACK step is a safe upper bound on callback return.
            ack_end = job.get('ack_step_end_s')
            t = event.get('time')
            if ack_end is not None and t is not None and ack_end < t < followup['time']:
                return 'COMPLETED_THEN_EVICTED', 'Explicit eviction of the same STORE lifecycle follows ACK and precedes lookup.'
        return 'INSUFFICIENT_EVIDENCE', 'ACK preceded lookup; later missing prefix does not identify eviction or continued residency.'
    # A still-pending native job seen at a later callback entry proves absence of
    # a terminal native ACK at the earlier lookup, given this creation lifecycle.
    pending = any(r.get('present_before') is True and
                  isinstance(r.get('pending_count_before'), int) and r['pending_count_before'] > 0 and
                  r.get('entry_time_s', -1) > followup['time'] and
                  isinstance(r.get('step'), int) and r['step'] > followup['scheduler_step']
                  for r in completion['reports'])
    if pending:
        return 'NOT_YET_COMPLETED', 'Same native job is explicitly still pending after lookup; CUDA completion time remains unknown.'
    return 'INSUFFICIENT_EVIDENCE', 'Creation without an ordered terminal ACK or positive pending snapshot is inconclusive.'


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def analyze(path, extra_events=()):
    path = Path(path).resolve(); raw = read_json(path)
    resources = read_json(path.parent/'resources.json', {})
    status = read_json(path.parent/'status.json', {})
    requests = {r['request_id']: r for r in raw.get('requests', [])}
    if len(requests) != len(raw.get('requests', [])):
        raise ValueError('Duplicate request_id would overwrite an arrived request')
    external_ids = [r.get('external_id') or rid for rid, r in requests.items()]
    if len(set(external_ids)) != len(external_ids):
        raise ValueError('Duplicate external_id makes request cost pairing ambiguous')
    work, work_totals = position_work(raw)
    if set(work_totals)-set(requests):
        raise ValueError('Scheduled request is missing from the arrivals table')
    # A recorded scheduler trace with no row for an arrival has zero recorded
    # work; an absent trace has unknown work. Neither request may be dropped.
    work_by_external_id = ({external_id: work_totals.get(rid, dict(
        scheduled_positions=0, unique_positions=0, repeated_positions=0))
        for (rid, _), external_id in zip(requests.items(), external_ids)}
        if 'scheduler_steps' in raw else None)
    ends = [s['end_s'] for s in raw.get('steps', [])]
    observer = raw.get('host_history_observer') or {}
    records = observer.get('records', [])
    observer_issues = []
    if observer.get('enabled') is not True:
        observer_issues.append('host_history_observer_missing_or_disabled')
    if observer.get('errors'):
        observer_issues.append('observer_errors')
    if [r.get('record_index') for r in records] != list(range(len(records))):
        observer_issues.append('record_index_gap_or_duplicate')
    if any(r.get('status') != 'returned' for r in records):
        observer_issues.append('nonreturned_observation')
    layout = resources.get('group_config', [])
    chunk = layout[0].get('tokens_per_chunk') if len(layout) == 1 else None
    bytes_per_chunk = None
    if (chunk and layout[0].get('tokens_per_block') == chunk and
        resources.get('host_blocks') and resources.get('gpu_blocks')):
        hb, gb = resources.get('host_bytes', 0), resources.get('gpu_bytes', 0)
        hn, gn = resources['host_blocks'], resources['gpu_blocks']
        if hb > 0 and hb % hn == 0 and gb % gn == 0 and hb//hn == gb//gn:
            bytes_per_chunk = hb//hn
    events = []; index = {}

    def emit(kind, rid, pointer, **kw):
        e = common_event(path, kind, rid, pointer, **kw)
        events.append(e); return len(events)-1

    preempted = {c['request_id'] for c in raw.get('commits', [])}
    for rid in sorted(preempted):
        for row in work.get(rid, []):
            if row['repeated']:
                for a, b in row['repeated']:
                    emit('RECOMPUTE', rid, row['pointer'], time=row['time_s'], step=row['step'],
                         blocks=[a//chunk, (b+chunk-1)//chunk] if chunk else None,
                         tier='GPU', token_interval=[a, b], time_semantics='schedule_return_not_gpu_execution',
                         logical_block_semantics='blocks_touched; edge blocks may be partial')
    for i, d in enumerate(raw.get('decisions', [])):
        hit = d.get('host_hit_tokens')
        index[('lookup', d['request_id'], d['event'])] = emit('HOST_LOOKUP', d['request_id'],
            f'/decisions/{i}', time=d['decision_s'], step=bisect_left(ends, d['decision_s']),
            blocks=[0, hit//chunk] if chunk and isinstance(hit, int) and hit >= 0 else None,
            tier='Host', hit_tokens=hit, known_tokens=d.get('known_tokens'),
            native_scheduler_step=d.get('scheduler_step'), time_semantics='lookup_entry',
            logical_block_semantics='contiguous ready prefix, not all resident blocks')
    jobs_by_record = {}; all_jobs = []
    for i, record in enumerate(records):
        op = record.get('operation'); ptr = f'/host_history_observer/records/{i}'
        step, time = record.get('step_index'), record.get('time_s')
        if op == 'allocation':
            before = record['before']; after = record.get('after', {})
            index[('alloc', record['record_index'])] = emit('STORE_CURSOR', before['request_id'], ptr,
                time=time, step=step, tier='Host', cursor_before=before.get('groups'),
                cursor_after=after.get('groups'), time_semantics='allocation_callback_entry',
                preemptions=before.get('preemptions'))
        elif op == 'store_builder':
            for bi, before in enumerate(record.get('before', [])):
                rid = before['request_id']
                after = next((a for a in record.get('after', []) if a['request_id'] == rid), {})
                index[('builder', record['record_index'], rid)] = emit('STORE_BUILDER_RETURN', rid, ptr,
                    time=time, step=step, tier='Host', cursor_before=before.get('groups'),
                    cursor_after=after.get('groups'), computed_tokens=before.get('computed_tokens'),
                    scheduled_tokens=before.get('scheduled_tokens'), known_tokens=before.get('known_tokens'),
                    returned_job_count=sum(j['request_id'] == rid for j in record.get('jobs', []))
                                       if 'jobs' in record else None,
                    status=record.get('status'), time_semantics='callback_entry; returned observed, return time unmeasured')
            for ji, job in enumerate(record.get('jobs', [])):
                spans = [r for g in job.get('groups', []) for r in g['chunk_intervals']]
                valid = (len(job.get('groups', [])) == 1 and job['groups'][0]['group_index'] == 0 and
                         job.get('unmatched_key_count') == 0 and size(spans) == job.get('key_count') and
                         sum(b-a for a, b in spans) == size(spans) and
                         size(spans) == job['groups'][0]['chunk_count'])
                byte_count = (job['key_count']*bytes_per_chunk if valid and bytes_per_chunk and
                              job.get('source_block_count') == job['key_count'] else None)
                eid = emit('STORE_CREATE', job['request_id'], f'{ptr}/jobs/{ji}', time=time,
                    step=step, job=job['job_id'], tier='Host',
                    blocks=spans[0] if len(spans) == 1 else None,
                    logical_block_intervals=spans, creation_record_index=record['record_index'],
                    mapping_complete=valid, payload_bytes=byte_count,
                    bytes_basis='returned source blocks × recorded homogeneous layout' if byte_count is not None else None,
                    time_semantics='builder_callback_entry; job returned, transfer start unmeasured')
                j = dict(request_id=job['request_id'], job_id=job['job_id'],
                    creation_record_index=record['record_index'], creation_step=step,
                    time_s=time, logical_block_intervals=spans, mapping_complete=valid,
                    payload_bytes=byte_count, content_version=None, event_index=eid, raw_job=job)
                all_jobs.append(j); jobs_by_record[(record['record_index'], job['job_id'])] = j
        elif op == 'store_completion':
            for bi, before in enumerate(record.get('before', [])):
                after = next((a for a in record.get('after', []) if a['job_id'] == before['job_id']), {})
                created = jobs_by_record.get((before['creation_record_index'], before['job_id']), {})
                spans = created.get('logical_block_intervals')
                emit('STORE_ACK_REPORT', before['request_id'], f'{ptr}/before/{bi}', time=time,
                    step=step, job=before['job_id'], tier='Host',
                    blocks=spans[0] if spans and len(spans) == 1 else None,
                    logical_block_intervals=spans,
                    creation_record_index=before['creation_record_index'], pending_count_before=before.get('pending_count_before'),
                    pending_count_after=after.get('pending_count_after'), removed_by_native=after.get('removed_by_native'),
                    time_semantics='callback_entry; native return observed separately', status=record.get('status'))
    for event in extra_events:
        required = {'request_id', 'job_id', 'kind', 'time', 'clock_domain', 'scheduler_step',
                    'logical_block_interval', 'storage_tier', 'raw_evidence_location'}
        if not required <= event.keys() or event['kind'] != 'EVICT':
            raise ValueError('extra events must be explicit EVICT records with the documented minimal fields')
        events.append(event)
    result = dict(raw_path=str(path), input_status=status.get('status'),
        schema='E.store_coverage_diagnostic.v1', clock_domain=f'episode_monotonic:{path}',
        clock_alignment=raw.get('clock_alignment'), observer_issues=observer_issues,
        layout=dict(tokens_per_logical_block=chunk, bytes_per_block=bytes_per_chunk,
                    supported_single_homogeneous_group=bytes_per_chunk is not None),
        coverage=dict(requests=len(requests), committed_recoveries=len(raw.get('commits', [])),
                      observer_records=len(records), eviction_records=len(extra_events)),
        events=events, episodes=[], insufficient_unlinked_commits=[], no_full_block_gap_commits=[],
        costs=dict(full_run_store_reported_bytes=sum(t.get('store', {}).get('bytes', 0) for t in raw.get('transfers', []))
                   if 'transfers' in raw else None,
                   full_run_load_reported_bytes=sum(t.get('load', {}).get('bytes', 0) for t in raw.get('transfers', []))
                   if 'transfers' in raw else None,
                   explicit_eviction_events=list(extra_events) if extra_events else None,
                   peer_evictions_attributed_to_repair=None,
                   peer_recomputation_attributed_to_eviction=None,
                   work_by_external_id=work_by_external_id,
                   bytes_scope='All transfer reports including drain; observer jobs cover preempted requests only.'),
        service=dict(arrived=len(requests), completed=sum(r.get('completed') is True for r in requests.values()),
            unfinished=sum(r.get('completed') is not True for r in requests.values()),
            mean_completion_s=(sum(r['completion_s']-r['arrival_s'] for r in requests.values())/len(requests)
                               if requests and all(r.get('completed') for r in requests.values()) else None),
            all_complete_s=raw.get('all_complete_s'), output_tokens=sum(len(r.get('output_token_ids', [])) for r in requests.values())),
        executed_cursor_repairs=[r for r in (raw.get('store_replay') or {}).get('records', []) if r.get('executed')],
        claim='Offline event diagnosis; no new service execution, inferred latency saving, or persistent residency claim.')
    for ci, commit in enumerate(raw.get('commits', [])):
        rid = commit['request_id']; h = commit.get('host_hit_tokens'); k = commit.get('known_tokens')
        if not chunk or not isinstance(h, int) or not isinstance(k, int) or h < 0 or h % chunk:
            result['insufficient_unlinked_commits'].append(dict(request_id=rid, event=commit['event'],
                classification='INSUFFICIENT_EVIDENCE',
                reason='Unsupported layout or nonnumeric/unaligned Host prefix.', evidence=f'/commits/{ci}'))
            continue
        if h >= k//chunk*chunk:
            result['no_full_block_gap_commits'].append(dict(request_id=rid, event=commit['event'],
                reason='Recorded Host prefix covers every full known logical block.', evidence=f'/commits/{ci}'))
            continue
        roots = [r for r in records if r.get('operation') == 'allocation' and
                 r.get('before', {}).get('request_id') == rid and
                 (r.get('lookup') or {}).get('event') == commit['event']]
        if len(roots) != 1:
            result['insufficient_unlinked_commits'].append(dict(request_id=rid, event=commit['event'],
                classification='INSUFFICIENT_EVIDENCE', reason='Missing or ambiguous allocation/cursor observation.',
                evidence=f'/commits/{ci}'))
            continue
        root = roots[0]; preemption = commit['preemptions']; first = h//chunk
        next_decision = next((d for d in raw.get('decisions', []) if d['request_id'] == rid and
                             d['decision_s'] > commit['allocation_s'] and d['preemptions'] > preemption), None)
        follow = None
        if next_decision is not None:
            hit = next_decision.get('host_hit_tokens')
            follow = dict(time=next_decision['decision_s'], scheduler_step=bisect_left(ends, next_decision['decision_s']),
                clock_domain=result['clock_domain'], hit_tokens=hit,
                hit_chunks=hit//chunk if isinstance(hit, int) and hit >= 0 and hit % chunk == 0 else None,
                known_tokens=next_decision.get('known_tokens'), event=next_decision['event'],
                event_index=index.get(('lookup', rid, next_decision['event'])))
        # Stop before the next preemption, not just the next successful allocation.
        next_preempt = next((p['time_s'] for p in raw.get('preemptions', []) if p['request_id'] == rid and
                             p['time_s'] > commit['allocation_s']), None)
        stop = min(x for x in [next_preempt, follow['time'] if follow else None, float('inf')] if x is not None)
        native = [w for w in work.get(rid, []) if commit['allocation_s'] <= w['time_s'] < stop]
        repeats = intervals([span for w in native for span in w['repeated']])
        builders, issues = [], list(observer_issues)
        groups = root['before'].get('groups', [])
        if (root.get('status') != 'returned' or len(groups) != 1 or groups[0].get('tokens_per_chunk') != chunk or
            root['before'].get('computed_tokens') != 0):
            issues.append('unsupported_allocation_state_or_layout')
        root_lookup = root.get('lookup') or {}
        if any(root_lookup.get(key) != commit.get(key) for key in
               ['preemptions', 'known_tokens', 'generated_tokens', 'host_hit_tokens', 'action']):
            issues.append('commit_allocation_mismatch')
        for record in records:
            if record.get('operation') != 'store_builder' or not commit['allocation_s'] <= record['time_s'] < stop:
                continue
            before = next((b for b in record.get('before', []) if b['request_id'] == rid and b['preemptions'] == preemption), None)
            if before is None:
                continue
            link = allocation_link(records, root, record, before)
            if not link['valid']:
                issues.extend(link['issues'])
            if record.get('status') != 'returned' or 'jobs' not in record:
                issues.append('builder_return_not_observed')
            builders.append((record, before))
        builder_steps = {b['step_index'] for b, _ in builders}
        if not native or any(w['step'] not in builder_steps for w in native):
            issues.append('scheduled_work_missing_builder_observation')
        chosen = []
        for record, before in builders:
            for raw_job in record.get('jobs', []):
                if raw_job['request_id'] != rid:
                    continue
                job = dict(jobs_by_record[(record['record_index'], raw_job['job_id'])])
                job.pop('raw_job')
                old_follow = dict(step=follow['scheduler_step'], decision_s=follow['time']) if follow else None
                job['completion'] = completion_view(records, raw_job, record['record_index'], rid, old_follow)
                ack_steps = [a['step'] for a in job['completion']['reports'] if a['acknowledged_before_next_lookup'] is True]
                job['ack_step_end_s'] = min((ends[s] for s in ack_steps if s < len(ends)), default=None)
                if not job['mapping_complete']:
                    issues.append('incomplete_job_mapping')
                chosen.append(job)
        full_block = size(intersect(repeats, [[h, h+chunk]])) == chunk
        category, reason = classify_gap(first, follow, chosen, not issues, full_block, extra_events)
        cursor_before = groups[0].get('next_stored_chunk_idx') if len(groups) == 1 else None
        after_groups = root.get('after', {}).get('groups', [])
        cursor_after = after_groups[0].get('next_stored_chunk_idx') if len(after_groups) == 1 else None
        total_bytes = sum(j['payload_bytes'] for j in chosen) if all(j['payload_bytes'] is not None for j in chosen) and not issues else None
        result['episodes'].append(dict(request_id=rid, external_id=requests.get(rid, {}).get('external_id'),
            event=commit['event'], preemptions=preemption, action=commit.get('actual_action'),
            allocation_s=commit['allocation_s'],
            allocation_event_index=index[('alloc', root['record_index'])],
            classification=category, reason=reason, evidence_issues=sorted(set(issues)),
            first_nonready_logical_block=[first, first+1],
            initial_contiguous_hit_tokens=h, known_tokens=k,
            tail_after_first_nonready_block_is_unknown=True,
            cursor_before=cursor_before, cursor_after=cursor_after,
            recomputed_token_intervals=repeats, first_nonready_block_fully_recomputed=full_block,
            builder_event_indices=[index[('builder', b['record_index'], rid)] for b, _ in builders],
            cursor_skips_recomputed_head_in_all_builders=bool(builders) and all(
                len(b.get('groups', [])) == 1 and b['groups'][0]['next_stored_chunk_idx'] > first
                for _, b in builders),
            jobs=chosen, new_job_payload_bytes_in_window=total_bytes,
            next_host_lookup=follow,
            scheduled_positions_in_window=sum(w['span'][1]-w['span'][0] for w in native),
            repeated_positions_in_window=sum(size(w['repeated']) for w in native),
            window_end_s=None if stop == float('inf') else stop,
            window_scope='successful recovery through next preemption (or episode end); no global never-stored claim'))
    result['classification_counts'] = dict(Counter(e['classification'] or 'COVERED_AT_NEXT_LOOKUP' for e in result['episodes']))
    result['classification_counts']['INSUFFICIENT_UNLINKED_COMMITS'] = len(result['insufficient_unlinked_commits'])
    result['classification_counts']['NO_FULL_BLOCK_GAP'] = len(result['no_full_block_gap_commits'])
    compact_evidence(result)
    return result


def compact_evidence(result):
    """Keep cited evidence, not another full dump of terminal decode callbacks."""
    events = result['events']
    keep = {i for i, e in enumerate(events) if e['kind'] in ('HOST_LOOKUP', 'EVICT')}
    lifecycles = set(); windows = []
    for episode in result['episodes']:
        keep.add(episode['allocation_event_index'])
        episode['observed_builder_count'] = len(episode['builder_event_indices'])
        episode['observed_store_job_count'] = len(episode['jobs'])
        if episode['next_host_lookup'] is None:
            # Keep aggregate work and payload costs; raw remains the full record.
            episode['jobs'] = []; episode['builder_event_indices'] = []
            episode['detail_scope'] = 'No later lookup; terminal-window totals retained, callback detail left in raw.'
            continue
        windows.append((episode['request_id'], episode['allocation_s'], episode['window_end_s']))
        keep.update(episode['builder_event_indices'])
        for job in episode['jobs']:
            keep.add(job['event_index'])
            lifecycles.add((job['request_id'], job['job_id'], job['creation_record_index']))
    for i, event in enumerate(events):
        if event['kind'] == 'STORE_ACK_REPORT' and (
            event['request_id'], event['job_id'], event['creation_record_index']) in lifecycles:
            keep.add(i)
        if event['kind'] == 'RECOMPUTE' and any(event['request_id'] == rid and
                start <= event['time'] < (stop if stop is not None else float('inf'))
                for rid, start, stop in windows):
            keep.add(i)
    ordered = sorted(keep); remap = {old: new for new, old in enumerate(ordered)}
    for episode in result['episodes']:
        episode['allocation_event_index'] = remap[episode['allocation_event_index']]
        episode['builder_event_indices'] = [remap[i] for i in episode['builder_event_indices']]
        if episode['next_host_lookup'] is not None:
            i = episode['next_host_lookup']['event_index']
            episode['next_host_lookup']['event_index'] = remap[i] if i is not None else None
        for job in episode['jobs']:
            job['event_index'] = remap[job['event_index']]
    result['events'] = [events[i] for i in ordered]
    result['event_scope'] = ('Lookup observations and evidence cited by windows with a later lookup; '
                             'terminal-window work/STORE totals retained; original raw is not filtered or changed.')


def compare(a, b):
    wa, wb = a['costs']['work_by_external_id'], b['costs']['work_by_external_id']
    invalid = dict(status='INCOMPARABLE', all_request_position_delta=None,
                   peer_repeated_position_delta=None, service=[a['service'], b['service']])
    if wa is None or wb is None:
        return dict(invalid, reason='Missing scheduler observations; recorded request work is unknown.')
    if len(wa) != a['service']['arrived'] or len(wb) != b['service']['arrived']:
        return dict(invalid, reason='Work table does not cover every arrived request.')
    if set(wa) != set(wb):
        return dict(invalid, reason='Different external request sets; an intersection is not a whole-service comparison.',
                    only_in_first=sorted(set(wa)-set(wb)), only_in_second=sorted(set(wb)-set(wa)))
    ids = sorted(wa)
    work_delta = {rid: {k: wb[rid][k]-wa[rid][k] for k in wa[rid]} for rid in ids}
    repairs = b['executed_cursor_repairs']
    targets = {r.get('request_id') for r in repairs}
    targets_external = {e['external_id'] for e in b['episodes'] if e['request_id'] in targets}
    windows = []
    for repair in repairs:
        after = next((e for e in b['episodes'] if e['request_id'] == repair['request_id'] and
                      e['event'] == repair['event']), None)
        if after is None:
            continue
        candidates = [e for e in a['episodes'] if e['external_id'] == after['external_id'] and
                      e['preemptions'] == after['preemptions']]
        before = candidates[0] if len(candidates) == 1 else None
        windows.append(dict(external_id=after['external_id'], before_event=before['event'] if before else None,
            after_event=after['event'], before_new_job_bytes=before['new_job_payload_bytes_in_window'] if before else None,
            after_new_job_bytes=after['new_job_payload_bytes_in_window'],
            next_host_hit_tokens=[before['next_host_lookup']['hit_tokens'] if before and before['next_host_lookup'] else None,
                                  after['next_host_lookup']['hit_tokens'] if after['next_host_lookup'] else None],
            before_jobs=before['jobs'] if before else None, after_jobs=after['jobs']))
    def delta(x, y):
        return y-x if x is not None and y is not None else None
    return dict(status='COMPARABLE_RECORDED_WORK',
        scope='Observed whole-trace second minus first, not simulated latency or per-event causal estimate.',
        same_external_id_set=set(wa) == set(wb), executed_cursor_repairs=len(repairs),
        repair_windows=windows, changed_position_work={k:v for k,v in work_delta.items() if any(v.values())},
        all_request_position_delta={k:sum(v[k] for v in work_delta.values()) for k in ('scheduled_positions','unique_positions','repeated_positions')},
        peer_repeated_position_delta=(sum(v['repeated_positions'] for k,v in work_delta.items() if k not in targets_external)
                                      if targets_external else None),
        peer_evictions_attributed_to_repair=None, peer_recompute_attributed_to_eviction=None,
        full_run_store_bytes_delta=delta(a['costs']['full_run_store_reported_bytes'], b['costs']['full_run_store_reported_bytes']),
        full_run_load_bytes_delta=delta(a['costs']['full_run_load_reported_bytes'], b['costs']['full_run_load_reported_bytes']),
        mean_completion_s_delta=delta(a['service']['mean_completion_s'], b['service']['mean_completion_s']),
        all_complete_s_delta=delta(a['service']['all_complete_s'], b['service']['all_complete_s']),
        service=[a['service'],b['service']],
        limitations=['ID pairing alone does not prove matched causal pre-state; use the original intervention audit.',
                     'Zero observed peer work delta does not establish no evictions; eviction attribution is unobserved.',
                     'Scheduled positions are executed scheduler work, not measured GPU time.'])


def cpu_fixture(path):
    """Read an existing native CPU counterexample; do not run or invent transfers."""
    path = Path(path).resolve(); raw = read_json(path); on = raw['arms']['on']; off = raw['arms']['off']
    removed = set(on['initial']['residents'])-set(on['final']['residents'])
    added = set(on['final']['residents'])-set(on['initial']['residents'])
    return dict(source=str(path), evidence_layer='existing native CPU fixture, not service trace',
        submitted_chunks=sum(len(t['keys']) for t in on['submitted_transfers']), added_payload_bytes=None,
        removed_resident_keys=sorted(removed), added_resident_keys=sorted(added),
        prefixes_off=off['final']['prefix_chunks'], prefixes_on=on['final']['prefix_chunks'],
        peer_later_repeated_positions=None, service_latency_s=None,
        events=[common_event(path, 'CPU_RESIDENCY_SNAPSHOT_DIFFERENCE', None, '/arms/on',
                            tier='Host', removed_keys=sorted(removed), added_keys=sorted(added))],
        limitation='Observed ready resident removal, not recorded prior STORE job ACK; no payload layout or future request execution. B prefix loss is not loss of every B block.')


def markdown(report):
    out = ['# STORE 覆盖诊断（自动生成）', '',
        '仅 CPU 读取已有证据；无 GPU、无新服务动作。完成回报不证明持续驻留，CPU 回放不证明时延收益。', '',
        '| 输入 | 请求/恢复 | 分类 |', '|---|---:|---|']
    for t in report['traces']:
        out.append(f"| {Path(t['raw_path']).parent.parent.parent.name}/{Path(t['raw_path']).parent.name} | "
                   f"{t['coverage']['requests']}/{t['coverage']['committed_recoveries']} | `{json.dumps(t['classification_counts'])}` |")
    for t in report['traces']:
        out += ['', f"## {t['raw_path']}", '', f"观察缺口：`{json.dumps(t['observer_issues'])}`。",
                '全程 STORE/LOAD 报告字节：'+str(t['costs']['full_run_store_reported_bytes'])+' / '+str(t['costs']['full_run_load_reported_bytes'])+'。', '',
                '| 请求/事件 | 判断 | 游标前→后 | 窗口新建字节 | 后续 Host hit |', '|---|---|---:|---:|---:|']
        for e in t['episodes']:
            follow=e['next_host_lookup']
            out.append(f"| {e['external_id']}/{e['event']} | {e['classification'] or 'COVERED_AT_NEXT_LOOKUP'} | "
                f"{e['cursor_before']}→{e['cursor_after']} | {e['new_job_payload_bytes_in_window']} | {follow['hit_tokens'] if follow else None} |")
    if (report.get('comparison') or {}).get('status') == 'INCOMPARABLE':
        out += ['', '## 对照不可比较', '', report['comparison']['reason'],
                '逐臂证据仍保留；不报告请求交集为全体收益。']
    elif report.get('comparison'):
        c=report['comparison']; out += ['', '## 既有干预对照（第二份减第一份）', '',
            f"实际游标干预 {c['executed_cursor_repairs']} 次。全部请求重复位置差 {c['all_request_position_delta']['repeated_positions']}；同伴重复位置差 {c['peer_repeated_position_delta']}。",
            f"全程 STORE 字节差 {c['full_run_store_bytes_delta']}；LOAD 字节差 {c['full_run_load_bytes_delta']}。",
            f"已有实测完成均值差 {c['mean_completion_s_delta']} s；排空完成时刻差 {c['all_complete_s_delta']} s。CPU 分析未生成新的时延收益。",
            'GPU 轨迹未记录可归因的同伴驱逐；该项及由驱逐造成的后续重算为 null，不是零。']
        for w in c['repair_windows']:
            out.append(f"补写窗口 {w['external_id']}：新建任务字节 {w['before_new_job_bytes']}→{w['after_new_job_bytes']}；下一 Host hit {w['next_host_hit_tokens']}。")
    if report.get('cpu_fixture'):
        f=report['cpu_fixture'];out += ['', '## 既有 CPU 驱逐反例', '',
            f"新增 {f['submitted_chunks']} chunk；字节未知。移除驻留 key `{f['removed_resident_keys']}`；前缀 {f['prefixes_off']}→{f['prefixes_on']}。",
            '没有后续请求执行，重复计算与时延为 null；没有原始 STORE 完成事件，不能升级为完整任务生命周期证据。']
    out += ['', '## 判定边界', '',
        '- NOT_CREATED：仅已覆盖窗口内该块无任务；不是全局从未写入或上游 bug。',
        '- NOT_YET_COMPLETED：原生任务尚未确认完成，不等于 CUDA 数据未传完。',
        '- COMPLETED_THEN_EVICTED：需要同一 STORE 生命周期的显式驱逐证据；历史 GPU 输入不具备。',
        '- INSUFFICIENT_EVIDENCE：缺观测、顺序或映射；不得把缺记录写成未创建。',
        '- COVERED_AT_NEXT_LOOKUP：只证明该次前缀命中覆盖该块，不证明持续驻留。',
        '', '事实与解释：任务、游标、调度区间和后续前缀为事件证据；游标抑制枚举为源码支持的局部解释。重复工作差不能自动归因于同伴驱逐，更不能直接换算墙钟收益。', '']
    return '\n'.join(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw', nargs='+', type=Path);p.add_argument('--out', type=Path, required=True)
    p.add_argument('--compare', action='store_true');p.add_argument('--extra-events', type=Path)
    p.add_argument('--cpu-fixture', type=Path)
    args=p.parse_args()
    if args.extra_events and len(args.raw) != 1:
        p.error('--extra-events requires exactly one raw input, preventing cross-run joins')
    if args.compare and len(args.raw) < 2:
        p.error('--compare requires at least two raw inputs')
    extra=read_json(args.extra_events) if args.extra_events else []
    traces=[analyze(path, extra) for path in args.raw]
    report=dict(schema='E.store_coverage_report.v1', traces=traces,
                comparison=compare(*traces[:2]) if args.compare else None,
                cpu_fixture=cpu_fixture(args.cpu_fixture) if args.cpu_fixture else None)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    args.out.with_suffix('.md').write_text(markdown(report))
    print(json.dumps(dict(output=str(args.out), counts=[t['classification_counts'] for t in traces]), ensure_ascii=False))


if __name__ == '__main__':
    main()

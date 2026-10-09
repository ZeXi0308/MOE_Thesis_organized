#!/usr/bin/env python3
"""Read-only CPU diagnosis for the existing B collectors (stdlib only)."""
import argparse
import bisect
import collections
import hashlib
import json
import math
from pathlib import Path
import time

VERSION = 'b-recovery-timeline-v1'
FILES = ('raw.json', 'recovery-order.json', 'capacity-handoff.json', 'source-handoff.json')
LIFECYCLE = ('job_created', 'ready', 'submit_begin', 'submit_end', 'job_completed', 'ack_retired')


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def ref(e):
    return e['event_id'] if e else None


def unknown(reason, **fields):
    return dict(status='unknown', reason=reason, **fields)


def interval(left, right):
    """Only unique, aligned, ordered observations define an elapsed interval."""
    out = dict(duration_s=None, from_events=[ref(e) for e in left], to_events=[ref(e) for e in right],
               causal_attribution='unknown')
    if not left or not right:
        return unknown('missing_endpoint', **out)
    if len(left) != 1 or len(right) != 1:
        return unknown('ambiguous_repeated_endpoint', **out)
    a, b = left[0], right[0]
    if a['clock_domain'] is None or a['clock_domain'] != b['clock_domain']:
        return unknown('unaligned_clock', **out)
    if not number(a['time_s']) or not number(b['time_s']):
        return unknown('missing_time', **out)
    if b['time_s'] < a['time_s']:
        return unknown('nonmonotonic_endpoints', **out)
    out['duration_s'] = b['time_s'] - a['time_s']
    return dict(status='observed', **out)


def first_observed(events, kind, anchor):
    rows = [e for e in events if e['kind'] == kind]
    if not rows:
        return []
    # Do not select a convenient subset if another occurrence cannot be aligned.
    if any(e['clock_domain'] != anchor['clock_domain'] or not number(e['time_s']) for e in rows):
        return rows
    t = min(e['time_s'] for e in rows)
    return [e for e in rows if e['time_s'] == t]


def normalize(documents, run_id):
    raw = documents.get('raw.json', {})
    origin = raw.get('measurement_origin_perf_counter_s')
    domain = 'host_perf_counter:' + run_id
    events, notes, counts = [], [], {}
    map_values = collections.defaultdict(set)
    for internal, source in raw.get('internal_to_source', {}).items():
        map_values[internal].add(source)
    for request in raw.get('requests', []):
        if request.get('internal_request_id') and request.get('request_id'):
            map_values[request['internal_request_id']].add(request['request_id'])

    def emit(file, index, kind, t, row, pointer=None, request=None, time_field=None):
        doc = documents[file]
        declared = row.get('clock_domain', doc.get('clock_domain'))
        known_host = 'perf_counter' in doc.get('clock', '')
        clock = declared if declared is not None else domain if known_host else None
        attrs = {k: v for k, v in row.items() if k not in (
            'kind', 'request', 'request_id', 'job_id', 'clock_domain', 'scheduler_step',
            'logical_block_range', 'storage_layer', 'host_perf_s')}
        rid = row.get('request') or row.get('request_id') or request
        mapping = map_values.get(rid, set())
        is_store = row.get('is_store')
        storage = row.get('storage_layer')
        if storage is None and is_store is not None:
            storage = 'GPU->Host' if is_store else 'Host->GPU'
        if storage is None and file == 'source-handoff.json':
            storage = 'Host'
        if storage is None and file == 'capacity-handoff.json':
            storage = 'GPU'
        pointer = pointer or '/events/' + str(index)
        eid = file + '#' + pointer
        if time_field:
            eid += ':' + time_field
        event = dict(event_id=eid, request_id=rid, source_request_id=next(iter(mapping)) if len(mapping) == 1 else None,
            job_id=row.get('job_id'), kind=kind, time_s=t if number(t) else None,
            clock_domain=clock, scheduler_step=row.get('scheduler_step'),
            logical_block_range=row.get('logical_block_range'), storage_layer=storage,
            evidence=dict(file=file, json_pointer=pointer, time_field=time_field or 'host_perf_s'),
            attributes=attrs)
        events.append(event)
        return event

    for file in FILES[1:]:
        doc = documents.get(file)
        if doc is None:
            notes.append(file + ': missing; related intervals are unknown')
            continue
        counts[file] = dict(collections.Counter(e.get('kind', 'unknown') for e in doc.get('events', [])))
        selected = doc.get('selected_request') if file == 'source-handoff.json' else None
        for i, row in enumerate(doc.get('events', [])):
            kind = row.get('kind', 'unknown')
            if file == 'recovery-order.json':
                # This is a LOAD/recovery tool, not an export of all STORE/reorder traffic.
                if row.get('is_store') is True or kind in ('reorder', 'flush_dependency', 'wait_begin', 'wait_end'):
                    continue
            if file == 'capacity-handoff.json' and kind in ('preempt', 'allocate'):
                emit(file, i, kind + '_begin', row.get('begin_host_perf_s'), row, time_field='begin_host_perf_s')
                emit(file, i, kind + '_end', row.get('end_host_perf_s'), row, time_field='end_host_perf_s')
            else:
                emit(file, i, kind, row.get('host_perf_s'), row, request=selected)

    # Job IDs are scoped to this input run. Conflicts are retained, never guessed away.
    by_job = collections.defaultdict(list)
    for e in events:
        if e['job_id'] is not None:
            by_job[e['job_id']].append(e)
    jobs = []
    for jid, rows in by_job.items():
        requests = {e['request_id'] for e in rows if e['request_id'] is not None}
        directions = {e['attributes'].get('is_store') for e in rows if e['attributes'].get('is_store') is not None}
        association = 'unique' if len(requests) == 1 and directions == {False} else 'unknown'
        rid = next(iter(requests)) if len(requests) == 1 else None
        for e in rows:
            if e['request_id'] is None and rid is not None:
                e['request_id'] = rid
                e['attributes']['request_association'] = 'unique_job_identity'
        kinds = {k: [e for e in rows if e['kind'] == k] for k in LIFECYCLE}
        phases = {a + '_to_' + b: interval(kinds[a], kinds[b]) for a, b in (
            ('job_created', 'ready'), ('ready', 'submit_begin'), ('submit_begin', 'submit_end'),
            ('submit_begin', 'job_completed'), ('job_completed', 'ack_retired'))}
        if association != 'unique':
            phases = {name: dict(p, status='unknown', reason='missing_or_conflicting_job_identity', duration_s=None)
                      for name, p in phases.items()}
        jobs.append(dict(job_id=jid, request_id=rid, association=association,
            candidate_requests=sorted(requests), event_ids=[ref(e) for e in rows],
            occurrence_counts={k: len(v) for k, v in kinds.items()}, phases=phases,
            gpu_completion_time=unknown('host_poll_report_is_not_absolute_GPU_completion'),
            gpu_elapsed_samples=[dict(value_s=e['attributes'].get('gpu_elapsed_s'), evidence=ref(e))
                                 for e in kinds['job_completed']], episode_candidates=[]))

    # Raw preemption hooks are corroboration; they must not create duplicate episodes.
    raw_clock = domain if number(origin) else None
    for i, row in enumerate(raw.get('preemption_events', [])):
        for kind, field in (('raw_preempt_begin', 'method_entered_s'), ('raw_preempt_end', 'method_returned_s')):
            relative = row.get(field)
            e = emit('raw.json', i, kind, origin + relative if number(origin) and number(relative) else None,
                dict(row, request=row.get('internal_request_id'), clock_domain=raw_clock),
                pointer='/preemption_events/' + str(i), time_field=field)
            e['source_request_id'] = row.get('request_id')
    # Create only the adjacent output observations needed by recovery episodes, not a second token dump.
    requests = collections.defaultdict(list)
    for i, r in enumerate(raw.get('requests', [])):
        if r.get('internal_request_id'):
            requests[r['internal_request_id']].append((i, r))
        else:
            for rid, sources in map_values.items():
                if sources == {r.get('request_id')}:
                    requests[rid].append((i, r))
    return events, jobs, requests, notes, counts, origin, domain


def analyze_documents(documents, run_id='fixture'):
    events, jobs, requests, notes, counts, origin, domain = normalize(documents, run_id)
    raw = documents.get('raw.json', {})
    by_request = collections.defaultdict(list)
    by_id = {ref(e): e for e in events}
    for e in events:
        if e['request_id'] is not None:
            by_request[e['request_id']].append(e)
    # Prefer one collector per request; mirrors remain visible as corroborating evidence.
    preempts = []
    for rid, rows in by_request.items():
        preferred = next((k for k in ('preempt', 'preempt_begin', 'raw_preempt_begin') if any(e['kind'] == k for e in rows)), None)
        preempts.extend(e for e in rows if e['kind'] == preferred)
    episodes = []
    for n, pre in enumerate(preempts):
        rid, t, clock = pre['request_id'], pre['time_s'], pre['clock_domain']
        recs = requests.get(rid, [])
        outputs, output_note = {}, None
        if len(recs) != 1:
            output_note = 'missing_or_ambiguous_request_mapping'
        elif clock != domain or not number(t) or not number(origin):
            output_note = 'unaligned_output_clock'
        else:
            request_index, request = recs[0]
            ts = request.get('token_times_s', [])
            if any(not number(x) for x in ts) or any(b < a for a, b in zip(ts, ts[1:])):
                output_note = 'nonmonotonic_or_missing_output_time'
            else:
                ix = bisect.bisect_right(ts, t - origin)
                for label, j in (('previous_output', ix - 1), ('next_output', ix)):
                    if 0 <= j < len(ts):
                        eid = 'raw.json#/requests/' + str(request_index) + '/token_times_s/' + str(j)
                        e = dict(event_id=eid, request_id=rid, source_request_id=request.get('request_id'),
                            job_id=None, kind='output_receipt', time_s=origin + ts[j], clock_domain=domain,
                            scheduler_step=None, logical_block_range=None, storage_layer=None,
                            evidence=dict(file='raw.json', json_pointer=eid.split('#', 1)[1], time_field='value'),
                            attributes=dict(token_index=j, semantics='host receipt; not GPU completion'))
                        if eid not in by_id:
                            events.append(e); by_id[eid] = e
                        outputs[label] = e
        nextout = outputs.get('next_output')
        observed_end = origin + raw['observation_end_s'] if number(origin) and number(raw.get('observation_end_s')) else None
        end = nextout['time_s'] if nextout else observed_end if clock == domain else None
        aligned = clock is not None and number(t) and number(end)
        rows = by_request[rid]
        related = [e for e in rows if aligned and e['clock_domain'] == clock and number(e['time_s']) and t <= e['time_s'] <= end]
        unaligned = [ref(e) for e in rows if e['clock_domain'] != clock or not number(e['time_s'])]
        sched_kind = 'scheduled' if any(e['kind'] == 'scheduled' for e in related) else 'resumed_schedule'
        schedule = first_observed(related, sched_kind, pre)
        # A new preemption before the next output can create overlapping output windows.
        # Capacity collection nevertheless ends at the first resumed schedule; never
        # label the unobserved running interval between episodes as a retry gap.
        cap_schedule = first_observed(related, 'resumed_schedule', pre)
        cap_end = cap_schedule[0]['time_s'] if len(cap_schedule) == 1 else end
        allocations = [e for e in related if e['kind'] == 'allocate_end'
                       and number(cap_end) and e['time_s'] <= cap_end]
        allocations.sort(key=lambda e: e['time_s'])
        failures = [e for e in allocations if e['attributes'].get('success') is False]
        successes = [e for e in allocations if e['attributes'].get('success') is True]
        # Directly observed callback spans, not duration of a supposed capacity bottleneck.
        retry_gaps = []
        for a, b in zip(allocations, allocations[1:]):
            if a['attributes'].get('success') is not False:
                continue
            begin_id = ref(b).rsplit(':', 1)[0] + ':begin_host_perf_s'
            begin = by_id.get(begin_id)
            gap = interval([a], [begin] if begin else [])
            descriptor = a['attributes'].get('descriptor', {})
            gap.update(attempt_after=a['attributes'].get('attempt'), attempt_before=b['attributes'].get('attempt'),
                       reason='no_recorded_allocator_retry_between_these_observations',
                       continuous_capacity_shortage='unknown',
                       failed_snapshot_shortfall_blocks=descriptor.get('shortfall_blocks') if descriptor.get('exact') else None)
            retry_gaps.append(gap)
        phases = dict(preempt_to_next_output=interval([pre], [nextout] if nextout else []),
            previous_to_next_output=interval([outputs['previous_output']] if 'previous_output' in outputs else [], [nextout] if nextout else []),
            preempt_to_first_lookup=interval([pre], first_observed(related, 'lookup', pre)),
            preempt_to_first_schedule=interval([pre], schedule),
            first_schedule_to_next_output=interval(schedule, [nextout] if nextout else []),
            first_failed_to_first_successful_allocation=interval(failures[:1], successes[:1]))
        if output_note:
            phases['preempt_to_next_output'] = unknown(output_note, duration_s=None, causal_attribution='unknown')
        handoffs = [e for e in related if e['kind'] == 'native_handoff']
        source_observations = [dict(evidence=ref(e), external_tokens=e['attributes'].get('external_tokens'),
                                    load_jobs=e['attributes'].get('load_jobs')) for e in handoffs]
        episode = dict(episode_id='episode-' + str(n + 1), request_id=rid,
            source_request_id=pre.get('source_request_id'), preempt_event=ref(pre),
            previous_output_event=ref(outputs.get('previous_output')), next_output_event=ref(nextout),
            output_association='observed' if nextout else 'unknown', output_unknown_reason=output_note or (None if nextout else 'no_later_output_observed'),
            event_ids=[ref(e) for e in related], unaligned_event_ids=unaligned, phases=phases,
            allocations=dict(attempts=len(allocations), failed=len(failures), successful=len(successes),
                event_ids=[ref(e) for e in allocations], retry_gaps=retry_gaps,
                coverage=documents.get('capacity-handoff.json', {}).get('scope', 'unknown; no every-attempt collector')),
            source_handoffs=source_observations, load_job_ids=[], ambiguous_load_job_ids=[],
            unassigned_intervals=[], overlapping_episode_ids=[],
            _start=t, _end=end, _clock=clock)
        episodes.append(episode)

    for job in jobs:
        rows = [by_id[x] for x in job['event_ids']]
        # created is the preferred epoch anchor; ready is only a fallback, never a future ID guess.
        anchors = [e for e in rows if e['kind'] == 'job_created'] or [e for e in rows if e['kind'] == 'ready']
        matches = []
        for ep in episodes:
            if ep['request_id'] not in job['candidate_requests']:
                continue
            if any(number(ep['_start']) and number(ep['_end']) and a['clock_domain'] == ep['_clock']
                   and number(a['time_s']) and ep['_start'] <= a['time_s'] <= ep['_end'] for a in anchors):
                matches.append(ep)
        job['episode_candidates'] = [e['episode_id'] for e in matches]
        anchors_aligned = bool(anchors) and len({e['clock_domain'] for e in anchors}) == 1 and all(e['clock_domain'] is not None and number(e['time_s']) for e in anchors)
        all_anchors_in_one = len(matches) == 1 and all(
            a['clock_domain'] == matches[0]['_clock'] and number(a['time_s'])
            and matches[0]['_start'] <= a['time_s'] <= matches[0]['_end'] for a in anchors)
        unique = len(matches) == 1 and job['association'] == 'unique' and anchors_aligned and all_anchors_in_one
        job['episode_association'] = 'unique' if unique else 'unknown'
        for ep in matches:
            ep['load_job_ids' if unique else 'ambiguous_load_job_ids'].append(job['job_id'])
        if unique:
            ep = matches[0]; pre = by_id[ep['preempt_event']]
            ready = [e for e in rows if e['kind'] == 'ready']
            ack = [e for e in rows if e['kind'] == 'ack_retired']
            ep_rows = [by_id[x] for x in ep['event_ids']]
            sk = 'scheduled' if any(e['kind'] == 'scheduled' for e in ep_rows) else 'resumed_schedule'
            job['phases']['preempt_to_ready'] = interval([pre], ready)
            job['phases']['ack_to_first_schedule'] = interval(ack, first_observed(ep_rows, sk, pre))

    for ep in episodes:
        if number(ep['_start']) and number(ep['_end']):
            ep['overlapping_episode_ids'] = [other['episode_id'] for other in episodes if other is not ep
                and other['request_id'] == ep['request_id'] and other['_clock'] == ep['_clock']
                and number(other['_start']) and number(other['_end'])
                and max(ep['_start'], other['_start']) <= min(ep['_end'], other['_end'])]
        # Timeline intervals are observations only. Never sum overlapping job spans as latency causes.
        points = [by_id[ep['preempt_event']]] + [by_id[x] for x in ep['event_ids']]
        if ep['next_output_event']:
            points.append(by_id[ep['next_output_event']])
        points = [p for p in points if number(p['time_s']) and p['clock_domain'] == ep['_clock']]
        groups = collections.defaultdict(list)
        for p in points:
            groups[p['time_s']].append(p)
        ordered = sorted(groups)
        for a, b in zip(ordered, ordered[1:]):
            ep['unassigned_intervals'].append(dict(start_s=a, end_s=b, duration_s=b-a,
                from_events=list(dict.fromkeys(ref(e) for e in groups[a])),
                to_events=list(dict.fromkeys(ref(e) for e in groups[b])),
                cause='unknown', interpretation='elapsed host timeline; ordering alone is not causality'))
    for ep in episodes:
        for key in ('_start', '_end', '_clock'):
            ep.pop(key)

    def gap(ep):
        p = ep['phases']['previous_to_next_output']
        return p.get('duration_s') if p.get('duration_s') is not None else -1
    ranked = sorted(episodes, key=lambda ep: (-gap(ep), ep['episode_id']))
    # Keep every normalized recovery/LOAD occurrence, including orphans and repeats.
    job_summary = collections.Counter(j['episode_association'] for j in jobs)
    return dict(version=VERSION, run_id=run_id, input_files_present=list(documents), notes=notes,
        semantics=dict(selection='largest observed output gap containing a recorded preemption; all episodes also reported',
            evidence_type='CPU analysis of existing host observations, not a GPU experiment or causal intervention',
            output_clock='raw relative host receipt + measurement_origin_perf_counter_s within the same run',
            job_completed='native completion report observed on host; absolute GPU completion unknown',
            schedule='native plan return, not GPU execution',
            allocation='all attempts from capacity-handoff; recovery-order allocation events are state changes only',
            normalization_scope='all LOAD/unknown-direction lifecycle and request recovery events; STORE/reorder/global flush traffic remains in raw evidence',
            missing_fields='null means unknown; physical block IDs and token counts are not invented logical ranges',
            causality='all cause attribution unknown; measured intervals may overlap and must not be summed'),
            source_event_counts=counts, summary=dict(episodes=len(episodes), jobs=len(jobs),
            load_episode_associations=dict(job_summary), normalized_events=len(events),
            zero_observed_load_episodes=sum(not e['load_job_ids'] and not e['ambiguous_load_job_ids'] for e in episodes),
            multi_load_episodes=sum(len(e['load_job_ids']) > 1 for e in episodes),
            selected_episode=ranked[0]['episode_id'] if ranked and gap(ranked[0]) >= 0 else None,
            raw_status=raw.get('status'), request_status_counts=dict(collections.Counter(r.get('status', 'unknown') for r in raw.get('requests', [])))),
        episodes=episodes, jobs=jobs, events=events)


def render(report):
    ev = {e['event_id']: e for e in report['events']}
    def dt(p):
        return f"{p['duration_s']:.6f} s" if p.get('status') == 'observed' else '未知（' + p.get('reason', 'missing') + '）'
    s = report['summary']
    lines = ['# KV 恢复时间线与容量等待诊断', '',
        f"输入：`{report['run_id']}`", '',
        f"{s['episodes']} 个抢占 episode；{s['jobs']} 个 LOAD/方向未知 job；"
        f"{s['zero_observed_load_episodes']} 个 episode 没有观察到关联 LOAD；{s['multi_load_episodes']} 个多 LOAD episode。", '',
        '**这是既有轨迹的 CPU 案例分析。时间差不是因果分解；host 完成回报不是 GPU 实际完成时刻。**', '',
        '所有重复事件保留；缺失、歧义、无法对齐的时钟返回未知。STORE/全局 flush 的完整记录仍在输入原件中。', '',
        '## 全部恢复 episode', '',
        '| episode | 外部请求 | 相邻输出间隔 | 抢占→下一输出 | LOAD job | 分配失败/尝试 |',
        '|---|---|---:|---:|---|---:|']
    for ep in report['episodes']:
        lines.append(f"| {ep['episode_id']} | {ep['source_request_id'] or ep['request_id']} | "
            f"{dt(ep['phases']['previous_to_next_output'])} | {dt(ep['phases']['preempt_to_next_output'])} | "
            f"{ep['load_job_ids']}（歧义 {ep['ambiguous_load_job_ids']}） | {ep['allocations']['failed']}/{ep['allocations']['attempts']} |")
    lines += ['', '同一请求可能在下一输出前再次抢占；这些输出窗口可重叠，不能合计为独立请求或可加等待。'
        'JSON 的 overlapping_episode_ids 和 job.episode_candidates 保留这种歧义。', '',
        '## 全部 LOAD 的 host 阶段', '',
        '| job | episode 关联 | ready→提交入口 | 提交入口→host 完成回报 | host 完成回报→ack |',
        '|---|---|---:|---:|---:|']
    for job in report['jobs']:
        p = job['phases']
        lines.append(f"| {job['job_id']} | {job['episode_association']} {job['episode_candidates']} | "
            f"{dt(p['ready_to_submit_begin'])} | {dt(p['submit_begin_to_job_completed'])} | "
            f"{dt(p['job_completed_to_ack_retired'])} |")
    selected = next((e for e in report['episodes'] if e['episode_id'] == s['selected_episode']), None)
    if selected:
        ep = selected
        lines += ['', '## 自动选择的历史案例', '',
            f"规则：选包含抢占的最大相邻输出间隔，不使用固定 request ID。选中 `{ep['episode_id']}` / `{ep['source_request_id'] or ep['request_id']}`。", '',
            '| 观测阶段 | 时间或未知原因 |', '|---|---|']
        for name, p in ep['phases'].items():
            lines.append(f'| {name} | {dt(p)} |')
        retries = sorted(ep['allocations']['retry_gaps'], key=lambda p: -(p.get('duration_s') or -1))
        if retries:
            r = retries[0]
            lines += ['', f"最长相邻 allocator 重试间隔：{dt(r)}；前一失败快照缺 {r['failed_snapshot_shortfall_blocks']} 块。"
                '区间中没有该请求的 allocator 重试记录；持续缺容量与未重试原因均为**未知**。',
                f"证据：`{r['from_events']}` → `{r['to_events']}`。"]
        else:
            lines += ['', '没有可确认的相邻失败→重试区间；不从变化日志补造重试。']
        if ep['source_handoffs']:
            lines += ['', f"source handoff 观测：`{json.dumps(ep['source_handoffs'], ensure_ascii=False)}`。",
                '记录中的 LOAD 列表只描述该次 handoff；不能将其他缺失阶段填成零耗时。']
        else:
            lines += ['', 'source handoff：**未知（该 episode 没有关联的 source handoff 观测）**。'
                '这不等于观察到了空 LOAD 列表；不能据此确认走了重算路径。']
        lines += ['', '### LOAD 的独立链路', '']
        for job in report['jobs']:
            if job['job_id'] not in ep['load_job_ids'] + ep['ambiguous_load_job_ids']:
                continue
            lines += [f"Job `{job['job_id']}`，episode 关联 `{job['episode_association']}`；事件次数 `{job['occurrence_counts']}`。", '']
            for name, p in job['phases'].items():
                lines.append(f'- {name}: {dt(p)}')
            lines += ['- GPU 绝对完成时刻：未知。GPU elapsed 样本单独保留，不从 host 区间相减。', '']
        if not ep['load_job_ids'] and not ep['ambiguous_load_job_ids']:
            lines += ['LOAD ready / submit / 完成回报 / ack：未知（没有观察到关联 LOAD）；不假设每次抢占必须有一次 LOAD。', '']
        lines += ['### 尚不能归因的区间（最长五段）', '', '| 时长 | 左端原始证据 | 右端原始证据 |', '|---:|---|---|']
        for p in sorted(ep['unassigned_intervals'], key=lambda x: -x['duration_s'])[:5]:
            lines.append(f"| {p['duration_s']:.6f} s | `{p['from_events']}` | `{p['to_events']}` |")
        lines += ['', '### 自动关联的时间线', '', '| 相对抢占秒 | kind | job | 原始证据 |', '|---:|---|---|---|']
        start = ev[ep['preempt_event']]['time_s']
        ids = list(ep['event_ids']) + [x for x in (ep['previous_output_event'], ep['next_output_event']) if x]
        for eid in sorted(set(ids), key=lambda x: (ev[x]['time_s'], x)):
            e = ev[eid]
            lines.append(f"| {e['time_s']-start:.6f} | {e['kind']} | {e['job_id']} | `{eid}` |")
    lines += ['', '## 功能边界与价值', '',
        '入口自动连接请求抢占、0..N 个 LOAD 的 host 生命周期、容量分配尝试、再调度计划与下一 host 输出。'
        '它补充本线原生 transfers 汇总所没有的请求/job 关联、失败快照与重试空档，并把无法归因的区间明确留下。', '',
        '普通 profiler 的 kernel/copy 时间仍有用；本工具只整合这里已有的调度语义和证据位置，不声称 profiler 普遍不能实现同类关联。'
        '逻辑块范围与 scheduler step 未采集时保持 null；source-handoff 只覆盖一个选定请求。', '',
        '当前定位：可复现的案例分析工具。自动时间线不能确定未重试原因或服务改进动作，没有独立工具论文或优化收益主张。', '']
    if report['notes']:
        lines += ['输入缺失：'] + ['- ' + x for x in report['notes']]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Existing local B output directory')
    parser.add_argument('--output', required=True, type=Path, help='New directory; never overwrite prior reports')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new directory')
    start = time.perf_counter()
    documents = {name: json.loads((args.input / name).read_text()) for name in FILES if (args.input / name).is_file()}
    if not documents:
        parser.error('no supported local input files')
    report = analyze_documents(documents, str(args.input.resolve()))
    report['analysis'] = dict(cpu_only=True, wall_s_before_serialization=time.perf_counter()-start,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'report.md').write_text(render(report))
    print(json.dumps(dict(output=str(args.output), summary=report['summary'], analysis=report['analysis']), ensure_ascii=False))


if __name__ == '__main__':
    main()

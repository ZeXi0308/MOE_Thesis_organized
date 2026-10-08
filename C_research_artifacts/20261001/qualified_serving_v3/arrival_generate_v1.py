"""Native generation with fixed offered arrivals and one episode clock; no answer keys."""
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import health_native as common

require, dump = common.require, common.dump


def generate(engine, requests, cap, label, output, arrivals_s):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    output = Path(output)
    require(len(requests) == len(arrivals_s) > 0 and type(cap) is int and cap > 0, 'invalid request/arrival/cap geometry')
    require(all(type(a) in (int,float) and math.isfinite(a) and 0 <= a < 900 for a in arrivals_s)
            and list(arrivals_s) == sorted(arrivals_s), 'arrivals must be finite ordered offsets in [0,900)')
    require(len({r['request_id'] for r in requests}) == len(requests), 'duplicate request IDs')
    require(all(not any(k in r for k in ('gold','answer','answer_key')) for r in requests), 'answer key in generation input')
    names = ('host-returns.jsonl','native-sampling.json','outputs.json','steps.json','arrival-receipt.json')
    require(all(not (output/(label+'-'+n)).exists() for n in names), 'generation output already exists')
    s = engine.engine_core.engine_core.scheduler
    original = s.schedule
    preempted, steps, native_sampling, scans = [], [], {}, []
    rows = {label+'/'+r['request_id']: dict(r, external_request_id=label+'/'+r['request_id'],
        output_text='', output_token_ids=[], token_times_s=[], host_returns=[], finished=False,
        finish_reason=None, stop_reason=None, arrival_s=arrival, add_request_s=None,
        add_request_return_s=None, submitted=False) for r,arrival in zip(requests,arrivals_s)}
    ordered = list(rows.items())
    start, epoch = time.perf_counter(), time.time()
    cursor, sampling_dirty, idle_waits, idle_s = 0, False, 0, 0.0
    receipt = dict(schema='c-arrival-generate-v1', status='INCOMPLETE',
        clock=dict(perf_counter_origin_s=start, epoch_origin_unix_s=epoch,
            basis='All row/journal/step timestamps are seconds from the same perf_counter origin; offered arrivals are unchanged.'),
        offered_arrivals_s=list(arrivals_s), planned_requests=len(rows), sampling_scans=scans,
        sampling_scope='Scan native requests only on first schedule after new submissions; append unseen native IDs.',
        sampling_parameters_scope='Same SamplingParams arguments as immutable health_native.generate; no repetition-penalty override.')

    def schedule(*args, **kwargs):
        nonlocal sampling_dirty
        if sampling_dirty:
            new_ids = []
            for rid, request in s.requests.items():
                if rid in native_sampling: continue
                params = request.sampling_params
                native_sampling[rid] = {name: getattr(params,name,None) for name in
                    ('stop','stop_token_ids','ignore_eos','min_tokens','max_tokens','_eos_token_id',
                     'temperature','repetition_penalty')}
                native_sampling[rid]['all_stop_token_ids'] = sorted(getattr(params,'_all_stop_token_ids',()))
                new_ids.append(rid)
            scans.append(dict(before_scheduler_step=s.current_step, submitted_requests=cursor,
                new_native_request_ids=new_ids, native_sampling_count=len(native_sampling)))
            sampling_dirty = False
        result = original(*args, **kwargs)
        preempted.extend(result.preempted_req_ids or [])
        return result

    s.schedule = schedule
    try:
        with (output/(label+'-host-returns.jsonl')).open('x') as journal:
            while cursor < len(ordered) or engine.has_unfinished_requests():
                require(time.perf_counter()-start <= 900, 'generation exceeds 900 seconds')
                # Recheck the clock after each submission so every arrived request precedes the next step.
                while cursor < len(ordered) and ordered[cursor][1]['arrival_s'] <= time.perf_counter()-start:
                    rid, row = ordered[cursor]
                    row['add_request_s'] = time.perf_counter()-start
                    require(row['add_request_s'] >= row['arrival_s'], 'request submitted before offered arrival')
                    row['native_arrival_time_unix_s'] = epoch+row['arrival_s']
                    try:
                        engine.add_request(rid, {'prompt_token_ids':row['prompt_token_ids']},
                            SamplingParams(n=1, temperature=0.0, max_tokens=cap, min_tokens=0,
                                ignore_eos=False, stop=[], stop_token_ids=[], detokenize=True,
                                output_kind=RequestOutputKind.CUMULATIVE), arrival_time=epoch+row['arrival_s'])
                    except BaseException as exc:
                        row['add_request_error'] = f'{type(exc).__name__}: {exc}'; raise
                    finally: row['add_request_return_s'] = time.perf_counter()-start
                    row['submitted'] = True; cursor += 1; sampling_dirty = True
                    require(time.perf_counter()-start <= 900, 'generation exceeds 900 seconds during submission')
                if not engine.has_unfinished_requests():
                    if cursor < len(ordered):
                        begin = time.perf_counter()
                        time.sleep(min(.01, max(0.0,ordered[cursor][1]['arrival_s']-(begin-start))))
                        idle_s += time.perf_counter()-begin; idle_waits += 1
                    continue
                before = time.perf_counter()-start
                if cursor < len(ordered) and ordered[cursor][1]['arrival_s'] <= before:
                    continue  # Arrival crossed the clock boundary after the submission pass.
                require(before <= 900, 'generation exceeds 900 seconds')
                outputs = engine.step()
                received = time.perf_counter()-start
                pool = s.kv_cache_manager.block_pool
                steps.append(dict(start_s=before, return_s=received, output_requests=len(outputs),
                    used_blocks_after_step=pool.num_gpu_blocks-1-pool.get_num_free_blocks()))
                for item in outputs:
                    row = rows[item.request_id]
                    require(row['submitted'] and not row['finished'] and len(item.outputs)==1, 'unexpected output/completion')
                    completion = item.outputs[0]
                    tokens, old = list(completion.token_ids), row['output_token_ids']
                    require(tokens[:len(old)]==old and len(old)<=len(tokens)<=cap, 'cumulative token prefix/cap violated')
                    event = dict(request_id=item.request_id, return_s=received, delta_token_ids=tokens[len(old):],
                        cumulative_tokens=len(tokens), finished=bool(item.finished),
                        finish_reason=completion.finish_reason, stop_reason=getattr(completion,'stop_reason',None))
                    journal.write(json.dumps(event,ensure_ascii=False)+'\n')
                    row['host_returns'].append(event)
                    row['token_times_s'].extend([received]*(len(tokens)-len(old)))
                    row.update(output_token_ids=tokens, output_text=completion.text, finished=event['finished'],
                        finish_reason=event['finish_reason'], stop_reason=event['stop_reason'])
                    if item.finished:
                        require(completion.finish_reason in ('stop','length'), 'unexpected finish reason')
                        row['host_elapsed_s'] = received
                journal.flush()
            require(all(r['finished'] for r in rows.values()), 'unfinished planned requests')
            require(len(native_sampling)==len(rows), 'native sampling inventory incomplete across arrivals')
        receipt['status'] = 'COMPLETE'
        return dict(status='COMPLETE', request_count=len(rows), preemptions=len(preempted),
            observation_end_s=time.perf_counter()-start, output_tokens=sum(len(r['output_token_ids']) for r in rows.values()),
            finish_reason_counts=dict(Counter(r['finish_reason'] for r in rows.values())))
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'; raise
    finally:
        s.schedule = original
        cohorts = []
        for offered in sorted(set(arrivals_s)):
            group = [r for r in rows.values() if r['arrival_s']==offered]
            submitted = [r for r in group if r['submitted']]
            cohorts.append(dict(arrival_s=offered, planned_requests=len(group), submitted_requests=len(submitted),
                finished_requests=sum(r['finished'] for r in group),
                min_add_request_s=min((r['add_request_s'] for r in submitted),default=None),
                max_add_request_return_s=max((r['add_request_return_s'] for r in submitted),default=None),
                max_submission_lag_s=max((r['add_request_s']-offered for r in submitted),default=None)))
        receipt.update(submitted_requests=cursor, finished_requests=sum(r['finished'] for r in rows.values()),
            native_sampling_count=len(native_sampling), idle_wait_count=idle_waits, idle_wait_wall_s=idle_s,
            cohorts=cohorts, observation_end_s=time.perf_counter()-start, hooks_restored=True)
        dump(output/(label+'-native-sampling.json'), native_sampling)
        dump(output/(label+'-outputs.json'), list(rows.values()))
        dump(output/(label+'-steps.json'), dict(steps=steps, preempted_request_ids=preempted,
            timing='Episode-relative host return timestamps, not device ITL; arrivals have not been subtracted.'))
        dump(output/(label+'-arrival-receipt.json'), receipt)

def episode(engine, work, policy_name, path, target_ms):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from prefill_policy import Policy
    path.mkdir()
    scheduler = engine.engine_core.engine_core.scheduler
    assert not scheduler.requests and not engine.has_unfinished_requests()
    policy = PulseController(policy_name, engine)
    native = scheduler.schedule
    rows = {x['request_id']: dict(x, prompt_tokens=len(x['prompt_token_ids']),
        add_s=None, first_scheduled_s=None, completion_s=None, finished=False,
        token_times_s=[], output_token_ids=[]) for x in work}
    steps = []
    origin = time.perf_counter()
    epoch = time.time()
    def now(): return time.perf_counter()-origin
    def schedule(*args, **kwargs):
        started = now()
        decoding = [r for r in scheduler.running if r.num_computed_tokens >= r.num_prompt_tokens]
        decode_ids = {r.request_id for r in decoding}
        contexts = sum(r.num_computed_tokens for r in decoding)
        t = time.perf_counter()
        budget = policy.choose(len(decoding), scheduler, rows, len(steps), now)
        scheduler.d_prefill_budget = budget
        overhead = (time.perf_counter()-t)*1e6
        before = {rid:(r.num_computed_tokens,r.num_prompt_tokens,
                  engine.output_processor.request_states[rid].external_req_id)
                  for rid,r in scheduler.requests.items()}
        try:
            out = native(*args, **kwargs)
        except BaseException as exc:
            policy.native_failed(exc)
            raise
        policy.native_returned(out, before)
        detail = []
        for rid, amount in out.num_scheduled_tokens.items():
            previous, prompt, external = before[rid]
            start = scheduler.requests[rid].num_computed_tokens-amount
            prefill = min(amount,max(0,prompt-start))
            row = rows[external]
            if row['first_scheduled_s'] is None: row['first_scheduled_s'] = started
            detail.append(dict(request_id=external, prefill_tokens=prefill,
                               decode_tokens=amount-prefill, computed_start=start))
        pt = sum(r['prefill_tokens'] for r in detail)
        dt = sum(r['decode_tokens'] for r in detail)
        assert pt <= budget
        assert not out.preempted_req_ids, 'First experiment requires zero KV pressure/preemption'
        assert decode_ids <= set(out.num_scheduled_tokens), 'Native decoder unexpectedly skipped'
        assert all(out.num_scheduled_tokens[rid] == 1 for rid in decode_ids)
        steps.append(dict(start_s=started, end_s=None, budget=budget, prefill_tokens=pt,
            prefill_backlog_tokens_before=sum(max(0,prompt-previous) for previous,prompt,external in before.values()),
            prefill_backlog_requests_before=sum(previous<prompt for previous,prompt,external in before.values()),
            decode_tokens=dt, decode_count_before=len(decoding), decode_context_sum=contexts,
            schedule_s=now()-started, decision_us=overhead, requests=detail,
            running_count=len(scheduler.running),waiting_count=len(scheduler.waiting),
            kv_used_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks-1-scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
            kv_total_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks-1,
            preempted=list(out.preempted_req_ids or [])))
        policy.attach(steps[-1], rows)
        return out
    scheduler.schedule = schedule
    pending = sorted(work, key=lambda x:x['arrival_s'])
    pos = 0
    try:
        while pos < len(pending) or engine.has_unfinished_requests():
            if now()>180: raise TimeoutError('Complete-service bound exceeded; retain partial results')
            while pos<len(pending) and pending[pos]['arrival_s']<=now():
                item = pending[pos]
                p = SamplingParams(temperature=0, max_tokens=item['max_tokens'],
                    min_tokens=item['max_tokens'], ignore_eos=True, detokenize=False,
                    output_kind=RequestOutputKind.CUMULATIVE)
                engine.add_request(item['request_id'], {'prompt_token_ids':item['prompt_token_ids']},
                                   p, arrival_time=epoch+item['arrival_s'])
                rows[item['request_id']]['add_s']=now()
                pos += 1
            if not engine.has_unfinished_requests():
                time.sleep(min(.002,max(0,pending[pos]['arrival_s']-now())))
                continue
            start = now()
            before_count = len(steps)
            outputs = engine.step()
            end = now()
            assert len(steps)==before_count+1
            step = steps[-1]
            step.update(start_s=start,end_s=end)
            policy.observe(end-start,step['prefill_tokens'],step['decode_tokens'])
            for output in outputs:
                row = rows[output.request_id]
                tokens = list(output.outputs[0].token_ids)
                old = len(row['output_token_ids'])
                assert tokens[:old]==row['output_token_ids']
                row['token_times_s'].extend([end]*(len(tokens)-old))
                row['output_token_ids']=tokens
                if output.finished:
                    row.update(finished=True,completion_s=end,finish_reason=output.outputs[0].finish_reason)
        assert all(r['finished'] and len(r['output_token_ids'])==r['max_tokens'] for r in rows.values())
        assert not scheduler.requests
    finally:
        scheduler.schedule = native
        data = dict(policy=policy_name,elapsed_s=now(),requests=list(rows.values()),steps=steps)
        data['pulse'] = policy.summary()
        dump(path/'raw.json',data)
    return dict(policy=policy_name,elapsed_s=data['elapsed_s'],requests=len(rows),
                output_tokens=sum(len(r['output_token_ids']) for r in rows.values()),
                mixed_steps=sum(s['prefill_tokens']>0 and s['decode_tokens']>0 for s in steps))

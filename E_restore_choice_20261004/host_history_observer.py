"""Opt-in, passive native Host-history observations for preempted requests.

Install after RestoreSelector and restore in reverse hook order. Wraps only
OffloadingConnectorScheduler.update_state_after_alloc(request, blocks,
num_external_tokens), _build_store_jobs(scheduler_output), and
update_connector_output(connector_output). Pass selector.now
as clock to share existing lookup/commit timestamps; optional selector supplies
the already-recorded pending event, without another lookup. service_step/drain
labels match the existing transfer probe. Disabled entry installs no wrappers;
disabled wrappers call through without clocks or state observations.

Allocation records expose native STORE-cursor rewinds; store-builder records
show whether actual returned STORE jobs cover those chunks, including calls
with no jobs. Only already-known token counts and the current native scheduled
work are read. No cache lookup/touch, CUDA operation, token hashing, sampling,
STORE policy, or execution order is added. Chunk intervals are half-open and
come from native job keys matched to the request's existing offload-key list.

For previously observed STORE jobs, completion records copy worker-reported
counts and native pending_count before/after, including native job removal.
This establishes native completion acknowledgement, not subsequent Host
residency, cache eviction causality, or performance benefit. Existing
lookup/commit/transfer records and actual online comparisons remain necessary.
Observer errors are counted and never replace native results/exceptions; any
such errors make the affected observation incomplete. This module does not
claim zero overhead when enabled, and records must not be used to subtract
overhead from service latency.
"""


class HostHistoryObserver:
    def __init__(self, connector_scheduler, clock, selector=None, enabled=False):
        self.connector = connector_scheduler
        self.clock, self.selector, self.enabled = clock, selector, enabled
        self.records, self.errors = [], []
        self.phase, self.step_index = 'outside_service', None
        self._last_allocation = {}
        self._store_jobs = {}
        self._saved = []

    def service_step(self, index):
        self.phase, self.step_index = 'service', index

    def drain(self):
        self.phase, self.step_index = 'drain', None

    def _safe(self, fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            self.errors.append(dict(operation=fn.__name__, error_type=type(exc).__name__))
            return None

    def _state(self, state, scheduled=None):
        req = state.req
        return dict(request_id=req.request_id, preemptions=req.num_preemptions,
                    known_tokens=req.num_tokens, generated_tokens=req.num_output_tokens,
                    computed_tokens=req.num_computed_tokens,
                    scheduled_tokens=scheduled,
                    groups=[dict(group_index=i, tokens_per_chunk=cfg.tokens_per_chunk,
                                 next_stored_chunk_idx=gs.next_stored_chunk_idx,
                                 num_hit_chunks=gs.num_hit_chunks,
                                 offload_key_count=len(gs.offload_keys))
                            for i, (cfg, gs) in enumerate(zip(
                                state.config.kv_group_configs, state.group_states))])

    def _pending(self, request_id):
        if self.selector is None:
            return None
        event = self.selector.pending.get(request_id)
        if event is None:
            return None
        fields = ('event', 'decision_s', 'scheduler_step', 'known_tokens',
                  'generated_tokens', 'preemptions', 'prefix_sha256',
                  'host_hit_tokens', 'action', 'eligible', 'fallback')
        return {key: event.get(key) for key in fields}

    def _new_row(self, operation, before):
        row = dict(record_index=len(self.records), operation=operation,
                   phase=self.phase, step_index=self.step_index,
                   time_s=self.clock(), status='entered', before=before)
        self.records.append(row)
        return row

    def _begin_alloc(self, args, kwargs):
        req = args[0] if args else kwargs.get('request')
        if req is None or req.num_preemptions == 0:
            return None
        state = self.connector._req_status.get(req.request_id)
        if state is None:
            return None
        external = args[2] if len(args) > 2 else kwargs.get(
            'num_external_tokens', kwargs.get('external_tokens'))
        row = self._new_row('allocation', self._state(state))
        row.update(external_tokens=external, lookup=self._pending(req.request_id))
        return row, state

    def _end_alloc(self, context):
        row, state = context
        row.update(status='returned', after=self._state(state))
        self._last_allocation[state.req.request_id] = dict(
            record_index=row['record_index'], time_s=row['time_s'],
            preemptions=row['before']['preemptions'], external_tokens=row['external_tokens'],
            lookup=row['lookup'])

    def _begin_store(self, args, kwargs):
        output = args[0] if args else kwargs.get('scheduler_output')
        if output is None:
            return None
        ids = list(output.num_scheduled_tokens) + list(output.finished_req_ids or ())
        states, before = {}, []
        for rid in ids:
            if rid in states:
                continue
            state = self.connector._req_status.get(rid)
            if state is None or state.req.num_preemptions == 0:
                continue
            states[rid] = state
            snapshot = self._state(state, output.num_scheduled_tokens.get(rid, 0))
            previous = self._last_allocation.get(rid)
            snapshot['last_allocation'] = (previous if previous is not None and
                previous['preemptions'] == state.req.num_preemptions else None)
            before.append(snapshot)
        if not states:
            return None
        return self._new_row('store_builder', before), states

    @staticmethod
    def _intervals(indices):
        intervals = []
        for index in indices:
            if intervals and intervals[-1][1] == index:
                intervals[-1][1] += 1
            else:
                intervals.append([index, index+1])
        return intervals

    def _end_store(self, context, jobs):
        row, states = context
        actual = []
        for jid, job in jobs.items():
            if job.req_id not in states:
                continue
            status = self.connector._jobs.get(jid)
            if status is None:
                raise ValueError('Returned native STORE job lacks scheduler status')
            if not status.is_store:
                raise ValueError('Store builder returned non-STORE job')
            state = states[job.req_id]
            groups, matched = [], set()
            for i, gs in enumerate(state.group_states):
                indices = [n for n, key in enumerate(gs.offload_keys) if key in status.keys]
                matched.update(gs.offload_keys[n] for n in indices)
                groups.append(dict(group_index=i, chunk_intervals=self._intervals(indices),
                                   chunk_count=len(indices)))
            actual.append(dict(job_id=jid, request_id=job.req_id,
                               key_count=len(status.keys), groups=groups,
                               unmatched_key_count=len(status.keys-matched),
                               source_block_count=len(job.src_spec.block_ids)))
            self._store_jobs[jid] = dict(creation_record_index=row['record_index'],
                                        request_id=job.req_id)
        row.update(status='returned', after=[self._state(state) for state in states.values()],
                   jobs=actual)

    def _begin_update(self, args, kwargs):
        output = args[0] if args else kwargs.get('connector_output', kwargs.get('output'))
        meta = getattr(output, 'kv_connector_worker_meta', None)
        if meta is None:
            return None
        reports, statuses = [], {}
        for jid, count in meta.completed_jobs.items():
            tracked = self._store_jobs.get(jid)
            if tracked is None:
                continue
            status = self.connector._jobs.get(jid)
            statuses[jid] = status
            reports.append(dict(job_id=jid, reported_count=count, **tracked,
                                present_before=status is not None,
                                pending_count_before=status.pending_count if status is not None else None))
        if not reports:
            return None
        return self._new_row('store_completion', reports), statuses

    def _end_update(self, context):
        row, statuses = context
        after = []
        for jid, before_status in statuses.items():
            current = self.connector._jobs.get(jid)
            removed = before_status is not None and current is None
            after.append(dict(job_id=jid, present_after=current is not None,
                              removed_by_native=removed,
                              pending_count_after=(current.pending_count if current is not None else
                                                   before_status.pending_count if before_status is not None else None)))
            if removed:
                self._store_jobs.pop(jid, None)
        row.update(status='returned', after=after)

    @staticmethod
    def _raised(context):
        context[0]['status'] = 'native_raised'

    def __enter__(self):
        if not self.enabled:
            return self
        if self._saved:
            raise RuntimeError('HostHistoryObserver is already installed')
        hooks = (
            ('update_state_after_alloc', self._begin_alloc, self._end_alloc),
            ('_build_store_jobs', self._begin_store, self._end_store),
            ('update_connector_output', self._begin_update, self._end_update),
        )
        originals = {name: getattr(self.connector, name) for name, _, _ in hooks}
        for name, begin, end in hooks:
            original = originals[name]
            had_instance = name in self.connector.__dict__
            saved_instance = self.connector.__dict__.get(name)

            def observed(*args, _original=original, _begin=begin, _end=end, **kwargs):
                if not self.enabled:
                    return _original(*args, **kwargs)
                context = self._safe(_begin, args, kwargs)
                try:
                    result = _original(*args, **kwargs)
                except BaseException:
                    if context is not None:
                        self._safe(self._raised, context)
                    raise
                if context is not None:
                    if _end == self._end_store:
                        self._safe(_end, context, result)
                    else:
                        self._safe(_end, context)
                return result

            self._saved.append((name, had_instance, saved_instance))
            setattr(self.connector, name, observed)
        return self

    def restore(self):
        for name, had_instance, saved_instance in reversed(self._saved):
            if had_instance:
                setattr(self.connector, name, saved_instance)
            else:
                delattr(self.connector, name)
        self._saved.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()
        return False

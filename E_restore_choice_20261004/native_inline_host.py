"""Bounded native Host recovery probe: wait for its LOAD before same-step forward.

Install after RestoreSelector. No new transfer executor, request-history edit,
STORE edit, or missing-KV approximation. The unchanged native job completion
metadata must still reach the manager; only the redundant scheduler receive
notification is removed for a request whose LOAD was waited inline.

Supported action: ready complete storable prefix h, with 0 < K-h < chunk.
All other requests keep their original asynchronous/native action. This module
establishes no performance claim; waiting also delays this batch's peers.
"""

import sys


class NativeInlineHostRecovery:
    def __init__(self, selector, worker_connector, enabled=False):
        self.selector = selector
        self.cs = selector.cs
        self.connector = worker_connector
        self.enabled = enabled
        self.records = []
        self.phase, self.step_index = 'outside_service', None
        self._saved = []
        self._planned = {}
        self._committed = {}
        self._by_job = {}
        self._await_recv = {}
        self._ack_jobs = set()

    def service_step(self, index):
        self.phase, self.step_index = 'service', index

    def drain(self):
        self.phase, self.step_index = 'drain', None

    @staticmethod
    def _state(req):
        return (id(req), req.request_id, tuple(req.all_token_ids),
                req.num_tokens, req.num_output_tokens, req.num_preemptions,
                req.num_computed_tokens, id(req.sampling_params),
                req.status, req.stop_reason)

    def _validate(self):
        sel, cs, sched = self.selector, self.cs, self.selector.scheduler
        sel._validate_headroom_scope()
        assert sel.policy == 'host' and sel.target_spec is None
        assert sched.scheduler_reserve_full_isl
        assert not getattr(sched, 'needs_kv_cache_zeroing', False)
        assert len(cs.config.kv_group_configs) == 1
        cfg = cs.config.kv_group_configs[0]
        assert cfg.sliding_window_size_in_chunks is None and not cfg.is_eagle_group
        assert cfg.tokens_per_chunk > 0 and cfg.tokens_per_chunk % cfg.tokens_per_block == 0
        assert cs.config.num_workers == 1
        assert type(cs.manager).__name__ == 'CPUOffloadingManager'
        worker = self.connector.connector_worker
        assert worker is not None and worker.worker is not None
        assert type(worker.worker).__name__ == 'CPUOffloadingWorker'
        assert callable(worker.worker.wait)
        assert not self._saved, 'adapter already installed'
        self.chunk = cfg.tokens_per_chunk

    def _hook(self, obj, name, replacement):
        self._saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, replacement)

    def __enter__(self):
        if not self.enabled:
            return self
        self._validate()
        lookup = self.cs.get_num_new_matched_tokens
        allocated = self.cs.update_state_after_alloc
        build = self.cs.build_connector_meta
        start = self.connector.start_load_kv
        finished = self.connector.get_finished

        def observed_lookup(request, num_computed_tokens):
            result = lookup(request, num_computed_tokens)
            if not self.selector.enabled or request.num_preemptions == 0:
                return result
            rid = request.request_id
            old = self._planned.pop(rid, None)
            if old is not None:
                old['reason'] = 'lookup_retried_without_allocation'
            event = self.selector.pending.get(rid)
            row = dict(request_id=rid, phase=self.phase, step_index=self.step_index,
                       time_s=self.selector.now(), known_tokens=request.num_tokens,
                       generated_tokens=request.num_output_tokens,
                       host_hit_tokens=result[0], requested=False, committed=False,
                       waited=False, filtered=False, job_ids=[], reason='native_fallback')
            self.records.append(row)
            if not event or not (event.get('eligible') and event.get('joint_capacity')
                                and event.get('action') == 'host'
                                and event.get('fallback') == 'none'):
                return result
            h = result[0]
            if (type(h) is not int or result[1] is not True or h <= 0
                    or not 0 < request.num_tokens-h < self.chunk
                    or h != request.num_tokens//self.chunk*self.chunk):
                row['reason'] = 'not_complete_storable_prefix_with_incomplete_tail'
                return result
            assert num_computed_tokens == 0 and request.num_computed_tokens == 0
            assert rid not in self._committed and rid not in self._await_recv
            state = self.cs._req_status[rid]
            module = sys.modules[type(self.cs).__module__]
            assert state.offloading_context.policy == module.OffloadPolicy.BLOCK_LEVEL
            assert state.req is request and not state.transfer_jobs
            assert len(state.group_states) == 1
            before = self._state(request)
            row.update(requested=True, reason='awaiting_native_allocation',
                       cursor_before=state.group_states[0].next_stored_chunk_idx,
                       tokens_per_chunk=self.chunk, selector_event=event['event'],
                       request_state_preserved=True)
            self._planned[rid] = row
            assert self._state(request) == before
            return h, False

        def observed_alloc(request, blocks, num_external_tokens):
            rid = request.request_id
            row = self._planned.get(rid)
            if row is not None:
                assert num_external_tokens == row['host_hit_tokens']
                before = self._state(request)
                assert before[3] == row['known_tokens']
            result = allocated(request, blocks, num_external_tokens)
            if row is not None:
                assert self._state(request) == before
                jobs = {jid: job for jid, job in self.cs._current_batch_load_jobs.items()
                        if job.req_id == rid}
                assert len(jobs) == 1, 'expected one native LOAD job per inline recovery'
                row.update(committed=True, reason='native_load_committed',
                           allocation_s=self.selector.now(), job_ids=sorted(jobs),
                           cursor_after=self.cs._req_status[rid].group_states[0].next_stored_chunk_idx)
                assert row['cursor_after'] == row['host_hit_tokens']//self.chunk
                self._planned.pop(rid)
                self._committed[rid] = row
                for jid in jobs:
                    assert jid not in self._by_job
                    self._by_job[jid] = row
            return result

        def observed_build(scheduler_output):
            batch = dict(self._committed)
            for rid, row in batch.items():
                req = self.cs._req_status[rid].req
                n = scheduler_output.num_scheduled_tokens.get(rid, 0)
                assert req.status.name == 'RUNNING'
                assert req.num_computed_tokens == row['host_hit_tokens']
                assert 0 < n <= row['known_tokens']-row['host_hit_tokens']
                assert (req.num_computed_tokens+n)//self.chunk == row['host_hit_tokens']//self.chunk
            meta = build(scheduler_output)
            assert not any(job.req_id in batch for job in meta.store_jobs.values()), \
                'inline LOAD must not share a same-batch native STORE for its request'
            for rid, row in batch.items():
                actual = {jid for jid, job in meta.load_jobs.items() if job.req_id == rid}
                assert actual == set(row['job_ids'])
                row['metadata_s'] = self.selector.now()
                self._committed.pop(rid)
            for row in self._planned.values():
                row['reason'] = 'native_allocation_not_committed'
            self._planned.clear()
            return meta

        def observed_start(*args, **kwargs):
            meta = self.connector._connector_metadata
            selected = {jid: self._by_job[jid] for jid in meta.load_jobs if jid in self._by_job}
            for jid, row in selected.items():
                assert not row['waited'] and meta.load_jobs[jid].req_id == row['request_id']
                assert not any(job.req_id == row['request_id'] for job in meta.store_jobs.values())
            result = start(*args, **kwargs)
            if selected:
                worker = self.connector.connector_worker
                for jid, row in selected.items():
                    assert worker._load_jobs.get(jid) == row['request_id']
                begin = self.selector.now()
                worker.worker.wait(set(selected))
                end = self.selector.now()
                for jid, row in selected.items():
                    row.update(waited=True, reason='native_wait_completed',
                               wait_start_s=begin, wait_end_s=end)
                    assert row['request_id'] not in self._await_recv
                    self._await_recv[row['request_id']] = row
            return result

        def observed_finished(*args, **kwargs):
            sending, recving = finished(*args, **kwargs)
            filtered = set(recving).intersection(self._await_recv)
            for rid in filtered:
                row = self._await_recv.pop(rid)
                assert row['waited']
                worker_meta = self.connector.connector_worker._connector_worker_meta
                for jid in row['job_ids']:
                    assert worker_meta.completed_jobs.get(jid, 0) > 0, \
                        'native completion metadata must remain intact'
                    assert jid not in self.connector.connector_worker._load_jobs
                    self._ack_jobs.add(jid)
                    self._by_job.pop(jid)
                row.update(filtered=True, reason='native_ack_preserved_recv_notification_filtered',
                           filtered_s=self.selector.now())
            return sending, recving-filtered

        self._hook(self.cs, 'get_num_new_matched_tokens', observed_lookup)
        self._hook(self.cs, 'update_state_after_alloc', observed_alloc)
        self._hook(self.cs, 'build_connector_meta', observed_build)
        self._hook(self.connector, 'start_load_kv', observed_start)
        self._hook(self.connector, 'get_finished', observed_finished)
        return self

    def restore(self):
        for obj, name, had_instance, saved in reversed(self._saved):
            if had_instance:
                setattr(obj, name, saved)
            else:
                delattr(obj, name)
        self._saved.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()
        if self.enabled and exc_type is None:
            assert not self._committed and not self._by_job and not self._await_recv, \
                'inline native LOAD did not finish its normal acknowledgement path'
            assert self._ack_jobs.isdisjoint(self.cs._jobs), \
                'native manager did not consume inline LOAD completion metadata'
            for row in self._planned.values():
                row['reason'] = 'context_ended_without_allocation'
            self._planned.clear()
        return False

"""One-entry causal probe for native STORE-history feedback; opt-in only.

Install after RestoreSelector and before HostHistoryObserver. After the real
selector has committed its preassigned legal RECOMPUTE event, rewind only that
request's native STORE cursor to the recorded ready Host-prefix boundary. The
unchanged native _build_store_jobs/prepare_store then decide which missing
chunks to store; this module never submits copies or scans cache residency.

This intentionally changes subsequent STORE behavior and must have its full
cost included in an online control. It is not a selector improvement or a
general semantic repair. Supports one non-draft full-attention BLOCK_LEVEL
group only. Disabled entry installs no hook. Native calls, return objects and
exceptions pass through; probe failures are recorded, not raised over native
results. No future output, extra cache lookup/touch, CUDA or scheduling call.
"""

import sys
from selector import token_hash


class NativeStoreReplay:
    def __init__(self, selector, enabled=False):
        self.selector, self.connector, self.enabled = selector, selector.cs, enabled
        self.records, self.errors = [], []
        self.executed_count = 0
        self.phase, self.step_index = 'outside_service', None
        self._saved = None

    def service_step(self, index):
        self.phase, self.step_index = 'service', index

    def drain(self):
        self.phase, self.step_index = 'drain', None

    @staticmethod
    def _request_state(req):
        return (id(req), req.request_id, tuple(req.all_token_ids),
                req.num_tokens, req.num_output_tokens, req.num_preemptions,
                req.num_computed_tokens, id(req.sampling_params), req.status, req.stop_reason)

    def _begin(self, request, external):
        event = self.selector.pending.get(request.request_id)
        if event is None and request.num_preemptions == 0:
            return None
        event = dict(event) if event is not None else {}
        keys = ('event', 'decision_s', 'request_id', 'preemptions', 'known_tokens',
                'generated_tokens', 'prefix_sha256', 'host_hit_tokens', 'action',
                'eligible', 'joint_capacity', 'fallback', 'target_selected')
        row = {key: event.get(key) for key in keys}
        row.update(request_id=request.request_id, phase=self.phase, step_index=self.step_index,
                   status='skipped', requested=False, executed=False, reason='not_evaluated',
                   native_external_tokens=external, cursor_before=None, cursor_after=None,
                   ready_prefix_chunk_index=None, tokens_per_chunk=None,
                   target_commit_observed=False, request_state_preserved=None,
                   known_prefix_matches_record=None, known_tokens_after=None,
                   generated_tokens_after=None, prefix_sha256_after=None)
        self.records.append(row)
        context = dict(row=row, event=event, request=request,
                       commit_count=len(self.selector.commits), candidate=False)
        if self.executed_count:
            row['reason'] = 'already_executed'
        elif not self.selector.enabled:
            row['reason'] = 'selector_disabled'
        elif not event:
            row['reason'] = 'no_pending_event'
        elif event.get('fallback') != 'none' or not event.get('eligible') or not event.get('joint_capacity'):
            row['reason'] = 'fallback_or_not_jointly_legal'
        elif not event.get('target_selected'):
            row['reason'] = 'not_target'
        elif event.get('action') != 'recompute' or external != 0:
            row['reason'] = 'target_action_not_recompute'
        elif not self.selector.target_spec or self.selector.target_spec.get('schema') != 'E.one_event_target.v1':
            row['reason'] = 'no_preassigned_one_event_target'
        else:
            row.update(requested=True, reason='awaiting_native_commit')
            context.update(candidate=True, request_state=self._request_state(request))
        return context

    def _finish(self, context):
        row, event, req = context['row'], context['event'], context['request']
        if not context['candidate']:
            return
        current_hash = token_hash(req.all_token_ids)
        row.update(known_tokens_after=req.num_tokens, generated_tokens_after=req.num_output_tokens,
                   prefix_sha256_after=current_hash,
                   request_state_preserved=self._request_state(req) == context['request_state'],
                   known_prefix_matches_record=current_hash == event.get('prefix_sha256'))
        commits = self.selector.commits[context['commit_count']:]
        identity = ('event', 'request_id', 'decision_s', 'preemptions', 'known_tokens',
                    'generated_tokens', 'prefix_sha256', 'host_hit_tokens', 'action',
                    'eligible', 'joint_capacity', 'fallback')
        if len(commits) != 1 or any(commits[0].get(k) != event.get(k) for k in identity):
            row['reason'] = 'target_commit_not_observed'
            return
        commit = commits[0]
        if not (commit.get('target_committed') is True and commit.get('actual_action') == 'recompute'
                and commit.get('external_tokens') == 0 and commit.get('target_commit_count_after') == 1
                and self.selector.target_commit_count == 1):
            row['reason'] = 'target_commit_or_latch_mismatch'
            return
        row.update(target_commit_observed=True, native_allocation_s=commit.get('allocation_s'))
        target = self.selector.target_spec['target']
        observed = dict(external_id=self.selector.external_ids.get(req.request_id),
                        num_preemptions=req.num_preemptions, known_tokens=req.num_tokens,
                        generated_tokens=req.num_output_tokens, prefix_sha256=current_hash)
        if any(target.get(k) != value for k, value in observed.items()):
            row['reason'] = 'preassigned_target_state_mismatch'
            return
        if not row['request_state_preserved'] or not row['known_prefix_matches_record']:
            row['reason'] = 'request_state_changed'
            return
        state = self.connector._req_status.get(req.request_id)
        if state is None or state.req is not req:
            row['reason'] = 'native_request_state_missing'
            return
        configs = self.connector.config.kv_group_configs
        if len(configs) != 1 or len(state.group_states) != 1:
            row['reason'] = 'unsupported_group_count'
            return
        config, group = configs[0], state.group_states[0]
        if config.sliding_window_size_in_chunks is not None or config.is_eagle_group:
            row['reason'] = 'unsupported_non_full_attention_or_draft_group'
            return
        native_module = sys.modules.get(type(self.connector).__module__)
        native_policies = getattr(native_module, 'OffloadPolicy', None)
        block_policy = getattr(native_policies, 'BLOCK_LEVEL', None)
        if block_policy is None or state.offloading_context.policy != block_policy:
            row['reason'] = 'unsupported_offload_policy'
            return
        size, hit = config.tokens_per_chunk, event['host_hit_tokens']
        if type(size) is not int or size <= 0 or type(hit) is not int or not 0 < hit <= req.num_tokens or hit % size:
            row['reason'] = 'unsupported_host_prefix_alignment'
            return
        old, new = group.next_stored_chunk_idx, hit//size
        row.update(cursor_before=old, cursor_after=old, ready_prefix_chunk_index=new,
                   tokens_per_chunk=size)
        if type(old) is not int or not 0 <= old <= len(group.offload_keys):
            row['reason'] = 'unsupported_native_cursor'
            return
        if old <= new:
            row['reason'] = 'cursor_already_at_or_before_ready_prefix'
            return
        # The only execution-state mutation in this module. Actual STORE keys
        # remain filtered by the original native builder and manager.
        try:
            group.next_stored_chunk_idx = new
            if group.next_stored_chunk_idx != new or self._request_state(req) != context['request_state']:
                raise RuntimeError('STORE cursor write or request-state preservation failed')
        except Exception:
            group.next_stored_chunk_idx = old
            row['cursor_after'] = group.next_stored_chunk_idx
            raise
        self.executed_count += 1
        row.update(cursor_after=new, status='executed', executed=True,
                   reason='rewound_to_recorded_ready_host_prefix')

    def _error(self, stage, exc, context):
        self.errors.append(dict(stage=stage, error_type=type(exc).__name__))
        if context is not None:
            context['row'].update(status='failed', reason='probe_'+stage+'_failed')

    def __enter__(self):
        if not self.enabled:
            return self
        if self._saved is not None:
            raise RuntimeError('NativeStoreReplay is already installed')
        name = 'update_state_after_alloc'
        original = getattr(self.connector, name)
        self._saved = (name in self.connector.__dict__, self.connector.__dict__.get(name))

        def observed(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            context = None
            try:
                req = args[0] if args else kwargs.get('request')
                external = args[2] if len(args) > 2 else kwargs.get(
                    'num_external_tokens', kwargs.get('external_tokens'))
                context = self._begin(req, external) if req is not None else None
            except Exception as exc:
                self._error('capture', exc, context)
            try:
                result = original(*args, **kwargs)
            except BaseException:
                if context is not None:
                    context['row'].update(status='native_raised', reason='native_allocation_raised')
                raise
            if context is not None:
                try:
                    self._finish(context)
                except Exception as exc:
                    self._error('application', exc, context)
            return result

        setattr(self.connector, name, observed)
        return self

    def restore(self):
        if self._saved is not None:
            had_instance, saved = self._saved
            if had_instance:
                self.connector.update_state_after_alloc = saved
            else:
                del self.connector.update_state_after_alloc
            self._saved = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()
        return False

"""One bounded head-completion action probe, not a deployed online algorithm.

The 2048 action was frozen from an offline warm-model exercise. Runtime executes
no rollout or prediction and observes only the currently known native queues.
"""
from prefill_demand.controller import DemandController


class ProbeController(DemandController):
    def __init__(self,name,engine):
        assert name in ('fixed1024','fixed2048','head2048'),name
        super().__init__('fixed2048' if name=='fixed2048' else 'fixed1024',engine)
        self.name=name
        self.probe_trigger_step=None
        self.probe_head_id=self.probe_head_internal_id=None
        self.probe_guard_passed=None
        self.probe_state='WARM_FIXED2048' if name=='fixed2048' else 'WAITING_TRIGGER'
        self.probe_action_decisions=0
        self.probe_action_forwards=0
        self.probe_actual_above=0
        self.probe_action_prefill_tokens=0

    def _snapshot(self,scheduler,rows,record):
        running,waiting=list(scheduler.running),list(scheduler.waiting)
        active=running+waiting
        assert len({r.request_id for r in active})==len(active), 'Duplicate active native request'
        assert {r.request_id for r in active}==set(scheduler.requests), 'Known active state omitted from FIFO snapshot'
        def convert(request):
            external=self.external(request)
            row=rows[external]
            times=row['token_times_s']
            emitted=len(row['output_token_ids'])
            assert len(times)==emitted and (not times or times[-1]<=record['decision_time_s'])
            assert 0<=emitted<row['max_tokens'], 'Snapshot requires unfinished declared-output requests'
            assert request.num_prompt_tokens==row['prompt_tokens']
            if request.num_computed_tokens>=request.num_prompt_tokens:
                assert request.num_computed_tokens+1==request.num_prompt_tokens+emitted
            else:
                assert emitted==0, 'Partial prompt already has output outside frozen domain'
            return dict(id=external,prompt=request.num_prompt_tokens,output_limit=row['max_tokens'],
                arrival=row['arrival_s'],prefill_done=min(request.num_computed_tokens,request.num_prompt_tokens),
                emitted=emitted,first=times[0] if times else None,last=times[-1] if times else None,
                max_gap=max((b-a for a,b in zip(times,times[1:])),default=0.))
        run_state=[convert(r) for r in running]
        wait_state=[convert(r) for r in waiting]
        block_size=self.engine.vllm_config.cache_config.block_size
        assert block_size==16, 'Frozen full-allocation guard assumes 16-token blocks'
        total=scheduler.kv_cache_manager.block_pool.num_gpu_blocks-1
        reserved=sum((q['prompt']+q['output_limit']+15)//16 for q in run_state+wait_state)
        self.probe_head_internal_id=next(r.request_id for r in active if self.external(r)==self.probe_head_id)
        return dict(now=record['decision_time_s'],running=run_state,waiting=wait_state,total_blocks=total,
            head_id=self.probe_head_id,decode_count=record['decode_count'],
            decode_context_source=record['decode_context_source'],free_kv_blocks=record['free_kv_blocks'],
            known_full_allocation_blocks=reserved,full_allocation_fits=reserved<=total,
            resource_guard_scope='All currently known active declared prompt+output blocks; conservative fit guard, not native allocation replay.')

    def choose(self,decode_count,scheduler,rows,step_index,now):
        policy=self.name
        changed_before=self.changed_cap
        self.name='fixed2048' if policy=='fixed2048' else 'fixed1024'
        try:
            cap=super().choose(decode_count,scheduler,rows,step_index,now)
        finally:
            self.name=policy
        record=self.pending
        head=record['head_prefill_id']
        remaining=0 if head is None else rows[head]['prompt_tokens']-record['head_prefill_context_estimate']
        trigger_condition=decode_count>=128 and head is not None and remaining>1024
        trigger_this_step=False
        snapshot=None
        if policy!='fixed2048' and self.probe_trigger_step is None and trigger_condition:
            self.probe_trigger_step=step_index
            self.probe_head_id=head
            trigger_this_step=True
            snapshot=self._snapshot(scheduler,rows,record)
            self.probe_guard_passed=snapshot['full_allocation_fits']
            self.probe_state=('DOMAIN_REJECTED' if not self.probe_guard_passed else
                              'REFERENCE_RECORDED' if policy=='fixed1024' else 'ACTIVE')
        first_seen=(self.probe_head_id is not None and bool(rows[self.probe_head_id]['token_times_s']))
        action_requested=False
        if policy=='head2048' and self.probe_state=='ACTIVE':
            if first_seen:
                self.probe_state='HEAD_FIRST_OUTPUT_OBSERVED'
            elif self.probe_head_internal_id not in scheduler.requests:
                self.probe_state='HEAD_MISSING_NO_OUTPUT'
            elif self.probe_action_decisions>=4:
                self.probe_state='FOUR_DECISION_LIMIT'
            elif head is None:
                self.probe_state='NO_PREFILL_HEAD_AFTER_TRIGGER'
            else:
                cap=2048
                self.probe_action_decisions+=1
                action_requested=True
        elif policy=='fixed1024' and self.probe_state=='REFERENCE_RECORDED' and first_seen:
            self.probe_state='REFERENCE_HEAD_FIRST_OUTPUT_OBSERVED'
        assert self.probe_action_decisions<=4
        legal=[2048] if policy=='fixed2048' else [1024] if policy=='fixed1024' else [1024,2048]
        source=('fixed2048_warm' if policy=='fixed2048' else
                'fixed1024_reference' if policy=='fixed1024' else
                'head2048_bounded_action' if action_requested else 'head2048_common1024')
        record.update(policy=policy,candidate_cap=cap if policy=='head2048' else None,
            requested_cap=cap,final_cap=cap,requested_total_budget=cap+decode_count,
            legal_prefill_caps=legal,legal_total_tiers=[c+decode_count for c in legal],
            execution_source=source,reason=self.probe_state,
            probe_state=self.probe_state,probe_trigger_condition=trigger_condition,
            probe_trigger_this_step=trigger_this_step,probe_trigger_step=self.probe_trigger_step,
            probe_head_id=self.probe_head_id,probe_guard_passed=self.probe_guard_passed,
            probe_head_first_output_seen=first_seen,probe_action_requested=action_requested,
            probe_action_forward_completed=False,
            probe_action_decisions_requested=self.probe_action_decisions)
        if snapshot is not None:
            record['probe_snapshot']=snapshot
        self.changed_cap=changed_before+int(record['prefill_backlog_tokens']>0 and cap!=1024)
        return cap

    def observe(self,elapsed,p,d):
        record=self.awaiting_forward['component_decision']
        super().observe(elapsed,p,d)
        if record['probe_action_requested']:
            record['probe_action_forward_completed']=True
            self.probe_action_forwards+=1
            self.probe_actual_above+=int(p>1024)
            self.probe_action_prefill_tokens+=p

    def summary(self):
        result=super().summary()
        result.update(probe_triggered=self.probe_trigger_step is not None,
            probe_trigger_step=self.probe_trigger_step,probe_head_id=self.probe_head_id,
            probe_guard_passed=self.probe_guard_passed,
            probe_action_decisions_requested=self.probe_action_decisions,
            probe_action_forwards_completed=self.probe_action_forwards,
            probe_actual_P_above_1024_action_forwards=self.probe_actual_above,
            probe_actual_prefill_tokens_during_action=self.probe_action_prefill_tokens,
            probe_terminal_state=self.probe_state)
        return result

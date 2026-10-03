from vllm.core.scheduler import * 
from vllm.core.andes_utils.knapsack_solver import KnapSack
from vllm.core.andes_utils.lqf import LeastQoEFirst
from vllm.core.andes_utils.preemption_tracker import PreemptionTracker

import atexit

class AndesScheduler(Scheduler):
    """
    Implementation of the Andes, which is a QoE-aware serving system.
    Paper: https://arxiv.org/abs/2404.16283
    """
    def __init__(
            self,
            scheduler_config: SchedulerConfig,
            cache_config: CacheConfig,
            lora_config: Optional[LoRAConfig],
            pipeline_parallel_size: int = 1,
            output_proc_callback: Optional[Callable] = None,
        ) -> None:
        super().__init__(scheduler_config, cache_config, lora_config,
                         pipeline_parallel_size, output_proc_callback)
        self.num_total_gpu_blocks = self.cache_config.num_gpu_blocks
        if scheduler_config.scheduling_strategy == 'qoe-avg' or scheduler_config.scheduling_strategy == 'qoe-min':
            self.schedule_solver = KnapSack(self.cache_config.block_size,
                                            self.num_total_gpu_blocks,
                                            'greedy', )
        elif scheduler_config.scheduling_strategy == 'lqf': 
            self.schedule_solver = LeastQoEFirst(self.cache_config.block_size,
                                            self.num_total_gpu_blocks,
                                            'greedy')
            
        if self.user_specified_preemption_mode is None:
            self.user_specified_preemption_mode = PreemptionMode.RECOMPUTE
        elif self.user_specified_preemption_mode == 'swap':
            self.user_specified_preemption_mode = PreemptionMode.SWAP
        elif self.user_specified_preemption_mode == 'recompute':
            self.user_specified_preemption_mode = PreemptionMode.RECOMPUTE
        
        # self.preempt_tracker = PreemptionTracker(600, 30)
        # self.request_arrival_tracker = PreemptionTracker(600, 30)
        self.preemption_freq = scheduler_config.preemption_freq
        self.iter = 0
        self.system_logfile = scheduler_config.system_logfile
        self.scheduler_action = []
        atexit.register(self.write_log_at_exit)

    def write_log_at_exit(self):
        log_result = "Your log message here"  # Replace with actual log result
        with open(self.system_logfile, 'a') as f:
            f.write(f"{self.scheduler_action}\n")

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)
        # self.request_arrival_tracker.add_request(seq_group.metrics.arrival_time)

    def _schedule(self) -> SchedulerOutputs:
        return self._schedule_qoe_aware()
    
    def _schedule_qoe_aware(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The policy is designed to optimize avg QoE under 
        GPU memory pressure by preempting. 
        """
        self.iter += 1
        enable_chunking = False
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        if self.iter % 10 != 0: # or \
            # self.preempt_tracker.get_request_count() > self.request_arrival_tracker.get_request_count() * self.preemption_freq: 
            utilization = 0
        else:
            num_free_blocks = max(0, self.block_manager.gpu_allocator.get_num_free_blocks() - len(self.running)) 
            utilization = (self.num_total_gpu_blocks - num_free_blocks) / self.num_total_gpu_blocks

        # step 1. Andes decides the scheduling change
        seq_to_admit, seq_to_swap_in, seq_to_evict = \
            self.schedule_solver.schedule_requests(budget, 
                                                    self.running, 
                                                    self.waiting, 
                                                    self.swapped, 
                                                    utilization, 
                                                    None)
        # logger.info("[Andes] Admitting %d, swapping in %d, evicting %d", 
        #             len(seq_to_admit), len(seq_to_swap_in), len(seq_to_evict))

        preempted = 0 
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        blocks_to_swap_in: List[Tuple[int, int]] = [] 

        # step 2. evict requests to make room for other requests
        for seq_group in seq_to_evict:
            preempted_mode = self._preempt(seq_group,
                                        blocks_to_swap_out)
            if preempted_mode == PreemptionMode.RECOMPUTE:
                self.waiting.append(seq_group)
            else:
                self.swapped.append(seq_group)
            self.running.remove(seq_group)
            # logger.info(f"[Andes] Evict - {preempted_mode} req - {seq_group.request_id}")
        preempted += len(seq_to_evict)
        # now = time.monotonic()
        # self.preempt_tracker.add_request(now, preempted)

        if len(seq_to_evict) > 0:
            logger.info(f"[Andes] Evicting {len(seq_to_evict)} requests")
            self.scheduler_action.append(f"[{int(time.time())}] Evicting {len(seq_to_evict)} requests")

        # step 3. if there is new request to admit then do not schedule decode
        if seq_to_admit:
            # logger.info(f"[Andes] Admitting {len(seq_to_admit)} requests")
            prefills = self._schedule_prefills(budget,
                                               curr_loras,
                                               seq_to_admit,
                                               enable_chunking=enable_chunking)

            if len(prefills.seq_groups) > 0:
                self.running.extend([s.seq_group for s in prefills.seq_groups])

        num_prefill_groups = len(prefills.seq_groups)
        # step 4. scheduling running requests, preempt if overload the server
        # skip this for now, assuming extra block for each request in the knapsack solver, 
        # there should be no need to preempt running requests
        # TODO: may need to check if there is infeasible requests
        if num_prefill_groups == 0:
            running_scheduled = self._schedule_running(budget,
                                                       curr_loras,
                                                       enable_chunking=enable_chunking)
            self.running.extend(running_scheduled.decode_seq_groups_list)
            self.waiting.extendleft(running_scheduled.preempted)
            self.swapped.extend(running_scheduled.swapped_out)
            preempted += (len(running_scheduled.preempted) + len(running_scheduled.swapped_out))


        # step 5. swap in requests, add to the running list
        # swap in happen only before decoding, no swap in on prefill
        
        if num_prefill_groups == 0 and len(seq_to_swap_in) > 0:
            swapped_in = self._schedule_swapped(budget, curr_loras, seq_to_swap_in)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])

        # Merge lists
        # FIXME: num_prefill_groups may be inaccurate 
        # num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=False),
            running_queue_size=len(self.running),
            preempted=preempted,
        )


    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        seq_to_swap_in: Deque[SequenceGroup],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = seq_to_swap_in

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group, self._get_num_lookahead_slots(is_prefill))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, 
                                                      budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                # break
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_STOPPED
                self.swapped.remove(seq_group)
                swapped_queue.popleft()
                continue


            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            self.swapped.remove(seq_group)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        ret: SchedulerRunningOutputs = \
            self._scheduler_running_outputs_cache[self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[
            ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        running_queue = self.running
        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()
            # With async postprocessor, an extra decode run is done
            # to process the final tokens. The check below avoids this extra
            # decode run when the model max len is reached, in order to avoid
            # a memory overflow.
            if self.use_async_output_proc and seq_group.seqs[0].get_len(
            ) > self.scheduler_config.max_model_len:
                self._async_stopped.append(seq_group)
                continue

            # NOTE(woosuk): Preemption happens only when there is no available
            # # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    # TODO
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # It may be that the async pending "victim_seq_group"
                    # becomes finished, in which case we simply free it.
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # Do preemption
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    logger.info(f"Preempt for running requests - {preempted_mode} - {victim_seq_group.request_id}")
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()

                scheduled_seq_group: ScheduledSequenceGroup = \
                    self._scheduled_seq_group_cache[self.cache_id].get_object()
                scheduled_seq_group.seq_group = seq_group
                # READ: why this seq can be prefill, for chuncked prefill?
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id,
                                                num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)
        # set _index to 0
        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        request_to_admit: Deque[SequenceGroup],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = request_to_admit

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)

            # assert len(waiting_seqs) == 1, (
            #     "Waiting sequence group should have only one prompt "
            #     "sequence. ")
            # the request gets preempted, while also finished, delayed due to async proc
            if len(waiting_seqs) == 0: 
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_STOPPED
                # ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                self.waiting.remove(seq_group)
                continue

            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                # logger.info(f"Cannot schedule requests. num_new_tokens: {num_new_tokens}, num_new_seqs: {num_new_seqs}")
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)
            seq_group.init_multi_step(
                num_scheduler_steps=self._get_num_lookahead_slots(
                    is_prefill=True) + 1)
            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            self.waiting.remove(seq_group)
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))


    def _preempt(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: Dict[int, int], 
        ) -> PreemptionMode:  

            preemption_mode = self.user_specified_preemption_mode

            if preemption_mode == PreemptionMode.RECOMPUTE:
                self._preempt_by_recompute(seq_group)
            elif preemption_mode == PreemptionMode.SWAP:
                preemption_mode = self._preempt_by_swap(seq_group, blocks_to_swap_out)
            else:
                raise AssertionError("Invalid preemption mode.")
            # seq_group.preempt_signal()
            return preemption_mode

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if self._swap_out(seq_group, blocks_to_swap_out):
            return PreemptionMode.SWAP
        else:
            self._preempt_by_recompute(seq_group)
            return PreemptionMode.RECOMPUTE

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            logger.warning(
                "Recompute due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
            return False
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
            seq.reset_state_for_swap()
        
        return True

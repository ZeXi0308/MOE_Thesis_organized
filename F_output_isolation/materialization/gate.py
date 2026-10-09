"""Bounded output materialization around unchanged native generation updates.

Static controls are competitive early gates, not post-serialization buffering.
The sole candidate (lazy_static) also batches sample-logprob materialization;
numeric cumulative state still advances on each engine update. No dynamic rule.
"""
import asyncio
import time

import numpy as np
from vllm.sampling_params import RequestOutputKind
from vllm.v1.outputs import LogprobsLists

ARMS = ('native1', 'native2', 'native4', 'token4', 'bytes1024', 'class_static', 'lazy_static')


class RequestGate:
    def __init__(self, manager, state, heavy):
        assert state.output_kind == RequestOutputKind.DELTA
        assert state.parent_req is None and not state.streaming_input
        assert state.detokenizer is not None and state.queue is not None
        self.manager, self.state, self.heavy = manager, state, heavy
        self.original_make = state.make_request_output
        self.lp = state.logprobs_processor
        self.sample = self.lp._update_sample_logprobs
        self.offset = 0
        self.pending_bytes = 0
        self.pending_since_ns = None
        self.handle = None
        self.raw = []
        self.raw_base_cumulative = None
        self.done = False
        self.native = manager.arm.startswith('native')
        state.stream_interval = int(manager.arm[-1]) if self.native else 1
        self.lazy = manager.arm == 'lazy_static' and heavy
        self.lp._update_sample_logprobs = self.counted_sample
        if self.lazy:
            self.lp.update_from_output = self.defer_sample
        state.make_request_output = self.make

    def counted_sample(self, value):
        self.manager.stats['sample_materialization_calls'] += 1
        self.manager.stats['sample_positions_materialized'] += len(value.sampled_token_ranks)
        return self.sample(value)

    def defer_sample(self, output):
        if output.new_prompt_logprobs_tensors is not None:
            raise NotImplementedError('prompt logprobs outside frozen scope')
        value = output.new_logprobs
        if value is None:
            return
        if not self.raw:
            self.raw_base_cumulative = self.lp.cumulative_logprob
        self.raw.append(value)
        # Same scalar conversion and addition order as native processing.
        # Text/stop state was already advanced by native process_outputs.
        for row in value.logprobs:
            self.lp.cumulative_logprob += row[0].tolist()
        self.manager.stats['sample_positions_deferred'] += len(value.sampled_token_ranks)

    def realize(self):
        if not self.raw:
            return
        expected_cumulative = self.lp.cumulative_logprob
        if len(self.raw) == 1:
            combined = self.raw[0]
        else:
            combined = LogprobsLists(
                np.concatenate([x.logprob_token_ids for x in self.raw], axis=0),
                np.concatenate([x.logprobs for x in self.raw], axis=0),
                np.concatenate([x.sampled_token_ranks for x in self.raw], axis=0), None)
            self.manager.stats['raw_merge_calls'] += 1
            self.manager.stats['raw_merge_copy_bytes'] += sum(
                x.nbytes for x in combined[:3])
        # Native per-position decoding, contextual UTF-8 correction and
        # Logprob construction are preserved. This is not eliminated work.
        self.lp.cumulative_logprob = self.raw_base_cumulative
        self.counted_sample(combined)
        assert self.lp.cumulative_logprob == expected_cumulative
        self.raw.clear()
        self.raw_base_cumulative = None

    def note_input(self, output):
        now = time.perf_counter_ns()
        if self.pending_since_ns is None and output.new_token_ids:
            self.pending_since_ns = now
        size = 8 * len(output.new_token_ids)  # explicit fixed-width ID budget
        if output.new_logprobs is not None:
            size += sum(x.nbytes for x in output.new_logprobs[:3])
        self.pending_bytes += size
        self.manager.pending_bytes += size
        self.manager.stats['pending_raw_bytes_peak'] = max(
            self.manager.stats['pending_raw_bytes_peak'], self.manager.pending_bytes)

    def cancel_timer(self):
        if self.handle is not None:
            self.handle.cancel()
            self.handle = None

    def reset_pending(self):
        self.cancel_timer()
        self.manager.pending_bytes -= self.pending_bytes
        self.pending_bytes = 0
        self.pending_since_ns = None

    def count_emission(self, output, reason):
        if output is None:
            return
        now = time.perf_counter_ns()
        count = sum(len(x.token_ids) for x in output.outputs)
        stats = self.manager.stats
        stats['request_outputs_materialized'] += 1
        stats['tokens_materialized'] += count
        stats['max_tokens_per_materialization'] = max(stats['max_tokens_per_materialization'], count)
        stats['flush_reasons'][reason] = stats['flush_reasons'].get(reason, 0) + 1
        if self.pending_since_ns is not None:
            age = now - self.pending_since_ns
            stats['max_pending_age_ms'] = max(stats['max_pending_age_ms'], age / 1e6)
            if not self.native and self.offset > 0:
                stats['max_deadline_overrun_ms'] = max(stats['max_deadline_overrun_ms'],
                    max(0, age - self.manager.max_wait_ns) / 1e6)
        self.offset = self.state.detokenizer.num_output_tokens()
        self.reset_pending()

    def emit(self, pooling_output=None, finish_reason=None, stop_reason=None,
             kv_transfer_params=None, ec_transfer_params=None, reason='timer'):
        self.realize()
        pending = self.state.detokenizer.output_token_ids[self.offset:]
        output = self.original_make(pending, pooling_output, finish_reason,
                                    stop_reason, kv_transfer_params, ec_transfer_params)
        assert output is not None
        self.count_emission(output, reason)
        self.state.sent_tokens_offset = self.offset
        return output

    def on_timer(self):
        self.handle = None
        if self.done or self.manager.processor.request_states.get(self.state.request_id) is not self.state:
            return
        if self.state.detokenizer.num_output_tokens() == self.offset:
            return
        start = time.thread_time_ns()
        output = self.emit(reason='timer')
        self.state.queue.put(output)
        self.manager.stats['timer_materialization_cpu_ns'] += time.thread_time_ns()-start

    def make(self, new_token_ids, pooling_output, finish_reason, stop_reason,
             kv_transfer_params=None, ec_transfer_params=None):
        if pooling_output is not None or kv_transfer_params is not None or ec_transfer_params is not None:
            raise NotImplementedError('pooling/transfer metadata outside frozen scope')
        if self.native:
            output = self.original_make(new_token_ids, pooling_output, finish_reason,
                                        stop_reason, kv_transfer_params, ec_transfer_params)
            self.count_emission(output, 'native')
        else:
            count = self.state.detokenizer.num_output_tokens()-self.offset
            first = self.offset == 0
            immediate = self.manager.arm in ('class_static','lazy_static') and not self.heavy
            threshold = self.pending_bytes >= 1024 if self.manager.arm == 'bytes1024' else count >= 4
            due = (self.pending_since_ns is not None and
                   time.perf_counter_ns()-self.pending_since_ns >= self.manager.max_wait_ns)
            if finish_reason is not None or first or immediate or threshold or due:
                reason = ('finish' if finish_reason is not None else 'first' if first else
                          'class_immediate' if immediate else 'quota' if threshold else 'deadline_on_input')
                output = self.emit(pooling_output, finish_reason, stop_reason,
                                   kv_transfer_params, ec_transfer_params, reason)
            else:
                if self.handle is None and self.pending_since_ns is not None:
                    delay = max(0, self.pending_since_ns+self.manager.max_wait_ns-time.perf_counter_ns())/1e9
                    self.handle = asyncio.get_running_loop().call_later(delay, self.on_timer)
                output = None
        if finish_reason is not None:
            self.manager.final_states[self.state.request_id] = {
                'tokens':self.state.detokenizer.num_output_tokens(),
                'logprob_positions':len(self.lp.logprobs) if self.lp.logprobs is not None else 0,
                'cumulative_logprob':self.lp.cumulative_logprob,
                'finish_reason':str(finish_reason), 'stop_reason':stop_reason}
            self.close()
        return output

    def close(self):
        if self.done:
            return
        self.done = True
        self.reset_pending()
        # Remove bound-method cycles so completed native states are freed just
        # as in the baseline, rather than retained until GC or /stats.
        self.state.__dict__.pop('make_request_output', None)
        self.lp.__dict__.pop('_update_sample_logprobs', None)
        if self.lazy:
            self.lp.__dict__.pop('update_from_output', None)
        self.manager.gates.pop(self.state.request_id, None)


class Materializer:
    def __init__(self, adapter, arm, max_wait_ms):
        assert arm in ARMS
        self.arm, self.processor = arm, adapter.processor
        self.max_wait_ns = int(max_wait_ms*1e6)
        self.pending_bytes = 0
        self.final_states = {}
        self.stats = {k:0 for k in ('sample_materialization_calls','sample_positions_materialized',
            'sample_positions_deferred','raw_merge_calls','raw_merge_copy_bytes',
            'pending_raw_bytes_peak','request_outputs_materialized','tokens_materialized',
            'max_tokens_per_materialization','max_pending_age_ms','max_deadline_overrun_ms',
            'timer_materialization_cpu_ns')}
        self.stats['flush_reasons'] = {}
        self.gates = {rid:RequestGate(self,state,bool(adapter.requests[rid]['heavy']))
                      for rid,state in self.processor.request_states.items()}
        self.original_process = self.processor.process_outputs
        self.processor.process_outputs = self.process

    def process(self, outputs, *args, **kwargs):
        for output in outputs:
            gate = self.gates.get(output.request_id)
            if gate is not None:
                gate.note_input(output)
        return self.original_process(outputs,*args,**kwargs)

    def report(self):
        return {'arm':self.arm,'max_wait_ms':self.max_wait_ns/1e6,
                'byte_definition':'8 bytes per sampled ID + native raw logprob array nbytes; not wire bytes',
                'stats':self.stats,'final_states':self.final_states,
                'pending_bytes_at_end':self.pending_bytes,'active_gates_at_end':len(self.gates)}

    def close(self):
        for gate in list(self.gates.values()):
            gate.close()
        self.processor.process_outputs = self.original_process


def install(adapter, arm, max_wait_ms=75):
    return Materializer(adapter,arm,max_wait_ms)

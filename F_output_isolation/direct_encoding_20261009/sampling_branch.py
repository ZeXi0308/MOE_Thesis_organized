"""Experimental in-process vLLM 0.26 post-sampling branch, before candidate D2H.

Scope: TP=DP=PP=1, synchronous scheduling, no speculative decoding, ordinary
streaming text/top-20, no prompt logprobs or token-ID placeholders. Install before
creating LLMEngine(enable_multiprocessing=False); use DirectFrontend for delivery.
This source is NOT evidence of a successful CUDA/model run.

CPU and GPU share request mapping, snapshot lifetime, progress, and headers.
CPU mode copies compact arrays then uses CPUEncoder; GPU mode encodes on device
then copies a fixed expanded slot plus raw fallback data. No future-token window.
"""
from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace
import threading
import time
import numpy as np


@dataclass
class DeferredLogprobs:
    sampled_ids: np.ndarray
    sampled_values: np.ndarray
    metadata_future: Future


class DeferredBatch:
    def __init__(self, jobs, ids, values):
        self.jobs,self.ids,self.values=jobs,ids,values

    def slice_request(self, req_idx, num_positions):
        if num_positions == 0:
            # Native scheduler may inspect logprobs of an empty stopped output.
            from vllm.v1.outputs import LogprobsLists
            return LogprobsLists(np.empty((0,21),dtype=np.int64),
                np.empty((0,21),dtype=np.float32),np.empty(0,dtype=np.int64),None)
        assert num_positions == 1 and req_idx in self.jobs
        return DeferredLogprobs(self.ids[req_idx:req_idx+1],
            self.values[req_idx:req_idx+1],self.jobs[req_idx])


class TensorProxy:
    def __init__(self, raw, jobs, ids, values, ready):
        self.raw,self.jobs,self.ids,self.values,self.ready=raw,jobs,ids,values,ready

    def __getattr__(self, name):return getattr(self.raw,name)

    def tolists(self, *args, **kwargs):
        assert not args and not kwargs
        # _to_list(sampled_ids) already synchronized the producer stream after
        # these tiny headers. Query verifies that dependency; never add a wait.
        assert self.ready.query(), 'native sampled-ID ordering differs from v0.26'
        return DeferredBatch(self.jobs,self.ids.numpy().copy(),self.values.numpy().copy())


class SamplingBranch:
    def __init__(self, table, mode, *, max_requests=64, max_inflight=256):
        assert mode in ('cpu','gpu')
        import torch
        from gpu_async_encoder import GPUAsyncEncoder
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        self.torch=torch;self.mode=mode;self.max_requests=max_requests
        self.backend=GPUAsyncEncoder(table,max_inflight=max_inflight,
            max_rows_per_request=1,num_streams=4,device=torch.cuda.current_device())
        self.header_ids=torch.empty(max_requests,dtype=torch.int64,pin_memory=True)
        self.header_values=torch.empty(max_requests,dtype=torch.float32,pin_memory=True)
        self.header_ready=torch.cuda.Event()
        self.runner_class=GPUModelRunner;self.original=GPUModelRunner._bookkeeping_sync
        self.lock=threading.Lock();self.pending=[];self.stopping=False
        self.wakeup=threading.Event()
        self.stats={'batches':0,'requests':0,'compact_fallbacks':0,
                    'control_cpu_ns':0,'progress_cpu_ns':0,'header_d2h_bytes':0,
                    'request_metrics':[],'cancelled_metadata_results':0}
        self.device=torch.cuda.current_device()
        self.progress=threading.Thread(target=self._progress,name='f-output-progress',daemon=True)
        self.progress.start()
        owner=self
        def wrapped(runner,scheduler_output,sampler_output,*args,**kwargs):
            raw=sampler_output.logprobs_tensors
            if raw is None:return owner.original(runner,scheduler_output,sampler_output,*args,**kwargs)
            # Refuse unsupported global semantics BEFORE enqueueing anything.
            assert not runner.use_async_scheduling
            assert sampler_output.sampled_token_ids.shape[-1] == 1
            assert raw.cu_num_generated_tokens is None
            requested=runner.input_batch.num_logprobs
            assert requested and all(value == 20 for value in requested.values())
            assert not runner.input_batch.logprob_token_ids
            assert raw.logprob_token_ids.is_cuda and raw.logprobs.is_cuda
            assert raw.logprobs.shape[1] == 21
            started=time.thread_time_ns();jobs={}
            producer=torch.cuda.current_stream()
            discarded=set(np.nonzero(runner.discard_request_mask.np[:runner.input_batch.num_reqs])[0].tolist())
            for i,rid in enumerate(runner.input_batch.req_ids):
                if rid not in requested or i in discarded:continue
                jobs[i]=owner._submit(raw,i,producer,rid)
            count=raw.logprobs.shape[0]
            assert count <= max_requests
            # No candidate array has crossed D2H before submission. Only these
            # sampled scalar fields join the native generation dependency.
            owner.header_ids[:count].copy_(raw.logprob_token_ids[:,0],non_blocking=True)
            owner.header_values[:count].copy_(raw.logprobs[:,0],non_blocking=True)
            owner.header_ready.record(producer)
            owner.stats['header_d2h_bytes']+=count*12
            owner.stats['batches']+=1
            owner.stats['control_cpu_ns']+=time.thread_time_ns()-started
            sampler_output.logprobs_tensors=TensorProxy(raw,jobs,owner.header_ids[:count],
                owner.header_values[:count],owner.header_ready)
            try:return owner.original(runner,scheduler_output,sampler_output,*args,**kwargs)
            finally:sampler_output.logprobs_tensors=raw
        GPUModelRunner._bookkeeping_sync=wrapped

    def _submit(self, raw, i, producer, rid):
        from gpu_async_encoder import CompatibilityRequired
        ids=raw.logprob_token_ids[i:i+1]
        values=raw.logprobs[i:i+1]
        ranks=raw.selected_token_ranks[i:i+1]
        future=Future()
        ticket=self.backend.submit_device(ids,values,ranks,producer,encode=self.mode=='gpu')
        if isinstance(ticket,CompatibilityRequired):
            # Ring pressure does not block the frontend or change admission.
            # This slower allocation fallback is charged, never silently lost.
            transfer=self._compact_fallback(ids,values,ranks,producer)
            self.stats['compact_fallbacks']+=1
            item=(future,None,transfer,rid)
        else:item=(future,ticket,None,rid)
        with self.lock:self.pending.append(item)
        self.stats['requests']+=1;self.wakeup.set()
        return future

    def _compact_fallback(self, ids, values, ranks, producer):
        torch=self.torch
        # Snapshot on the producer stream precedes reuse by the next inference.
        with torch.cuda.stream(producer):
            snapshots=tuple(x.clone() for x in (ids,values,ranks))
            snapshot_ready=torch.cuda.Event();snapshot_ready.record(producer)
        hosts=tuple(torch.empty_like(x,device='cpu',pin_memory=True) for x in snapshots)
        stream=torch.cuda.Stream();ready=torch.cuda.Event()
        with torch.cuda.stream(stream):
            stream.wait_event(snapshot_ready)
            for host,device in zip(hosts,snapshots):host.copy_(device,non_blocking=True)
            ready.record(stream)
        size=sum(x.numel()*x.element_size() for x in snapshots)
        return dict(snapshots=snapshots,hosts=hosts,stream=stream,ready=ready,
            snapshot_ready=snapshot_ready,metrics={'compatibility_fallback':True,
                'compact_d2h_bytes':size,'snapshot_d2d_bytes':size})

    def _progress(self):
        self.torch.cuda.set_device(self.device)
        while True:
            with self.lock:busy=bool(self.pending)
            self.wakeup.wait(.0005 if busy else None);self.wakeup.clear()
            start=time.thread_time_ns()
            with self.lock:items=list(self.pending)
            for item in items:
                future,ticket,fallback,rid=item
                try:
                    if fallback is not None:
                        if not fallback['ready'].query():continue
                        ids,values,ranks=(x.numpy().copy() for x in fallback['hosts'])
                        encoded=None;metrics=fallback['metrics']
                    else:
                        if not self.backend.poll_ready(ticket):continue
                        result=self.backend.collect(ticket)
                        assert result is not None
                        ids,values,ranks=result.raw_ids,result.raw_values,result.raw_ranks
                        encoded=result.encoded;metrics=result.metrics
                        assert self.backend.release(ticket)
                    value=SimpleNamespace(logprob_token_ids=ids,logprobs=values,
                        sampled_token_ranks=ranks,encoded=encoded)
                    self.stats['request_metrics'].append(dict(request_id=rid,**metrics))
                    if future.cancelled():self.stats['cancelled_metadata_results']+=1
                    elif not future.done():future.set_result(value)
                except BaseException as exc:
                    if not future.done():future.set_exception(exc)
                with self.lock:self.pending.remove(item)
            self.stats['progress_cpu_ns']+=time.thread_time_ns()-start
            with self.lock:
                if self.stopping and not self.pending:return

    def close(self):
        self.runner_class._bookkeeping_sync=self.original
        with self.lock:self.stopping=True
        self.wakeup.set();self.progress.join(timeout=30)
        if self.progress.is_alive():raise RuntimeError('pending device jobs did not drain')
        self.backend.close()

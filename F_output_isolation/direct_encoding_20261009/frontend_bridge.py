"""Direct logprob fragments inside unchanged native text/stop/SSE processing.

Each request submits only its own already-ready positions to one CPU worker.
Native text/stop processing never waits for that worker. The request SSE iterator
awaits only its own metadata. No timer, priority, or future-token coalescing. Unsupported API variants
remain native. The bridge retains outer native response models but substitutes
the logprob JSON subtree, avoiding its repeated per-candidate object tree.
"""
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import asyncio
import inspect
import time
import numpy as np
from pydantic import PrivateAttr
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProbs, ChatCompletionStreamResponse)
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.logprobs import append_logprobs_for_next_position
from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens


class FragmentLogprobs(ChatCompletionLogProbs):
    _fragment: str = PrivateAttr(default='')


@dataclass(frozen=True)
class PendingFragment:
    future: object
    row: int


class DirectFrontend:
    def __init__(self, processor, chat, tokenizer, encoder):
        self.processor,self.chat,self.tokenizer,self.encoder=processor,chat,tokenizer,encoder
        self.original_process=processor.process_outputs
        self.original_abort=processor.abort_requests
        self.original_chat=chat._create_chat_logprobs
        self.original_stream=chat.chat_completion_stream_generator
        self.stream_signature=inspect.signature(self.original_stream)
        # ctypes releases the GIL in the C++ encoder. Worker inherits the same
        # process CPU affinity; benchmarks must not add a core for this arm.
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix="f-output")
        self.original_dump=ChatCompletionStreamResponse.model_dump_json
        self.histories={};self.original_updates={}
        self.stats={'rows':0,'fallback_rows':0,'batch_calls':0,'encoded_bytes':0,
                    'packing_cpu_ns':0,'encoding_cpu_ns':0,'fallback_cpu_ns':0,
                    'request_jobs':0,'request_wait_ns':0}
        self._install_serializer()
        processor.process_outputs=self.process
        processor.abort_requests=self.abort
        chat._create_chat_logprobs=self.chat_logprobs
        chat.chat_completion_stream_generator=self.stream

    def _install_serializer(self):
        original=self.original_dump
        def dump(model,*args,**kw):
            fragments=[c.logprobs for c in model.choices if isinstance(c.logprobs,FragmentLogprobs)]
            result=original(model,*args,**kw)
            if not fragments:return result
            assert len(fragments)==1 and len(model.choices)==1
            marker='"logprobs":{"content":[]}'
            assert result.count(marker)==1, 'Unexpected native serialization shape'
            return result.replace(marker,'"logprobs":'+fragments[0]._fragment,1)
        ChatCompletionStreamResponse.model_dump_json=dump

    def chat_logprobs(self, token_ids, top_logprobs, tokenizer, **kw):
        if not token_ids or not top_logprobs or not isinstance(top_logprobs[0],bytes):
            return self.original_chat(token_ids,top_logprobs,tokenizer,**kw)
        assert len(token_ids)==len(top_logprobs)
        assert kw.get('num_output_top_logprobs')==20 and not kw.get('logprob_token_ids')
        assert not kw.get('return_as_token_id')
        response=FragmentLogprobs(content=[])
        response._fragment=(b'{"content":['+b','.join(top_logprobs)+b']}').decode('utf-8')
        return response

    def fallback(self, rid, ids, values, rank):
        # Retained for encoder-only comparisons with explicit per-request history.
        lp=self.processor.request_states[rid].logprobs_processor
        return self._fallback_context(lp,list(self.histories[rid]),ids,values,rank)

    def _fallback_context(self, lp, context, ids, values, rank):
        start=time.thread_time_ns()
        token_ids=ids.tolist(); probs=values.tolist()
        decoded=convert_ids_list_to_tokens(self.tokenizer,token_ids)
        decoded=lp._verify_tokens(decoded,token_ids,context)
        container=[]
        append_logprobs_for_next_position(container,token_ids,probs,decoded,int(rank),20)
        value=self.original_chat([int(ids[0])],container,self.tokenizer,
                                 num_output_top_logprobs=20)
        fragment=value.content[0].model_dump_json().encode('utf-8')
        self.stats['fallback_cpu_ns']+=time.thread_time_ns()-start
        self.stats['fallback_rows']+=1
        return fragment

    def _encode_job(self, lp, context, raw):
        start=time.thread_time_ns()
        # The numpy payload owns/retains its CPU backing allocation; no mutation
        # is made by this bridge. Conversion, if needed, belongs to this job.
        ids=np.ascontiguousarray(raw.logprob_token_ids,dtype=np.int64)
        values=np.ascontiguousarray(raw.logprobs,dtype=np.float32)
        self.stats['packing_cpu_ns']+=time.thread_time_ns()-start
        start=time.thread_time_ns()
        encoded=getattr(raw,'encoded',None)
        if encoded is None:encoded=self.encoder.encode(ids,values,top_k=20)
        self.stats['encoding_cpu_ns']+=time.thread_time_ns()-start
        self.stats['batch_calls']+=1
        self.stats['rows']+=len(ids)
        self.stats['encoded_bytes']+=len(encoded.packed)
        history=deque(context,maxlen=4); fragments=[]
        for i in range(len(ids)):
            status=int(encoded.status[i])
            assert status in (0,1),('invalid direct encoding input',status)
            fragment=(self._fallback_context(lp,list(history),ids[i],values[i],
                raw.sampled_token_ranks[i]) if status else encoded.row(i))
            fragments.append(fragment)
            history.append(int(ids[i,0]))
        return fragments

    def _attach(self, rid, state):
        if rid in self.original_updates:return
        lp=state.logprobs_processor
        assert lp.num_logprobs==20 and lp.num_prompt_logprobs is None
        assert not lp.logprobs, 'Install before the first output of each request'
        self.histories[rid]=deque(maxlen=4)
        self.original_updates[rid]=(lp,lp.update_from_output)
        def update(output):
            assert output.new_prompt_logprobs_tensors is None
            raw=output.new_logprobs
            if raw is None:return
            context=list(self.histories[rid])
            if hasattr(raw,'metadata_future'):
                # A sampling branch reports each request independently. Never
                # occupy the CPU worker while waiting for a device transfer.
                future=Future()
                def ready(metadata):
                    try:
                        job=self.worker.submit(self._encode_job,lp,context,metadata.result())
                        def finished(job):
                            try:future.set_result(job.result())
                            except BaseException as exc:future.set_exception(exc)
                        job.add_done_callback(finished)
                    except BaseException as exc:future.set_exception(exc)
                raw.metadata_future.add_done_callback(ready)
                sampled_values=raw.sampled_values
                sampled_ids=raw.sampled_ids
            else:
                future=self.worker.submit(self._encode_job,lp,context,raw)
                sampled_values=raw.logprobs[:,0]
                sampled_ids=raw.logprob_token_ids[:,0]
            self.stats['request_jobs']+=1
            for i in range(len(sampled_ids)):
                lp.cumulative_logprob+=float(sampled_values[i])
                lp.logprobs.append(PendingFragment(future,i))
                self.histories[rid].append(int(sampled_ids[i]))
        lp.update_from_output=update

    def process(self, outputs,*args,**kw):
        for output in outputs:
            state=self.processor.request_states.get(output.request_id)
            if state is None or output.new_logprobs is None:continue
            lp=state.logprobs_processor
            if (lp.num_logprobs!=20 or lp.num_prompt_logprobs is not None or state.parent_req is not None
                    or state.streaming_input):continue
            self._attach(output.request_id,state)
        result=self.original_process(outputs,*args,**kw)
        self._cleanup_finished()
        return result

    async def _ready(self, source):
        try:
            async for result in source:
                for output in result.outputs:
                    # Native delta slicing with -0 can expose all prior rows on
                    # abort. The zero-token terminal message requires none.
                    if not output.token_ids or not output.logprobs:continue
                    resolved=[]; cache={}
                    for item in output.logprobs[:len(output.token_ids)]:
                        if isinstance(item,PendingFragment):
                            if item.future not in cache:
                                start=time.perf_counter_ns()
                                cache[item.future]=await asyncio.shield(asyncio.wrap_future(item.future))
                                self.stats['request_wait_ns']+=time.perf_counter_ns()-start
                            resolved.append(cache[item.future][item.row])
                        else:resolved.append(item)
                    output.logprobs=resolved
                yield result
        finally:
            # Preserve generator-finally cancellation/collector cleanup.
            close=getattr(source,'aclose',None)
            if close is not None:await close()

    def stream(self,*args,**kw):
        bound=self.stream_signature.bind(*args,**kw)
        bound.arguments['result_generator']=self._ready(bound.arguments['result_generator'])
        return self.original_stream(*bound.args,**bound.kwargs)

    def abort(self,*args,**kw):
        result=self.original_abort(*args,**kw)
        self._cleanup_finished()
        return result

    def _cleanup_finished(self):
        for rid in list(self.original_updates):
            if rid not in self.processor.request_states:
                lp,original=self.original_updates.pop(rid)
                lp.update_from_output=original
                self.histories.pop(rid,None)

    def close(self):
        self.processor.process_outputs=self.original_process
        self.processor.abort_requests=self.original_abort
        self.chat._create_chat_logprobs=self.original_chat
        self.chat.chat_completion_stream_generator=self.original_stream
        ChatCompletionStreamResponse.model_dump_json=self.original_dump
        for lp,original in self.original_updates.values():lp.update_from_output=original
        self.original_updates.clear();self.histories.clear()
        self.worker.shutdown(wait=True,cancel_futures=True)

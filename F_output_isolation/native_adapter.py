"""CPU replay seam around the installed vLLM 0.26.0 output path.

Only the model executor is replaced. OutputProcessor, its detokenizer and
LogprobsProcessor, RequestOutputCollector, API response models, and the chat SSE
generator are imported unmodified from the installed runtime. No GPU is used.
"""
from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from bootstrap import apply

BOOTSTRAP_SKIPPED_ORPHANS = apply()

import numpy as np
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import RequestOutput, STREAM_FINISHED
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine import (
    EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason,
)
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.outputs import LogprobsLists


DEFAULT_MODEL = "/root/autodl-tmp/moe-research-20261002/model"


class NativeAdapter:
    def __init__(self, requests: list[dict], model_path: str = DEFAULT_MODEL,
                 top_logprobs: int = 20):
        self.model_path = str(Path(model_path))
        self.top_logprobs = top_logprobs
        self.tokenizer = get_tokenizer(self.model_path, trust_remote_code=False)
        self.processor = OutputProcessor(self.tokenizer, log_stats=False,
                                         stream_interval=1)
        self.requests = {r["request_id"]: r for r in requests}
        self.collectors = {}
        self.api_requests = {}
        self.metadata = {}
        self.ledger = {rid: [] for rid in self.requests}
        self._logprobs = {}
        self.chat = self._make_chat_serving()
        self.model_name = "native-cpu-replay"

        # Stable legal vocabulary IDs. Sampled token is first, as in vLLM's
        # sampler; the other ranks are synthetic because replay is CPU-only.
        words = ("The response contains words about systems computer memory "
                 "performance isolation scheduling latency bandwidth output "
                 "processor tokenizer requests completion scientific analysis "
                 "engineering numbers model probability streaming language.")
        alternatives = list(dict.fromkeys(self.tokenizer.encode(words)))
        if len(alternatives) < top_logprobs + 1:
            alternatives = [i for i in range(len(self.tokenizer))
                            if i not in set(self.tokenizer.all_special_ids)]

        for rid, spec in self.requests.items():
            tokens = list(spec["output_token_ids"])
            heavy = bool(spec.get("heavy", False))
            prompt_ids = list(spec["prompt_token_ids"])
            request = ChatCompletionRequest(
                model=self.model_name,
                messages=[{"role": "user", "content": "Replay fixed output."}],
                stream=True, max_tokens=len(tokens),
                logprobs=heavy, top_logprobs=top_logprobs if heavy else None,
                stream_options={"include_usage": True},
            )
            params = request.to_sampling_params(len(tokens), {})
            assert params.output_kind == RequestOutputKind.DELTA
            core_request = EngineCoreRequest(
                request_id=rid, external_req_id=rid,
                prompt_token_ids=prompt_ids, mm_features=None,
                sampling_params=params, pooling_params=None,
                arrival_time=0.0, lora_request=None, cache_salt=None,
                data_parallel_rank=None,
            )
            collector = RequestOutputCollector(RequestOutputKind.DELTA, rid)
            self.processor.add_request(
                core_request, self.tokenizer.decode(prompt_ids), queue=collector)
            self.collectors[rid] = collector
            self.api_requests[rid] = request
            self.metadata[rid] = RequestResponseMetadata(request_id=rid)
            if heavy:
                # vLLM emits sampled ID followed by top-k IDs. A rank-one
                # sample therefore occurs twice in the native array and is
                # deduplicated by the unchanged LogprobsProcessor.
                ids = np.asarray([
                    [token, token] + [i for i in alternatives if i != token][:top_logprobs - 1]
                    for token in tokens
                ], dtype=np.int64)
                logits = -np.arange(top_logprobs, dtype=np.float32) * 0.5
                top_values = logits - np.log(np.exp(logits).sum())
                values = np.concatenate((top_values[:1], top_values))
                self._logprobs[rid] = LogprobsLists(
                    ids, np.tile(values, (len(tokens), 1)),
                    np.ones(len(tokens), dtype=np.int64), None,
                )

    @staticmethod
    def _make_chat_serving():
        # __init__ loads the model/renderer for request input processing. The
        # replay seam already has the resulting token IDs, so only configure
        # fields read by the unchanged output generator.
        chat = OpenAIServingChat.__new__(OpenAIServingChat)
        chat.parser_cls = None
        chat.response_role = "assistant"
        chat.enable_force_include_usage = False
        chat.enable_log_outputs = False
        chat.enable_log_deltas = False
        chat.request_logger = None
        chat.enable_prompt_tokens_details = False
        chat.enable_per_request_metrics = False
        chat.return_tokens_as_token_ids = False
        chat.system_fingerprint = None
        return chat

    def build_batch(self, batch: dict) -> EngineCoreOutputs:
        """Preconstruct a native batch, outside the timed delivery path.

        Each output accepts token_start/token_end (half-open), or one token's
        position/token_id. Native engine order and batch boundaries are kept.
        """
        outputs = []
        for item in batch["outputs"]:
            rid = item["request_id"]
            spec = self.requests[rid]
            start = item.get("token_start", item.get("position", 0))
            end = item.get("token_end", start + 1)
            token_ids = list(spec["output_token_ids"][start:end])
            if "token_id" in item:
                assert token_ids == [item["token_id"]]
            if "new_token_ids" in item:
                assert token_ids == item["new_token_ids"]
            finished = item.get("finished", end == len(spec["output_token_ids"]))
            lps = self._logprobs.get(rid)
            if lps is not None:
                lps = LogprobsLists(lps.logprob_token_ids[start:end],
                                   lps.logprobs[start:end],
                                   lps.sampled_token_ranks[start:end], None)
            outputs.append(EngineCoreOutput(
                request_id=rid, new_token_ids=token_ids, new_logprobs=lps,
                finish_reason=FinishReason.LENGTH if finished else None,
            ))
        return EngineCoreOutputs(outputs=outputs,
                                 timestamp=batch.get("timestamp", batch.get("time_s", 0)))

    def feed(self, batch: dict | EngineCoreOutputs):
        if isinstance(batch, dict):
            batch = self.build_batch(batch)
        return self.processor.process_outputs(batch.outputs, batch.timestamp)

    async def _results(self, rid: str) -> AsyncIterator[RequestOutput]:
        q = self.collectors[rid]
        try:
            finished = False
            while not finished:
                # Exact normal-path queue-draining semantics of
                # AsyncLLM.generate; no artificial fairness yield here.
                out = q.get_nowait() or await q.get()
                assert isinstance(out, RequestOutput)
                finished = out.finished
                if out is not STREAM_FINISHED:
                    for output in out.outputs:
                        self.ledger[rid].append({
                            "token_count": len(output.token_ids),
                            "consumed_ns": time.perf_counter_ns(),
                        })
                    yield out
        finally:
            q.close()

    def stream(self, rid: str):
        return self.chat.chat_completion_stream_generator(
            request=self.api_requests[rid], result_generator=self._results(rid),
            request_id=rid, model_name=self.model_name, conversation=[],
            tokenizer=self.tokenizer, request_metadata=self.metadata[rid],
        )

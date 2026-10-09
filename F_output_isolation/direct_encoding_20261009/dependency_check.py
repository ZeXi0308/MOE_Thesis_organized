"""Controlled dependency checks using the native tokenizer, collector and SSE.

The encoder is deliberately stopped with threading.Event, not made artificially
fast or measured as a performance workload. A single worker still serializes
encoding jobs: this checks removal of the all-request completion barrier, not
that a later heavy request can overtake a blocked earlier encoding job.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import Future
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ControlledEncoder:
    """One explicit gate per submitted encode; actual bytes use CPUEncoder."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.loop_thread = threading.get_ident()
        self.lock = threading.Lock()
        self.calls = []
        self.auto_release = False

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def encode(self, token_ids, logprobs, **kwargs):
        assert threading.get_ident() != self.loop_thread, \
            "metadata encoder was called on the event-loop thread"
        call = {"gate": threading.Event(), "completed": threading.Event(),
                "rows": len(token_ids), "sampled_ids": token_ids[:, 0].tolist()}
        with self.lock:
            self.calls.append(call)
            if self.auto_release: call["gate"].set()
        if not call["gate"].wait(20):
            raise TimeoutError("controlled encoder watchdog; the test did not release this job")
        value = self.encoder.encode(token_ids, logprobs, **kwargs)
        call["completed"].set()
        return value

    def call(self, index):
        with self.lock:
            return self.calls[index] if len(self.calls) > index else None

    def release(self, index):
        call = self.call(index)
        assert call is not None, ("encoder job has not started", index)
        call["gate"].set()

    def release_all(self):
        with self.lock:
            self.auto_release = True
            for call in self.calls:
                call["gate"].set()


class Streams:
    def __init__(self, adapter):
        self.adapter = adapter
        self.payloads = {rid: [] for rid in adapter.requests}
        self.done = dict.fromkeys(adapter.requests, 0)
        self.tasks = {rid: asyncio.create_task(self.consume(rid)) for rid in adapter.requests}

    async def consume(self, rid):
        async for chunk in self.adapter.stream(rid):
            assert chunk.startswith("data: "), (rid, chunk)
            raw = chunk[6:].strip()
            if raw == "[DONE]":
                self.done[rid] += 1
            else:
                value = json.loads(raw)
                assert "error" not in value, (rid, value)
                self.payloads[rid].append(value)

    def check_errors(self):
        for task in self.tasks.values():
            if task.done(): task.result()

    async def until(self, condition, detail):
        async def poll():
            while not condition():
                self.check_errors()
                await asyncio.sleep(.001)
            self.check_errors()
        try:
            await asyncio.wait_for(poll(), timeout=5)
        except asyncio.TimeoutError as exc:
            raise AssertionError(detail) from exc

    def text(self, rid):
        return "".join(choice.get("delta", {}).get("content") or ""
            for message in self.payloads[rid] for choice in message.get("choices", []))

    def positions(self, rid):
        return [row for message in self.payloads[rid] for choice in message.get("choices", [])
            for row in (choice.get("logprobs") or {}).get("content", [])]

    def semantic(self):
        result = {}
        for rid, messages in self.payloads.items():
            finishes, usage, roles = [], [], []
            for message in messages:
                if message.get("usage") is not None: usage.append(message["usage"])
                for choice in message.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finishes.append([choice["finish_reason"], choice.get("stop_reason")])
                    role = choice.get("delta", {}).get("role")
                    if role is not None: roles.append(role)
            assert len(finishes) == len(usage) == self.done[rid] == 1, (rid, finishes, usage, self.done[rid])
            assert roles == ["assistant"]
            count = sum(x["token_count"] for x in self.adapter.ledger[rid])
            assert usage[0]["completion_tokens"] == count
            rows = self.positions(rid)
            assert len(rows) == (count if self.adapter.requests[rid]["heavy"] else 0)
            result[rid] = {"text": self.text(rid), "logprobs": rows,
                "finish": finishes, "usage": usage, "roles": roles,
                "done": self.done[rid], "token_count": count}
        return result

    async def close(self):
        for task in self.tasks.values():
            if not task.done(): task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)


def snapshot(adapter, states):
    return {rid: {"token_ids": list(state.detokenizer.output_token_ids),
                  "text": state.detokenizer.output_text,
                  "cumulative_logprob": state.logprobs_processor.cumulative_logprob,
                  "registered": rid in adapter.processor.request_states}
            for rid, state in states.items()}


async def close_bridge(bridge):
    if bridge:
        value = bridge.close()
        if inspect.isawaitable(value): await value


async def ready_batch_case(model, specs, encoder, arm):
    from native_adapter import NativeAdapter
    from frontend_bridge import DirectFrontend
    adapter = NativeAdapter(specs, model_path=model)
    states = dict(adapter.processor.request_states)
    controlled = ControlledEncoder(encoder) if arm == "direct" else None
    bridge = DirectFrontend(adapter.processor, adapter.chat, adapter.tokenizer, controlled) if controlled else None
    streams = Streams(adapter)
    # Both heavy jobs and the light output are ready BEFORE the one feed call.
    batch = {"outputs": [{"request_id": rid, "token_start": 0,
                          "token_end": len(spec["output_token_ids"]), "finished": True}
                         for rid, spec in adapter.requests.items()]}
    checks = {}
    try:
        adapter.feed(batch)
        state_after_feed = snapshot(adapter, states)
        if controlled:
            await streams.until(lambda: controlled.call(0) is not None, "first encoding job never started")
            first = controlled.call(0)
            assert not first["gate"].is_set() and not first["completed"].is_set()
            assert first["sampled_ids"] == specs[0]["output_token_ids"], \
                "first submission spans requests instead of using its own result"
            await streams.until(lambda: streams.done["plain"] == 1,
                "plain output waited for deliberately blocked heavy metadata")
            assert streams.text("plain")
            assert not first["gate"].is_set(), "test accidentally released the dependency"
            assert not streams.positions("heavy1") and not streams.positions("heavy2")
            checks["plain_delivered_while_heavy_encoder_blocked"] = True
            controlled.release(0)
            await streams.until(lambda: controlled.call(1) is not None, "second encoding job never started")
            second = controlled.call(1)
            assert not second["gate"].is_set() and not second["completed"].is_set()
            assert second["sampled_ids"] == specs[1]["output_token_ids"]
            await streams.until(lambda: streams.done["heavy1"] == 1,
                "first heavy output waited for the other request's metadata")
            assert streams.positions("heavy1")
            assert not streams.positions("heavy2") and streams.done["heavy2"] == 0
            assert not second["gate"].is_set()
            checks["heavy1_delivered_while_heavy2_encoder_blocked"] = True
            controlled.release(1)
        await streams.until(lambda: all(streams.done.values()), "not all complete responses were delivered")
        await asyncio.gather(*streams.tasks.values())
        semantic = streams.semantic()
        assert not adapter.processor.request_states
        assert all(q.output is None for q in adapter.collectors.values())
        return {"semantic": semantic, "state_after_feed": state_after_feed,
                "checks": checks, "bridge_stats": dict(bridge.stats) if bridge else None}
    finally:
        if controlled: controlled.release_all()
        await streams.close()
        await close_bridge(bridge)


async def abort_case(model, specs, encoder, arm):
    from native_adapter import NativeAdapter
    from frontend_bridge import DirectFrontend
    adapter = NativeAdapter(specs, model_path=model)
    states = dict(adapter.processor.request_states)
    controlled = ControlledEncoder(encoder) if arm == "direct" else None
    bridge = DirectFrontend(adapter.processor, adapter.chat, adapter.tokenizer, controlled) if controlled else None
    streams = Streams(adapter)
    try:
        adapter.feed({"outputs": [
            {"request_id": "abort_heavy", "token_start": 0, "token_end": 2, "finished": False},
            {"request_id": "abort_plain", "token_start": 0,
             "token_end": len(adapter.requests["abort_plain"]["output_token_ids"]), "finished": True}]})
        after_feed = snapshot(adapter, states)
        if controlled:
            await streams.until(lambda: controlled.call(0) is not None, "abort encoder job never started")
            await streams.until(lambda: streams.done["abort_plain"] == 1, "plain delivery blocked in abort fixture")
            assert not controlled.call(0)["gate"].is_set()
            assert not streams.positions("abort_heavy")
        else:
            # Let native consume its first output before placing the separate
            # abort output. The direct consumer has pulled the same output but
            # is suspended solely on its metadata Future.
            await streams.until(lambda: len(streams.positions("abort_heavy")) == 2,
                                "native first output was not consumed")
        aborted = adapter.processor.abort_requests(["abort_heavy"], internal=True)
        assert aborted == ["abort_heavy"]
        assert "abort_heavy" not in adapter.processor.request_states
        after_abort = snapshot(adapter, states)
        if bridge:
            # Pending immutable response data may live until consumed. Active
            # native state and its monkey-patched updater must not survive abort.
            for name in ("original_updates", "histories"):
                active = getattr(bridge, name, {})
                assert "abort_heavy" not in active, ("abort retains active bridge state", name)
            assert not controlled.call(0)["gate"].is_set()
            assert not controlled.call(0)["completed"].is_set()
            controlled.release(0)
        await streams.until(lambda: all(streams.done.values()), "ABORT SSE did not drain after its metadata became ready")
        await asyncio.gather(*streams.tasks.values())
        semantic = streams.semantic()
        assert semantic["abort_heavy"]["finish"] == [["abort", None]]
        assert semantic["abort_heavy"]["token_count"] == 2
        before = {rid: len(adapter.ledger[rid]) for rid in adapter.requests}
        adapter.feed({"outputs": [{"request_id": "abort_heavy", "position": 2, "finished": False}]})
        await asyncio.sleep(0)
        assert before == {rid: len(adapter.ledger[rid]) for rid in adapter.requests}
        assert all(q.output is None for q in adapter.collectors.values())
        return {"semantic": semantic, "state_after_feed": after_feed,
                "state_after_abort": after_abort,
                "abort_returned_while_encoder_blocked": bool(controlled),
                "late_aborted_output_ignored": True}
    finally:
        if controlled: controlled.release_all()
        await streams.close()
        await close_bridge(bridge)


async def metadata_future_case(model, specs, encoder):
    """CPU-controlled stand-in for independent per-request D2H completion.

    The sample heads are ready together, but candidate metadata Future 2 is
    completed before Future 1. No CUDA operations or transfer-speed claims.
    """
    from native_adapter import NativeAdapter
    from frontend_bridge import DirectFrontend
    adapter = NativeAdapter(specs, model_path=model)
    states = dict(adapter.processor.request_states)
    bridge = DirectFrontend(adapter.processor, adapter.chat, adapter.tokenizer, encoder)
    streams = Streams(adapter)
    batch = adapter.build_batch({"outputs": [
        {"request_id": rid, "token_start": 0,
         "token_end": len(spec["output_token_ids"]), "finished": True}
        for rid, spec in adapter.requests.items()]})
    futures, complete_raw = {}, {}
    for output in batch.outputs:
        raw = output.new_logprobs
        if raw is None: continue
        rid = output.request_id
        complete_raw[rid] = raw
        futures[rid] = Future()
        # The initial output carries only the sampled ID/probability head and
        # an unresolved completion dependency. Full candidates remain withheld.
        output.new_logprobs = SimpleNamespace(
            sampled_ids=raw.logprob_token_ids[:, 0].copy(),
            sampled_values=raw.logprobs[:, 0].copy(),
            metadata_future=futures[rid])
    assert set(futures) == {"heavy1", "heavy2"}
    try:
        adapter.feed(batch)
        after_feed = snapshot(adapter, states)
        await streams.until(lambda: streams.done["plain"] == 1,
            "plain output waited for unresolved candidate metadata futures")
        assert streams.text("plain")
        assert not futures["heavy1"].done() and not futures["heavy2"].done()
        assert not streams.positions("heavy1") and not streams.positions("heavy2")
        # Reverse completion order explicitly: the first request is still not
        # device-ready, so it must not occupy the sole encoding worker.
        futures["heavy2"].set_result(complete_raw["heavy2"])
        await streams.until(lambda: streams.done["heavy2"] == 1,
            "ready heavy2 metadata waited for unrelated heavy1 completion")
        assert streams.positions("heavy2")
        assert not futures["heavy1"].done()
        assert not streams.positions("heavy1") and streams.done["heavy1"] == 0
        futures["heavy1"].set_result(complete_raw["heavy1"])
        await streams.until(lambda: all(streams.done.values()),
            "responses did not drain after both metadata futures completed")
        await asyncio.gather(*streams.tasks.values())
        semantic = streams.semantic()
        assert not adapter.processor.request_states
        assert not bridge.histories and not bridge.original_updates
        assert all(q.output is None for q in adapter.collectors.values())
        return {"semantic": semantic, "state_after_feed": after_feed,
            "checks": {"plain_delivered_with_both_metadata_futures_pending": True,
                "heavy2_delivered_with_heavy1_metadata_future_pending": True,
                "metadata_completion_order": ["heavy2", "heavy1"],
                "metadata_future_test_uses_cuda": False},
            "bridge_stats": dict(bridge.stats)}
    finally:
        # Release every outstanding artificial dependency even after failure;
        # the fixture must not leave a real worker or async consumer waiting.
        for rid, future in futures.items():
            if not future.done(): future.set_result(complete_raw[rid])
        await streams.close()
        await close_bridge(bridge)


async def consumer_cancel_case(model, specs, encoder):
    from native_adapter import NativeAdapter
    from frontend_bridge import DirectFrontend
    adapter = NativeAdapter(specs, model_path=model)
    controlled = ControlledEncoder(encoder)
    bridge = DirectFrontend(adapter.processor, adapter.chat, adapter.tokenizer, controlled)
    streams = Streams(adapter)
    try:
        adapter.feed({"outputs": [
            {"request_id": "abort_heavy", "token_start": 0, "token_end": 2, "finished": False},
            {"request_id": "abort_plain", "token_start": 0,
             "token_end": len(adapter.requests["abort_plain"]["output_token_ids"]), "finished": True}]})
        await streams.until(lambda: controlled.call(0) is not None, "cancel encoder job never started")
        await streams.until(lambda: streams.done["abort_plain"] == 1, "plain delivery blocked in cancel fixture")
        task = streams.tasks["abort_heavy"]
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelling the SSE consumer did not cancel its generator")
        assert task.cancelled()
        assert not controlled.call(0)["gate"].is_set()
        assert not controlled.call(0)["completed"].is_set()
        # NativeAdapter intentionally lacks AsyncLLM's transport cancellation
        # hook. Invoke the processor abort explicitly, as the real owner would.
        assert adapter.processor.abort_requests(["abort_heavy"], internal=True) == ["abort_heavy"]
        assert not adapter.processor.request_states
        controlled.release(0)
        return {"sse_consumer_cancels_while_encoder_blocked": True,
                "disconnected_response_completion_claimed": False,
                "running_native_encoding_is_preempted": False}
    finally:
        controlled.release_all()
        await streams.close()
        await close_bridge(bridge)


async def run(model):
    from native_adapter import get_tokenizer
    from cpu_encoder import build_token_table, CPUEncoder
    tokenizer = get_tokenizer(model, trust_remote_code=False)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prompt = encode("Check independent metadata dependencies.")
    specs = [{"request_id": rid, "heavy": heavy, "prompt_token_ids": prompt,
              "output_token_ids": encode(text)} for rid, heavy, text in (
                  ("heavy1", True, "First independent heavy response."),
                  ("heavy2", True, "Second independent heavy response."),
                  ("plain", False, "Ordinary text is immediately ready."))]
    table = build_token_table(tokenizer)
    abort_specs = [{"request_id": rid, "heavy": heavy, "prompt_token_ids": prompt,
                    "output_token_ids": encode(text)} for rid, heavy, text in (
                        ("abort_heavy", True, "Generated first second third fourth fifth."),
                        ("abort_plain", False, "Plain output remains ready during abort."))]
    assert len(abort_specs[0]["output_token_ids"]) >= 4
    with CPUEncoder(table) as encoder:
        native = await ready_batch_case(model, specs, encoder, "native")
        direct = await ready_batch_case(model, specs, encoder, "direct")
        metadata = await metadata_future_case(model, specs, encoder)
        abort_native = await abort_case(model, abort_specs, encoder, "native")
        abort_direct = await abort_case(model, abort_specs, encoder, "direct")
        cancelled = await consumer_cancel_case(model, abort_specs, encoder)
    assert native["state_after_feed"] == direct["state_after_feed"], "generation state waited for metadata encoding"
    assert native["semantic"] == direct["semantic"], "full native SSE semantics changed"
    assert native["state_after_feed"] == metadata["state_after_feed"], \
        "native generation state depends on full candidate transfer completion"
    assert native["semantic"] == metadata["semantic"], \
        "out-of-order metadata completion changed full native SSE semantics"
    assert abort_native["state_after_feed"] == abort_direct["state_after_feed"]
    assert abort_native["state_after_abort"] == abort_direct["state_after_abort"]
    assert abort_native["semantic"] == abort_direct["semantic"], "ABORT lost or changed previously generated fields"
    return {"ok": True, "performance_claim": False, "gpu_used": False,
            "controlled_blocking": "threading.Event per encode; event released only after dependency assertion",
            "same_ready_batch": True, "checks": {**direct["checks"],
                **metadata["checks"],
                "generation_state_eager_and_equal": True,
                "full_native_sse_semantics_equal": True,
                "metadata_future_full_native_sse_semantics_equal": True,
                "processor_abort_returns_while_encoder_blocked": abort_direct["abort_returned_while_encoder_blocked"],
                "full_abort_response_equal_after_metadata_ready": True,
                "late_aborted_output_ignored": abort_direct["late_aborted_output_ignored"],
                **cancelled},
            "semantic_sha256": digest(native["semantic"]),
            "abort_semantic_sha256": digest(abort_native["semantic"]),
            "bridge_stats": direct["bridge_stats"],
            "metadata_future_bridge_stats": metadata["bridge_stats"],
            "limitation": "The single worker serializes ready encoding jobs. An unresolved device-metadata future must not occupy it; CUDA transfer/kernel performance is not tested."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/autodl-tmp/moe-research-20261002/model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run(args.model))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output: args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

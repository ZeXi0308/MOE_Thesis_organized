"""Native semantic checks for materialization gates; this is not a benchmark.

Each arm runs in a fresh process. The processor still receives one native engine
event at a time. We compare its state immediately after every event, and compare
the complete OpenAI SSE response after flattening chunk boundaries. The small
stop/abort/deadline cases cover behavior absent from the fixed-length replay.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
ARMS = ("native1", "native2", "native4", "token4", "bytes1024",
        "class_static", "lazy_static")
MARKER = "SEMANTIC_WORKER_JSON="


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


async def worker(arm, model):
    # Import after subprocess isolation, using the same real pipeline as replay.
    from native_adapter import NativeAdapter, get_tokenizer
    from vllm.v1.engine import FinishReason
    from gate import install

    tokenizer = get_tokenizer(model, trust_remote_code=False)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    unicode_text = "A🙂中𐍈 e\u0301 🧪 CPU output: café, naïve, 東京."
    stop_text = "Keep this prefix. <STOP> This suffix must not be delivered."
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "Engine STOP case requires the model's EOS token"
    sequences = {
        "unicode_light": encode(unicode_text),
        "unicode_heavy": encode(unicode_text),
        "stop_string": encode(stop_text),
        "engine_stop": encode("The engine finishes here.") + [int(eos_id)],
        "abort_pending": encode("First second third fourth fifth sixth."),
        "deadline": encode("One two three four five six seven eight nine."),
    }
    assert all(len(ids) >= 4 for ids in sequences.values())
    prompt = encode("Check complete native output semantics.")
    specs = [{"request_id": rid, "prompt_token_ids": prompt,
              "output_token_ids": ids, "heavy": rid != "unicode_light"}
             for rid, ids in sequences.items()]
    adapter = NativeAdapter(specs, model_path=model)
    # Keep these references only in this semantic test, never in measurements.
    states = dict(adapter.processor.request_states)
    detok = states["stop_string"].detokenizer
    detok.stop = ["<STOP>"]
    detok.stop_buffer_length = len("<STOP>") - 1
    detok.include_stop_str_in_output = False
    manager = install(adapter, arm, max_wait_ms=20)
    traces = {rid: [] for rid in sequences}
    payloads = {rid: [] for rid in sequences}
    done_counts = {rid: 0 for rid in sequences}

    async def consume(rid):
        async for chunk in adapter.stream(rid):
            assert chunk.startswith("data: "), (rid, chunk)
            raw = chunk[6:].strip()
            if raw == "[DONE]":
                done_counts[rid] += 1
            else:
                obj = json.loads(raw)
                assert "error" not in obj, (rid, obj)
                payloads[rid].append(obj)

    tasks = {rid: asyncio.create_task(consume(rid)) for rid in sequences}

    def snapshot(rid, action, aborts):
        state = states[rid]
        lp = state.logprobs_processor
        traces[rid].append({
            "action": action,
            "token_ids": list(state.detokenizer.output_token_ids),
            "output_text": state.detokenizer.output_text,
            "cumulative_logprob": lp.cumulative_logprob,
            "registered": rid in adapter.processor.request_states,
            "engine_aborts": list(aborts),
        })

    async def settle():
        # Permit native collector -> API consumption without a timing claim.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def feed(rid, position, reason=None):
        batch = adapter.build_batch({"time_s": position / 100,
            "outputs": [{"request_id": rid, "position": position,
                         "finished": position == len(sequences[rid]) - 1}]})
        if reason is not None:
            batch.outputs[0].finish_reason = reason
            if reason == FinishReason.STOP:
                batch.outputs[0].stop_reason = int(eos_id)
        result = adapter.feed(batch)
        snapshot(rid, f"token:{position}", result.reqs_to_abort)
        await settle()
        return result

    try:
        for rid in ("unicode_light", "unicode_heavy", "engine_stop"):
            for pos in range(len(sequences[rid])):
                reason = (FinishReason.STOP if rid == "engine_stop" and
                          pos == len(sequences[rid]) - 1 else None)
                await feed(rid, pos, reason)

        # String stop is detected by the real detokenizer, before the gate.
        stop_aborts = []
        for pos in range(len(sequences["stop_string"])):
            result = await feed("stop_string", pos)
            stop_aborts.extend(result.reqs_to_abort)
            if "stop_string" not in adapter.processor.request_states:
                break
        assert stop_aborts == ["stop_string"], stop_aborts
        assert pos < len(sequences["stop_string"]) - 1

        # First token emitted; the second is pending for batching arms.
        await feed("abort_pending", 0)
        await feed("abort_pending", 1)
        aborted = adapter.processor.abort_requests(["abort_pending"], internal=True)
        snapshot("abort_pending", "abort", aborted)
        assert aborted == ["abort_pending"]
        await settle()

        await feed("deadline", 0)
        await feed("deadline", 1)
        before_wait = sum(x["token_count"] for x in adapter.ledger["deadline"])
        await asyncio.sleep(.05)  # No engine event during this deadline window.
        after_wait = sum(x["token_count"] for x in adapter.ledger["deadline"])
        if arm in ("native2", "native4"):
            assert before_wait == after_wait == 1, (arm, before_wait, after_wait)
        else:
            assert after_wait == 2, (arm, before_wait, after_wait)
        snapshot("deadline", "no_engine_event_deadline", [])
        for pos in range(2, len(sequences["deadline"])):
            await feed("deadline", pos)

        await asyncio.wait_for(asyncio.gather(*tasks.values()), timeout=10)
        assert not adapter.processor.has_unfinished_requests()
        ledger_before = {rid: len(adapter.ledger[rid]) for rid in sequences}
        await asyncio.sleep(.04)
        assert ledger_before == {rid: len(adapter.ledger[rid]) for rid in sequences}
        assert all(q.output is None for q in adapter.collectors.values()), \
            "Ended request has a ghost timer output"

        semantic = {}
        chunks = {}
        for rid, stream in payloads.items():
            content, lps, finishes, usages, roles = [], [], [], [], []
            for obj in stream:
                if obj.get("usage") is not None:
                    usages.append(obj["usage"])
                for choice in obj.get("choices", []):
                    delta = choice["delta"]
                    if delta.get("role") is not None:
                        roles.append(delta["role"])
                    content.append(delta.get("content") or "")
                    lp = choice.get("logprobs")
                    if lp:
                        lps.extend(lp.get("content", []))
                    if choice.get("finish_reason") is not None:
                        finishes.append([choice["finish_reason"],
                                         choice.get("stop_reason")])
            count = sum(x["token_count"] for x in adapter.ledger[rid])
            expected_count = len(states[rid].detokenizer.output_token_ids)
            assert count == expected_count, (rid, count, expected_count)
            assert len(lps) == (0 if rid == "unicode_light" else count)
            assert all(len(lp["top_logprobs"]) == 20 for lp in lps), rid
            assert len(usages) == 1 and usages[0]["completion_tokens"] == count
            assert len(finishes) == 1 and done_counts[rid] == 1
            assert roles == ["assistant"], (rid, roles)
            semantic[rid] = {"text": "".join(content), "logprobs": lps,
                             "finish": finishes, "usage": usages,
                             "roles": roles, "done": done_counts[rid],
                             "tokens": count}
            chunks[rid] = len(adapter.ledger[rid])
        assert semantic["unicode_light"]["text"] == tokenizer.decode(
            sequences["unicode_light"], skip_special_tokens=True)
        assert semantic["unicode_heavy"]["text"] == semantic["unicode_light"]["text"]
        assert semantic["stop_string"]["text"] == stop_text.split("<STOP>")[0]
        assert semantic["stop_string"]["finish"] == [["stop", "<STOP>"]]
        assert semantic["engine_stop"]["finish"] == [["stop", int(eos_id)]]
        assert semantic["abort_pending"]["finish"] == [["abort", None]]
        assert semantic["abort_pending"]["tokens"] == 2
        return {"arm": arm, "semantic": semantic, "state_trace": traces,
                "chunks": chunks, "deadline_tokens_before": before_wait,
                "deadline_tokens_after": after_wait, "deadline_ms": 20,
                "no_post_finish_output": True}
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        close = getattr(manager, "close", None)
        if callable(close):
            value = close()
            if asyncio.iscoroutine(value):
                await value


def compare_states(reference, current):
    assert reference.keys() == current.keys()
    for rid, expected in reference.items():
        observed = current[rid]
        assert len(expected) == len(observed), (rid, len(expected), len(observed))
        for a, b in zip(expected, observed):
            for field in a:
                if field == "cumulative_logprob" and a[field] is not None:
                    assert b[field] is not None and math.isclose(
                        a[field], b[field], rel_tol=1e-12, abs_tol=1e-12), \
                        (rid, a["action"], field, a[field], b[field])
                else:
                    assert a[field] == b[field], \
                        (rid, a["action"], field, a[field], b[field])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/root/autodl-tmp/moe-research-20261002/model")
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        result = asyncio.run(worker(args.worker, args.model))
        print(MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return
    baseline = None
    rows = []
    for arm in ARMS:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", arm,
                   "--model", args.model]
        proc = subprocess.run(command, env=env, text=True, capture_output=True,
                              timeout=180)
        if proc.returncode:
            raise RuntimeError(f"Semantic check failed for {arm}\n{proc.stdout}\n{proc.stderr}")
        lines = [line[len(MARKER):] for line in proc.stdout.splitlines()
                 if line.startswith(MARKER)]
        assert len(lines) == 1, (arm, proc.stdout, proc.stderr)
        result = json.loads(lines[0])
        if baseline is None:
            baseline = result
        else:
            assert result["semantic"] == baseline["semantic"], \
                f"Complete flattened response mismatch: {arm}"
            compare_states(baseline["state_trace"], result["state_trace"])
        rows.append({"arm": arm, "complete_semantics_equal": True,
                     "every_engine_step_state_equal": True,
                     "semantic_sha256": digest(result["semantic"]),
                     "chunks_by_case": result["chunks"],
                     "deadline_tokens_before": result["deadline_tokens_before"],
                     "deadline_tokens_after": result["deadline_tokens_after"],
                     "no_post_finish_output": result["no_post_finish_output"]})
    summary = {"ok": True, "gpu_used": False, "performance_claim": False,
               "native_frontend": True, "deadline_test_ms": 20,
               "cases": list(baseline["semantic"]), "arms": rows,
               "state_fields": ["detokenizer token_ids", "detokenizer output_text",
                                "cumulative_logprob", "request registration",
                                "engine abort decisions"],
               "not_compared": ["chunk boundaries", "pending logprob container length"],
               "native2_native4_have_no_deadline": True}
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

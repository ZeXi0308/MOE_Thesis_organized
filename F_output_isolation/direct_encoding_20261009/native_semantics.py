"""Native frontend equivalence for direct encoding (correctness, no benchmark).

python native_semantics.py --output native_semantics.result.json
python native_semantics.py --capture capture --output capture_semantics.result.json

Workers are separate CPU-only processes. The mechanism cases intentionally use
constructed arrays; --capture instead reads the actual trace/candidates, retains
each natural batch and compares the installed OutputProcessor -> native request
collector -> OpenAI SSE generator. Neither mode claims model or delivery speed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
MARKER = "DIRECT_SEMANTICS_JSON="


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def flatten(payloads, done_counts, adapter, states):
    semantic = {}
    for rid, stream in payloads.items():
        content, logprobs, finishes, usages, roles = [], [], [], [], []
        for obj in stream:
            if obj.get("usage") is not None:
                usages.append(obj["usage"])
            for choice in obj.get("choices", []):
                delta = choice["delta"]
                if delta.get("role") is not None:
                    roles.append(delta["role"])
                content.append(delta.get("content") or "")
                if choice.get("logprobs"):
                    logprobs.extend(choice["logprobs"].get("content") or [])
                if choice.get("finish_reason") is not None:
                    finishes.append([choice["finish_reason"], choice.get("stop_reason")])
        count = sum(x["token_count"] for x in adapter.ledger[rid])
        assert count == len(states[rid].detokenizer.output_token_ids), (rid, count)
        heavy = adapter.requests[rid].get("heavy", False)
        assert len(logprobs) == (count if heavy else 0), (rid, len(logprobs), count)
        assert len(usages) == 1 and usages[0]["completion_tokens"] == count, (rid, usages)
        assert len(finishes) == done_counts[rid] == 1, (rid, finishes, done_counts[rid])
        assert roles == ["assistant"], (rid, roles)
        semantic[rid] = {"text": "".join(content), "logprobs": logprobs,
            "finish": finishes, "usage": usages, "roles": roles,
            "done": done_counts[rid], "tokens": count}
    return semantic


async def worker(arm, model, capture):
    # NativeAdapter applies the same narrow installed-runtime bootstrap as the
    # existing CPU replay; it also prevents accidental GPU initialization.
    from native_adapter import NativeAdapter, get_tokenizer
    from vllm.v1.engine import FinishReason
    from vllm.v1.outputs import LogprobsLists
    from cpu_encoder import build_token_table, CPUEncoder
    from frontend_bridge import DirectFrontend
    import numpy as np

    tokenizer = get_tokenizer(model, trust_remote_code=False)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    unicode_text = "A🙂中𐍈 e\u0301 🧪 CPU output: café, naïve, 東京."
    stop_text = "Keep this prefix. <STOP> This suffix must not be delivered."
    trace = arrays = None
    if capture:
        trace_file, array_file = capture / "trace.json", capture / "candidates.npz"
        trace = json.loads(trace_file.read_text())
        assert trace["requests"] and trace["batches"], "empty captured workload"
        assert all(r["finished"] for r in trace["requests"]), "capture is incomplete"
        specs = trace["requests"]
        arrays = np.load(array_file, allow_pickle=False)
        source = {"kind": "captured_native_engine_output",
            "trace_sha256": hashlib.sha256(trace_file.read_bytes()).hexdigest(),
            "candidates_sha256": hashlib.sha256(array_file.read_bytes()).hexdigest(),
            "source_note": trace.get("note"), "batches": len(trace["batches"])}
    else:
        eos = tokenizer.eos_token_id
        assert eos is not None
        sequences = {"unicode_light": encode(unicode_text),
            "unicode_heavy": encode(unicode_text),
            "context_fallback": encode(unicode_text),
            "rank_outside_top20": encode("Alpha beta gamma delta epsilon."),
            "duplicate_last_value": encode("First second third fourth fifth."),
            "stop_string": encode(stop_text),
            "engine_stop": encode("The engine finishes here.") + [int(eos)],
            "abort_direct": encode("First second third fourth fifth sixth.")}
        assert all(len(x) >= 4 for x in sequences.values())
        prompt = encode("Check full native response semantics.")
        specs = [{"request_id": rid, "prompt_token_ids": prompt,
            "output_token_ids": ids, "heavy": rid != "unicode_light"}
            for rid, ids in sequences.items()]
        source = {"kind": "constructed_mechanism_cases", "real_candidate_distribution": False}

    adapter = NativeAdapter(specs, model_path=model)
    states = dict(adapter.processor.request_states)
    table = build_token_table(adapter.tokenizer)
    encoder = CPUEncoder(table) if arm == "direct" else None
    bridge = DirectFrontend(adapter.processor, adapter.chat, adapter.tokenizer, encoder) if encoder else None
    payloads = {rid: [] for rid in states}
    done_counts = dict.fromkeys(states, 0)
    state_trace = []
    observed_rows = 0
    capture_array_rows = 0
    assertions = {"no_post_finish_output": False,
                  "direct_abort_immediate_cleanup": False,
                  "unsupported_semantics_not_tested": ["prompt logprobs", "n>1", "streaming input", "token-ID placeholders"]}

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

    tasks = {rid: asyncio.create_task(consume(rid)) for rid in states}

    def snapshot(action, affected, aborts):
        values = {}
        for rid in affected:
            state = states[rid]
            values[rid] = {"token_ids": list(state.detokenizer.output_token_ids),
                "output_text": state.detokenizer.output_text,
                "cumulative_logprob": state.logprobs_processor.cumulative_logprob,
                "registered": rid in adapter.processor.request_states}
        state_trace.append({"action": action, "requests": values, "engine_aborts": list(aborts)})

    async def settle():
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # Propagate frontend exceptions immediately rather than timing out later.
        for task in tasks.values():
            if task.done():
                task.result()

    async def feed(description, *, action):
        nonlocal observed_rows, capture_array_rows
        batch = adapter.build_batch(description)
        for output, item in zip(batch.outputs, description["outputs"]):
            if capture:
                key = item.get("logprobs_key")
                if key:
                    ids = np.ascontiguousarray(arrays[key + "_ids"])
                    lp = np.ascontiguousarray(arrays[key + "_values"])
                    ranks = np.ascontiguousarray(arrays[key + "_ranks"])
                    assert ids.shape == lp.shape and ids.shape[1] == 21
                    assert list(ids[:, 0]) == output.new_token_ids
                    output.new_logprobs = LogprobsLists(ids, lp, ranks, None)
                    capture_array_rows += len(ids)
                else:
                    assert not adapter.requests[item["request_id"]].get("heavy") or not output.new_token_ids
                    output.new_logprobs = None
                reason = item.get("finish_reason")
                output.finish_reason = {None: None, "length": FinishReason.LENGTH,
                    "stop": FinishReason.STOP, "abort": FinishReason.ABORT}[reason]
                output.stop_reason = item.get("stop_reason")
            if output.new_logprobs is not None:
                observed_rows += len(output.new_logprobs.sampled_token_ranks)
        result = adapter.feed(batch)
        snapshot(action, [x.request_id for x in batch.outputs], result.reqs_to_abort)
        await settle()
        return result

    try:
        if capture:
            for index, batch in enumerate(trace["batches"]):
                await feed(batch, action=f"natural_batch:{index}")
            assert capture_array_rows == sum(len(r["output_token_ids"]) for r in specs if r["heavy"])
        else:
            unsafe_ids = np.flatnonzero(table.unsafe)
            assert len(unsafe_ids), "this tokenizer has no unsafe token; context mechanism case is inapplicable"
            unsafe_id = int(unsafe_ids[0])
            safe_pool = [int(x) for x in np.flatnonzero(table.unsafe == 0)[:128]
                         if int(x) not in set(tokenizer.all_special_ids)]
            assert len(safe_pool) >= 22
            for rid in ("rank_outside_top20", "duplicate_last_value", "context_fallback"):
                raw = adapter._logprobs[rid]
                for row, sampled in enumerate(sequences[rid]):
                    alternatives = [x for x in safe_pool if x != sampled][:20]
                    ids = [sampled] + alternatives
                    values = -np.arange(1, 22, dtype=np.float32) / np.float32(7)
                    rank = 47
                    if rid == "duplicate_last_value":
                        ids[2] = sampled
                        ids[5] = ids[1]
                        values[0], values[2], values[1], values[5] = -.125, -2.75, -1.25, -6.5
                        rank = 2
                    if rid == "context_fallback":
                        ids[12] = unsafe_id
                    raw.logprob_token_ids[row] = ids
                    raw.logprobs[row] = values
                    raw.sampled_token_ranks[row] = rank
            detok = states["stop_string"].detokenizer
            detok.stop = ["<STOP>"]
            detok.stop_buffer_length = len("<STOP>") - 1
            detok.include_stop_str_in_output = False

            # Same ready batch includes plain and costly streams. One output has
            # two ready positions to exercise context inside a single encode.
            for pos in range(0, len(sequences["unicode_light"]), 2):
                end = min(pos + 2, len(sequences["unicode_light"]))
                await feed({"outputs": [{"request_id": rid, "token_start": pos,
                    "token_end": end} for rid in ("unicode_light", "unicode_heavy", "context_fallback")]},
                    action=f"unicode_ready_batch:{pos}:{end}")
            for rid in ("rank_outside_top20", "duplicate_last_value", "engine_stop", "stop_string"):
                for pos in range(len(sequences[rid])):
                    description = {"outputs": [{"request_id": rid, "position": pos}]}
                    if rid == "engine_stop" and pos == len(sequences[rid]) - 1:
                        batch = adapter.build_batch(description)
                        batch.outputs[0].finish_reason = FinishReason.STOP
                        batch.outputs[0].stop_reason = int(eos)
                        result = adapter.feed(batch)
                        snapshot(f"{rid}:{pos}", [rid], result.reqs_to_abort)
                        await settle()
                    else:
                        result = await feed(description, action=f"{rid}:{pos}")
                    if rid == "stop_string" and rid not in adapter.processor.request_states:
                        assert result.reqs_to_abort == [rid]
                        assert pos < len(sequences[rid]) - 1
                        break
            for pos in (0, 1):
                await feed({"outputs": [{"request_id": "abort_direct", "position": pos}]},
                           action=f"abort_direct:{pos}")
            aborted = adapter.processor.abort_requests(["abort_direct"], internal=True)
            assert aborted == ["abort_direct"]
            snapshot("direct_abort", ["abort_direct"], aborted)
            # Check before ANY later process() or close(): cancellation itself
            # must release bridge state, not rely on another engine event.
            if bridge:
                assert "abort_direct" not in bridge.original_updates, "abort retains patched LogprobsProcessor"
                assert "abort_direct" not in bridge.histories, "abort retains sampled-token history"
            assertions["direct_abort_immediate_cleanup"] = True
            await settle()
            before_late = len(adapter.ledger["abort_direct"])
            await feed({"outputs": [{"request_id": "abort_direct", "position": 2}]}, action="late_output_after_abort")
            assert len(adapter.ledger["abort_direct"]) == before_late

        await asyncio.wait_for(asyncio.gather(*tasks.values()), timeout=15)
        assert not adapter.processor.has_unfinished_requests()
        assert not adapter.processor.request_states
        if bridge:
            assert not bridge.histories and not bridge.original_updates
            # Async representation retains immutable future references in the
            # historical position list. They must be resolved after complete
            # SSE delivery; pending data during abort itself is legitimate.
            for state in states.values():
                for item in state.logprobs_processor.logprobs or []:
                    future = getattr(item, "future", None)
                    if future is not None:
                        assert future.done() and not future.cancelled()
                        assert future.exception() is None
            if not capture:
                assert bridge.stats["fallback_rows"] >= len(sequences["context_fallback"])
        before = {rid: len(adapter.ledger[rid]) for rid in states}
        await settle()
        assert before == {rid: len(adapter.ledger[rid]) for rid in states}
        assert all(q.output is None for q in adapter.collectors.values())
        assertions["no_post_finish_output"] = True
        semantic = flatten(payloads, done_counts, adapter, states)
        if not capture:
            assert semantic["unicode_light"]["text"] == tokenizer.decode(sequences["unicode_light"], skip_special_tokens=True)
            for rid in ("unicode_heavy", "context_fallback"):
                assert semantic[rid]["text"] == semantic["unicode_light"]["text"]
            assert semantic["stop_string"]["text"] == stop_text.split("<STOP>")[0]
            assert semantic["stop_string"]["finish"] == [["stop", "<STOP>"]]
            assert semantic["engine_stop"]["finish"] == [["stop", int(eos)]]
            assert semantic["abort_direct"]["finish"] == [["abort", None]]
            assert semantic["abort_direct"]["tokens"] == 2
            rank_rows = semantic["rank_outside_top20"]["logprobs"]
            assert all(len(row["top_logprobs"]) == 20 for row in rank_rows)
            duplicate_rows = semantic["duplicate_last_value"]["logprobs"]
            assert all(row["logprob"] == -2.75 for row in duplicate_rows)
            assert all(len(row["top_logprobs"]) == 19 for row in duplicate_rows)
            assert all(row["top_logprobs"][1]["logprob"] == -6.5 for row in duplicate_rows)
        return {"arm": arm, "source": source, "semantic": semantic,
            "state_trace": state_trace, "assertions": assertions,
            "bridge_stats": dict(bridge.stats) if bridge else None,
            "observed_rows": observed_rows, "chunks": before,
            "table": {**table.metadata, "build_seconds": table.build_seconds}}
    finally:
        for task in tasks.values():
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        if bridge: bridge.close()
        if encoder: encoder.close()
        if arrays is not None: arrays.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/autodl-tmp/moe-research-20261002/model")
    parser.add_argument("--capture", type=Path, help="directory with trace.json and candidates.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("native", "direct"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = asyncio.run(worker(args.worker, args.model, args.capture))
        print(MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        return
    baseline, summary_rows = None, []
    for arm in ("native", "direct"):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", arm, "--model", args.model]
        if args.capture: command.extend(["--capture", str(args.capture.resolve())])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
        proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=360)
        if proc.returncode:
            raise RuntimeError(f"{arm} native semantic check failed\n{proc.stdout}\n{proc.stderr}")
        lines = [line[len(MARKER):] for line in proc.stdout.splitlines() if line.startswith(MARKER)]
        assert len(lines) == 1, (arm, proc.stdout, proc.stderr)
        current = json.loads(lines[0])
        if baseline is None:
            baseline = current
        else:
            assert current["source"] == baseline["source"]
            for rid in baseline["semantic"]:
                assert current["semantic"][rid] == baseline["semantic"][rid], f"flattened native SSE mismatch for {rid}"
            assert len(current["state_trace"]) == len(baseline["state_trace"])
            for expected, actual in zip(baseline["state_trace"], current["state_trace"]):
                assert expected == actual, ("native per-engine-event state mismatch", expected, actual)
        summary_rows.append({"arm": arm, "complete_semantics_equal": True,
            "every_engine_step_state_equal": True,
            "semantic_sha256": digest(current["semantic"]),
            "state_sha256": digest(current["state_trace"]),
            "request_semantic_sha256": {rid: digest(value) for rid, value in current["semantic"].items()},
            "chunks": current["chunks"], "assertions": current["assertions"],
            "bridge_stats": current["bridge_stats"], "table": current["table"]})
    result = {"ok": True, "gpu_used": False, "performance_claim": False,
        "native_frontend": True, "source": baseline["source"],
        "cases": list(baseline["semantic"]), "arms": summary_rows,
        "state_fields": ["detokenizer token_ids", "detokenizer output_text",
            "cumulative_logprob (exact)", "request registration", "engine abort decisions"],
        "response_fields": ["text", "ordered logprobs/content/top_logprobs",
            "JSON numerical values", "UTF-8 bytes", "finish/stop", "usage", "role", "DONE"],
        "not_compared": ["logprob object representation", "wall-clock delivery latency", "model speed"]}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output: args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

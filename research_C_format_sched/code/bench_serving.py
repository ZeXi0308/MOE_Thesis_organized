#!/usr/bin/env python3
"""One existing vLLM 0.26 server, frozen open-loop trace, board power accounting.

Must be a descendant of run_locked.py. That wrapper requires an initially empty
GPU, so external orchestration must start the server INSIDE the same locked job,
wait for readiness, then invoke this client. This file never starts a server,
imports torch, owns the GPU lock, installs packages, or downloads datasets.

Example client command inside that already-locked orchestration:
  python code/bench_serving.py --trace requests.jsonl --request-rate 2 --seed 41 \
    --endpoint http://127.0.0.1:8000/v1/completions --model served-model \
    --arm fp8 --config-id qwen-fp8 --repeat-id 1 --server-metadata server.json \
    --lock-status runs/serve_r01/status.json --output runs/serve_r01/serving

Trace JSONL: {"request_id":"r0","prompt":"...","max_tokens":128,
              "sampling":{"temperature":0,"seed":41}}
Either supply --request-rate (deterministic Poisson schedule from --seed), OR
give every trace row an explicit nondecreasing arrival_s and omit the rate.
The full normalized trace and arrivals are saved before requests are sent.
Prompt text is already rendered for /v1/completions; no chat template/tokenizer
is silently added. Natural EOS remains enabled unless explicitly in sampling.
Supported sampling keys are deliberately narrow; n=1 and echo=false are fixed.
Unspecified sampling values follow server defaults. --server-metadata is a
stable configuration manifest (checkpoint revision, launch options, generation
defaults), not a run-specific PID/timestamp file; its hash identifies comparable
repetitions. Wrapper timeout must cover warmup + idle + arrivals + drain.

Verified on installed vLLM 0.26.0, benchmarks/lib/endpoint_request_func.py,
entrypoints/openai/engine/protocol.py, and official v0.26 documentation:
https://docs.vllm.ai/en/v0.26.0/cli/bench/serve/
Uses its completions payload and first/last streamed-choice timing convention.
Adds continuous usage, fixed absolute arrivals, all-request terminal records,
and bounded drain; does not invoke a guessed vllm bench command. TTFT includes
HTTP/network queueing after send; client scheduling delay is a separate field.
--self-test is CPU-only and bypasses the execution/lock/NVML path.
"""
from __future__ import annotations

import argparse
import asyncio
import codecs
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from urllib.parse import urlparse

from analyze_serving import summarize_run

LOCK = Path("/root/autodl-tmp/moe-research-gpu.lock")
SAMPLING_KEYS = {"temperature", "top_p", "top_k", "seed", "stop", "stop_token_ids",
                 "ignore_eos", "min_tokens", "repetition_penalty", "presence_penalty", "frequency_penalty"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_lock(status_path):
    """Require the real live wrapper ancestor and its open lock FD; do not lock."""
    import fcntl
    status = json.loads(Path(status_path).read_text())
    controller = status.get("controller_pid")
    if status.get("state") != "RUNNING" or not isinstance(controller, int):
        raise RuntimeError("lock status is not a RUNNING run_locked controller")
    ancestors, pid = set(), os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        pid = int(stat[1])
    if controller not in ancestors:
        raise RuntimeError("run_locked controller must be a live ancestor of this client")
    cmdline = Path(f"/proc/{controller}/cmdline").read_bytes().split(b"\0")
    if not any(Path(os.fsdecode(x)).name == "run_locked.py" for x in cmdline if x):
        raise RuntimeError("controller is not run_locked.py")
    lock_stat = LOCK.stat()
    owns_fd = False
    for fd in Path(f"/proc/{controller}/fd").iterdir():
        try:
            st = fd.stat()
            owns_fd |= (st.st_dev, st.st_ino) == (lock_stat.st_dev, lock_stat.st_ino)
        except FileNotFoundError:
            pass
    if not owns_fd:
        raise RuntimeError("controller has no open shared GPU lock descriptor")
    with LOCK.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"controller_pid": controller, "status_path": str(Path(status_path).resolve()),
                    "gpu_uuid": status.get("gpu_uuid"), "lock": str(LOCK)}
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise RuntimeError("shared GPU lock is not held")


def load_trace(path, rate, seed):
    raw = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not raw:
        raise ValueError("empty trace")
    explicit = ["arrival_s" in r for r in raw]
    if any(explicit) != all(explicit) or (rate is not None) == all(explicit):
        raise ValueError("use either --request-rate or arrival_s on every row")
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError("request rate must be positive and finite")
    rng, arrival, ids, rows = random.Random(seed), 0.0, set(), []
    for i, item in enumerate(raw):
        rid = str(item.get("request_id", i))
        if rid in ids:
            raise ValueError("duplicate request_id")
        ids.add(rid)
        if not isinstance(item.get("prompt"), str) or not item["prompt"]:
            raise ValueError("each prompt must be a nonempty rendered string")
        cap = item.get("max_tokens")
        if type(cap) is not int or cap <= 0:
            raise ValueError("max_tokens must be a positive integer")
        sampling = item.get("sampling", {})
        if not isinstance(sampling, dict) or set(sampling) - SAMPLING_KEYS:
            raise ValueError("unsupported sampling fields")
        if rate is not None:
            arrival += rng.expovariate(rate)
        else:
            previous, arrival = arrival, float(item["arrival_s"])
            if not math.isfinite(arrival) or arrival < 0 or arrival < previous:
                raise ValueError("arrival_s must be finite, nonnegative and nondecreasing")
        rows.append({"request_id": rid, "prompt": item["prompt"], "max_tokens": cap,
                     "sampling": sampling, "arrival_s": arrival})
    return rows


class SSEDecoder:
    """Incremental UTF-8 and SSE framing; arbitrary HTTP chunk boundaries."""
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""

    def feed(self, chunk, final=False):
        self.buffer += self.decoder.decode(chunk, final=final)
        self.buffer = self.buffer.replace("\r\n", "\n")
        events = []
        while "\n\n" in self.buffer:
            event, self.buffer = self.buffer.split("\n\n", 1)
            lines = [line[5:].lstrip(" ") for line in event.splitlines() if line.startswith("data:")]
            if lines:
                events.append("\n".join(lines))
        if final and self.buffer.strip():
            raise ValueError("unterminated SSE event at EOF")
        return events


def consume_event(row, event, now):
    if event == "[DONE]":
        row["stream_done"] = True
        return
    data = json.loads(event)
    if "error" in data:
        raise ValueError("server stream error: " + json.dumps(data["error"]))
    if usage := data.get("usage"):
        for key, target in (("completion_tokens", "output_tokens"), ("prompt_tokens", "prompt_tokens")):
            if usage.get(key) is not None:
                value = usage[key]
                if type(value) is not int or value < 0:
                    raise ValueError("invalid usage token count")
                if target == "output_tokens" and row.get(target) is not None and value < row[target]:
                    raise ValueError("cumulative output-token usage decreased")
                row[target] = value
        if not data.get("choices"):
            row["final_usage_received"] = True
    if choices := data.get("choices"):
        if len(choices) != 1 or choices[0].get("index", 0) != 0:
            raise ValueError("expected exactly one completion choice")
        if row.get("first_choice_ns") is None:
            row["first_choice_ns"] = now
        row["last_choice_ns"] = now
        row["choice_chunk_times_ns"].append(now)
        row["generated_text"] += choices[0].get("text") or ""
        if choices[0].get("finish_reason") is not None:
            row["finish_reason"] = choices[0]["finish_reason"]


async def send_request(session, endpoint, model, trace, row, timeout_s, headers):
    row["sent_ns"] = time.monotonic_ns()
    row["client_send_delay_ms"] = max(0, (row["sent_ns"] - row["scheduled_arrival_ns"]) / 1e6)
    payload = {"model": model, "prompt": trace["prompt"], "max_tokens": trace["max_tokens"],
               "n": 1, "echo": False, "stream": True, "repetition_penalty": 1.0,
               "stream_options": {"include_usage": True, "continuous_usage_stats": True}, **trace["sampling"]}
    async def receive():
        async with session.post(endpoint, json=payload, headers=headers) as response:
            row["http_status"] = response.status
            if response.status != 200:
                row.update(status="error", error="HTTP " + str(response.status) + ": " + (await response.text())[:4096])
                return
            decoder = SSEDecoder()
            async for chunk in response.content.iter_any():
                for event in decoder.feed(chunk):
                    consume_event(row, event, time.monotonic_ns())
            for event in decoder.feed(b"", final=True):
                consume_event(row, event, time.monotonic_ns())
            if not (row["stream_done"] and row["final_usage_received"] and row.get("output_tokens") is not None
                    and row.get("first_choice_ns") is not None and row.get("finish_reason") is not None):
                raise ValueError("incomplete stream, final usage, first choice, or finish reason")
            row["status"] = "completed"
    try:
        if row["client_send_delay_ms"] >= timeout_s * 1000:
            raise asyncio.TimeoutError()
        await asyncio.wait_for(receive(), timeout=max(.000001, timeout_s - row["client_send_delay_ms"] / 1000))
    except asyncio.TimeoutError:
        row.update(status="timeout", error="request deadline from scheduled arrival exceeded")
    except asyncio.CancelledError:
        row.update(status="unfinished", error="run drain deadline or interruption")
        raise
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        row["terminal_ns"] = time.monotonic_ns()
        if row.get("first_choice_ns") is not None:
            row["ttft_ms"] = (row["first_choice_ns"] - row["sent_ns"]) / 1e6
            row["arrival_to_first_choice_ms"] = (row["first_choice_ns"] - row["scheduled_arrival_ns"]) / 1e6
        if row.get("last_choice_ns") is not None and (row.get("output_tokens") or 0) > 1:
            row["tpot_ms"] = (row["last_choice_ns"] - row["first_choice_ns"]) / 1e6 / (row["output_tokens"] - 1)


def initial_rows(trace):
    return [{"request_id": r["request_id"], "arrival_s": r["arrival_s"], "max_tokens": r["max_tokens"],
             "status": "unfinished", "arrived": False, "sent_ns": None, "terminal_ns": None,
             "output_tokens": None, "prompt_tokens": None, "first_choice_ns": None,
             "last_choice_ns": None, "ttft_ms": None, "tpot_ms": None,
             "stream_done": False, "final_usage_received": False, "finish_reason": None,
             "generated_text": "", "choice_chunk_times_ns": []} for r in trace]


async def run_schedule(trace, rows, send, arrival_window_s, drain_s, event_log, stop_event):
    start = time.monotonic_ns()
    for row in rows:
        row["scheduled_arrival_ns"] = start + round(row["arrival_s"] * 1e9)
    tasks = []
    async def dispatch(i):
        row = rows[i]
        try:
            await send(trace[i], row)
        finally:
            event_log({"event": "terminal", "request": row})
    async def drive():
        for i, row in enumerate(rows):
            await asyncio.sleep(max(0, (row["scheduled_arrival_ns"] - time.monotonic_ns()) / 1e9))
            row["arrived"] = True
            event_log({"event": "arrival", "request_id": row["request_id"],
                       "scheduled_arrival_ns": row["scheduled_arrival_ns"], "dispatch_ns": time.monotonic_ns()})
            tasks.append(asyncio.create_task(dispatch(i)))
        await asyncio.sleep(max(0, start / 1e9 + arrival_window_s - time.monotonic()))
        if tasks:
            await asyncio.gather(*tasks)
    driver, stopper = asyncio.create_task(drive()), asyncio.create_task(stop_event.wait())
    reason = None
    try:
        done, _ = await asyncio.wait({driver, stopper}, timeout=arrival_window_s + drain_s,
                                     return_when=asyncio.FIRST_COMPLETED)
        if stopper in done:
            reason = "interrupted"
        elif driver not in done:
            reason = "drain_deadline"
        else:
            await driver
    finally:
        driver.cancel()
        stopper.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(driver, stopper, *tasks, return_exceptions=True)
        end = time.monotonic_ns()
        for row in rows:
            row["arrived"] = row["scheduled_arrival_ns"] <= end
            if row["terminal_ns"] is None:
                row.update(status="unfinished", terminal_ns=end, error=reason or "not dispatched")
    return {"measurement_start_ns": start, "measurement_end_ns": end,
            "measurement_elapsed_s": (end - start) / 1e9, "aborted": reason is not None,
            "stop_reason": reason or "all_requests_terminal"}


async def execute(args, trace, rows, meta, output):
    import aiohttp
    from power import PowerSampler
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    headers = {"Authorization": "Bearer " + os.environ[args.api_key_env]} if args.api_key_env else {}
    meter = PowerSampler(device=args.nvml_device, interval_s=.05)
    power = None
    try:
        if meta["lock"].get("gpu_uuid") != meter.device.get("uuid"):
            raise RuntimeError("NVML device must match the GPU owned by run_locked")
        # limit=0 avoids an implicit concurrency limiter changing the offered trace.
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                timeout=aiohttp.ClientTimeout(total=None)) as session:
            if args.warmup_trace:
                warm = load_trace(args.warmup_trace, 1, args.seed)
                warm_rows = initial_rows(warm)
                for t, row in zip(warm, warm_rows):
                    if stop.is_set():
                        raise RuntimeError("interrupted during warmup")
                    row.update(arrived=True, scheduled_arrival_ns=time.monotonic_ns())
                    await send_request(session, args.endpoint, args.model, t, row, args.request_timeout_s, headers)
                    if row["status"] != "completed":
                        raise RuntimeError("warmup request failed: " + str(row.get("error")))
                write_json(output / "warmup.json", warm_rows)
            # Server resident and quiescent; caller guarantees exclusive ownership.
            idle = meter.measure_idle(args.idle_seconds)
            write_json(output / "idle_power.json", idle)
            if stop.is_set():
                raise RuntimeError("interrupted before trace arrival window")
            with (output / "events.jsonl").open("w") as events:
                def log(event):
                    events.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    events.flush()
                async def send(t, row):
                    await send_request(session, args.endpoint, args.model, t, row, args.request_timeout_s,
                                       dict(headers, **{"X-Request-Id": row["request_id"]}))
                meter.start()
                try:
                    meta.update(await run_schedule(trace, rows, send, args.arrival_window_s, args.drain_s, log, stop))
                finally:
                    power = meter.stop()
                    write_json(output / "power.json", power)
    finally:
        meter.close()
    return power


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.jsonl"
        path.write_text('\n'.join(json.dumps({"request_id": str(i), "prompt": "x", "max_tokens": 3}) for i in range(3)))
        a, b = load_trace(path, 3, 7), load_trace(path, 3, 7)
        assert a == b and all(x["arrival_s"] < y["arrival_s"] for x, y in zip(a, a[1:]))
    decoder = SSEDecoder()
    packet = ('data: ' + json.dumps({"choices": [{"text": "中", "finish_reason": None}]}, ensure_ascii=False) + '\r\n\r\n').encode()
    events = []
    for byte in packet:
        events += decoder.feed(bytes([byte]))
    assert len(events) == 1 and "中" in events[0]
    row = initial_rows([{"request_id": "a", "arrival_s": 0, "max_tokens": 3}])[0]
    consume_event(row, events[0], 100)
    consume_event(row, '{"choices":[{"text":"x","finish_reason":"stop"}],"usage":{"completion_tokens":3}}', 300)
    consume_event(row, '{"choices":[],"usage":{"completion_tokens":3,"prompt_tokens":1}}', 400)
    consume_event(row, '[DONE]', 500)
    assert row["output_tokens"] == 3 and row["first_choice_ns"] == 100 and row["last_choice_ns"] == 300
    assert row["final_usage_received"] and row["stream_done"]
    async def schedule_test():
        trace = [{"request_id": str(i), "arrival_s": i * .005, "max_tokens": 3} for i in range(3)]
        rows = initial_rows(trace)
        active = peak = 0
        async def send(t, row):
            nonlocal active, peak
            row["sent_ns"] = time.monotonic_ns()
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.1)
                row["status"] = "completed"
            except asyncio.CancelledError:
                row["status"] = "unfinished"
                raise
            finally:
                active -= 1
                row["terminal_ns"] = time.monotonic_ns()
        m = await run_schedule(trace, rows, send, .011, .02, lambda x: None, asyncio.Event())
        assert peak == 3 and m["stop_reason"] == "drain_deadline"
        assert len(rows) == 3 and all(r["arrived"] and r["status"] == "unfinished" for r in rows)
    async def request_test():
        # Exercise the production request coroutine without aiohttp or a network.
        class Response:
            def __init__(self, packets, status=200):
                self.status, self.packets, self.content = status, packets, self
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def text(self):
                return "busy"
            async def iter_any(self):
                for delay, packet in self.packets:
                    await asyncio.sleep(delay)
                    yield packet
        class Session:
            def __init__(self, response):
                self.response = response
            def post(self, *args, **kwargs):
                assert kwargs["json"]["stream_options"]["continuous_usage_stats"]
                return self.response
        first = b'data: {"choices":[{"text":"a","finish_reason":null}],"usage":{"completion_tokens":1}}\n\n'
        last = b'data: {"choices":[{"text":"b","finish_reason":"length"}]}\n\n'
        usage = b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n'
        t = {"request_id": "test", "arrival_s": 0, "max_tokens": 2, "prompt": "x", "sampling": {}}
        async def run(response, timeout=.1):
            row = initial_rows([t])[0]
            row["scheduled_arrival_ns"] = time.monotonic_ns()
            await send_request(Session(response), "/v1/completions", "model", t, row, timeout, {})
            return row
        good = await run(Response([(0, first), (.001, last + usage)]))
        assert good["status"] == "completed" and good["output_tokens"] == 2 and good["tpot_ms"] > 0
        failed = await run(Response([], 429))
        assert failed["status"] == "error" and failed["http_status"] == 429
        timed = await run(Response([(0, first), (.1, last + usage)]), .005)
        assert timed["status"] == "timeout" and timed["output_tokens"] == 1
        missing = await run(Response([(0, first + last + b'data: [DONE]\n\n')]))
        assert missing["status"] == "error" and not missing["final_usage_received"]
    asyncio.run(schedule_test())
    asyncio.run(request_test())
    print("CPU trace/SSE/open-loop/drain/HTTP-error/partial-timeout/missing-usage checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    for name in ("trace", "endpoint", "model", "arm", "config-id", "repeat-id", "server-metadata", "lock-status", "output"):
        p.add_argument("--" + name)
    p.add_argument("--request-rate", type=float)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--arrival-window-s", type=float)
    p.add_argument("--request-timeout-s", type=float, default=180)
    p.add_argument("--drain-s", type=float, default=180)
    p.add_argument("--idle-seconds", type=float, default=3)
    p.add_argument("--nvml-device", default="0", help="physical NVML index or GPU UUID; client must run on server host")
    p.add_argument("--warmup-trace", help="optional separate JSONL without arrival_s; sequential warmup before idle baseline")
    p.add_argument("--api-key-env", help="name of environment variable, never stored in outputs")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    for name in ("trace", "endpoint", "model", "arm", "config_id", "repeat_id", "server_metadata", "lock_status", "output"):
        if not getattr(a, name):
            p.error("missing --" + name.replace("_", "-"))
    endpoint = urlparse(a.endpoint)
    if endpoint.scheme not in ("http", "https") or endpoint.path.rstrip("/") != "/v1/completions" or endpoint.username:
        p.error("endpoint must be an OpenAI /v1/completions URL without embedded credentials")
    if endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
        p.error("client and NVML must run on the server host; use a loopback endpoint")
    for name in ("request_timeout_s", "drain_s", "idle_seconds"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(name + " must be positive and finite")
    lock = verify_lock(a.lock_status)
    if importlib.metadata.version("vllm") != "0.26.0":
        raise RuntimeError("this request contract was checked against vLLM 0.26.0")
    trace = load_trace(a.trace, a.request_rate, a.seed)
    if a.arrival_window_s is None:
        a.arrival_window_s = max(trace[-1]["arrival_s"], 1 / a.request_rate if a.request_rate else .001)
    if not math.isfinite(a.arrival_window_s) or a.arrival_window_s < trace[-1]["arrival_s"]:
        p.error("arrival window must cover every trace arrival")
    a.nvml_device = int(a.nvml_device) if a.nvml_device.isdigit() else a.nvml_device
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    server_metadata = json.loads(Path(a.server_metadata).read_text())
    rows = initial_rows(trace)
    meta = {"schema_version": 1, "config_id": a.config_id, "arm": a.arm, "model": a.model,
        "repeat_id": a.repeat_id, "endpoint": a.endpoint, "server_metadata": server_metadata,
        "server_metadata_sha256": digest(server_metadata), "trace_content_sha256": digest([
            {k: v for k, v in t.items() if k != "arrival_s"} for t in trace]),
        "schedule_sha256": digest(trace), "request_rate": a.request_rate, "schedule_seed": a.seed,
        "arrival_window_s": a.arrival_window_s, "request_timeout_s": a.request_timeout_s,
        "drain_s": a.drain_s, "warmup_trace_sha256": hashlib.sha256(Path(a.warmup_trace).read_bytes()).hexdigest() if a.warmup_trace else None,
        "idle_seconds": a.idle_seconds, "lock": lock, "created_unix_ns": time.time_ns(),
        "vllm_version": "0.26.0", "request_protocol": "OpenAI completions streaming, n=1, final and continuous usage",
        "measurement_start": "after idle baseline; includes resume-to-load transition and all open-loop idle gaps",
        "measurement_end": "all client requests terminal or fixed arrival-window + drain deadline; not GPU synchronization",
        "aborted": True, "state": "PREPARING"}
    root = Path(importlib.util.find_spec("vllm").origin).parent
    meta["interface_source_sha256"] = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in (
        "benchmarks/lib/endpoint_request_func.py", "entrypoints/openai/engine/protocol.py")}
    (output / "trace.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False, allow_nan=False) + "\n" for t in trace))
    write_json(output / "metadata.json", meta)
    # Initial population survives SIGKILL; events supply subsequent completions.
    (output / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
    power = None
    try:
        meta["state"] = "RUNNING"
        write_json(output / "metadata.json", meta)
        power = asyncio.run(execute(a, trace, rows, meta, output))
        meta["state"] = "COMPLETE" if not meta["aborted"] and all(r["status"] == "completed" for r in rows) else "FAILED"
    except BaseException as error:
        meta.update(state="FAILED", aborted=True, error=f"{type(error).__name__}: {error}")
    finally:
        meta["ended_unix_ns"] = time.time_ns()
        (output / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
        if power is None and (output / "power.json").exists():
            power = json.loads((output / "power.json").read_text())
        write_json(output / "metadata.json", meta)
        write_json(output / "summary.json", summarize_run(rows, meta, power))
    print(json.dumps({"output": str(output), "state": meta["state"], "planned": len(rows)}))
    raise SystemExit(0 if meta["state"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()

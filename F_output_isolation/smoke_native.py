"""Small semantic smoke, not a latency benchmark."""
import asyncio
import json

from native_adapter import NativeAdapter, DEFAULT_MODEL, get_tokenizer


async def main():
    tokenizer = get_tokenizer(DEFAULT_MODEL, trust_remote_code=False)
    output = tokenizer.encode("CPU output isolation preserves every response token.",
                              add_special_tokens=False)
    prompt = tokenizer.encode("Explain output isolation.", add_special_tokens=False)
    requests = [{"request_id": rid, "prompt_token_ids": prompt,
                 "output_token_ids": output, "heavy": heavy}
                for rid, heavy in [("light", False), ("heavy", True)]]
    adapter = NativeAdapter(requests)
    received = {}

    async def consume(rid):
        chunks = [chunk async for chunk in adapter.stream(rid)]
        assert chunks[-1] == "data: [DONE]\n\n"
        payloads = [json.loads(c[6:]) for c in chunks[:-1]]
        assert all("error" not in p for p in payloads), payloads
        content = "".join(c["delta"].get("content", "")
                          for p in payloads for c in p["choices"])
        assert content == tokenizer.decode(output), (rid, content)
        token_count = sum(r["token_count"] for r in adapter.ledger[rid])
        assert token_count == len(output)
        lps = [lp for p in payloads for c in p["choices"]
               for lp in (c.get("logprobs") or {}).get("content", [])]
        assert len(lps) == (len(output) if rid == "heavy" else 0)
        assert all(len(lp["top_logprobs"]) == 20 for lp in lps)
        assert payloads[-1]["usage"]["completion_tokens"] == len(output)
        received[rid] = {"chunks": len(chunks), "tokens": token_count,
                         "bytes": sum(len(c.encode()) for c in chunks),
                         "top_logprobs_per_token": 20 if lps else 0}

    tasks = [asyncio.create_task(consume(rid)) for rid in adapter.requests]
    # Deliberately include one collector coalescing case. These sleeps belong
    # only to this smoke producer and are absent from the adapter.
    for position, token in enumerate(output):
        adapter.feed({"time_s": position / 50,
                      "outputs": [{"request_id": rid, "position": position,
                                   "token_id": token,
                                   "finished": position == len(output) - 1}
                                  for rid in adapter.requests]})
        if position % 2:
            await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    assert not adapter.processor.has_unfinished_requests()
    print(json.dumps({"ok": True, "native_output_path": True,
                      "gpu_used": False, "requests": received}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

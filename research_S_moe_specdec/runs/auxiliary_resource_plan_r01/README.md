# Auxiliary workload KV resource plan

**Candidate fixed KV caps: OLMoE48 GiB, Qwen3 AWQ24 GiB, Granite3B24 GiB**, shared by n-gram and neural-source runs within each family. Use the new common128 workload domains with context4096, first64 development / next64 held out, and128 output tokens plus G7. These are CPU capacity estimates; **actual KV groups/allocation, peak memory and B64 admission/preemption behavior require the first GPU smoke**. No configuration or queue was changed.

For BF16, TP=1, full attention, per-model KV payload is `2 × layers × KV heads × head_dim × 2 bytes/token`. We sum each frozen64-request population after independently rounding `complete prompt + 128 output + 7 lookahead` to cache blocks. Development and held-out populations are separate, never128 simultaneous requests. No prefix-sharing or early-completion savings are credited.

| Family / neural source | Target + draft KV KiB/token | Largest selected B64 demand, block16 / block64 | Fixed KV candidate | Margin over block64 static payload |
|---|---:|---:|---:|---:|
| OLMoE / OLMo-1B | 128 + 128 = 256 | 37.563 / 37.906 GiB | 48 GiB | 10.094 GiB |
| Qwen3 AWQ / EAGLE3 | 96 + 2 = 98 | 11.917 / 12.071 GiB | 24 GiB | 11.929 GiB |
| Granite3B / Granite1B | 64 + 48 = 112 | 15.384 / 15.538 GiB | 24 GiB | 8.462 GiB |

Thus24 GiB would be insufficient for the OLMoE pair's frozen B64 payload under this model. Its48 GiB candidate has substantial slack for block/null-page rounding beyond the calculated37.9 GiB. The other families'24 GiB candidates also have several GiB of KV slack. These caps cover the **actual selected populations**, not64 arbitrary4096-token requests or64 requests all at the filtering cap. LCC cap3072 and MultiNews cap3584 were used without truncation; exact workload SHA,12 task/family/split rows, block16/32/64 counts and dimensions are in `resource_plan.json`.

The local OLMoE config `/Users/zhaozhenyu/Desktop/毕业设计/MoE_Scheduling_20261002/model/config.json` has SHA256 `42ea6f5fe31e9243290ba3c348a4ad615b88a6fbc6e10544e21a87b6dff5da6c`, exactly matching this run's remote CPU feasibility scan. It records16 layers,16 query/16 KV heads, hidden2048, BF16; head_dim is2048/16=128. Other dimensions come from pinned local source/download metadata, with the scan's actual config fingerprints retained. Granite head_dim64 is derived from1536/24 or1024/16. No fresh remote read was possible after SSH failed; missing runtime realization is not silently treated as confirmed.

Previously captured installed-vLLM source indicates one explicit worker KV budget across registered target/draft attention layers. Uniform full-attention layers of different sizes can retain one common block count with per-layer page sizes; the budget is **not24 GiB for target plus another24 GiB for draft**. Neural drafter lookahead is G7. Source hashes and exact line ranges are in JSON. This supports the sum-of-layer planning model, but the new engines' realized grouping/backend/padding still must be observed. Explicit `kv_cache_memory_bytes` bypasses the automatic memory-budget calculation and does not establish sufficient available VRAM.

Against the stated97 GiB device, candidate cap plus weight-file/loading proxy leaves a coarse planning balance of33.7 GiB (OLMoE),57.2 GiB (Qwen),64.4 GiB (Granite) for graphs, prefill activations, workspaces, allocator/runtime state and other usage. Those balances are **not measured free memory or peak guarantees**: OLMoE uses a prior12.89 GiB target-loading report plus half the FP32 OLMo file bytes; other pairs use tensor-file bytes. Packed/converted weights and shared EAGLE components may differ from file sizes. Fixed KV across sources is not fixed total resident VRAM; torch allocated bytes likewise do not equal NVML residency, peak or total weights. GPU smoke must use the actual4096 context, prefill limit and graph settings.

No tokenizer/model load, GPU experiment, remote upload or new queue entry was performed. The failed connection provides no evidence about the state of remote experiments. Old auxiliary populations and experiments are not relabelled as this common128 population.

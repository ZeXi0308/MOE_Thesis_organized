# Complete auxiliary-input feasibility, one CPU scan

The existing pinned LongBench archive supports 128 complete, common source records for each task across all three target tokenizers. Among the requested prompt caps 2048/3072/3584, the smallest sufficient common cap is **3072 for LCC** and **3584 for MultiNews**. No input was truncated, duplicated or selected using reference answers or generated outcomes. This establishes input availability, not GPU capacity, proposal coverage or performance.

The scan ran once on 2026-10-16 **01:45:47.276–01:46:14.281 UTC** (09:45:47–09:46:14 UTC+8), for **27.005 seconds**, pinned to CPU 0 at nice 19. CUDA visibility was empty; Transformers/Hugging Face were offline and tokenizer parallelism was disabled. `cpu_execution.json` records the exact times, affinity and exit code 0; `scan.log` preserves execution output. These times identify any overlap with concurrent service measurements; low CPU priority is not proof of zero interference.

## Full-source counts

Counts use the entire existing task content template, complete original context and each target's declared rendering. All 500 LCC and 200 MultiNews records were scanned in source order. The maximum column includes ineligible records; no maximum was shortened to fit a model.

| Target tokenizer | Task | ≤2048 | ≤3072 | ≤3584 | Full-source min / max tokens | Smallest length cap giving 128 for this tokenizer |
|---|---|---:|---:|---:|---:|---:|
| OLMoE | LCC | 82 | 282 | 337 | 1264 / 34418 | 2249 |
| Qwen3 AWQ | LCC | 195 | 350 | 385 | 1017 / 32647 | 1796 |
| Granite3B | LCC | 129 | 317 | 354 | 1029 / 34657 | 2039 |
| OLMoE | MultiNews | 91 | 142 | 157 | 165 / 14661 | 2747 |
| Qwen3 AWQ | MultiNews | 98 | 149 | 158 | 154 / 14300 | 2608 |
| Granite3B | MultiNews | 71 | 123 | 143 | 157 / 17042 | 3154 |

| Same original source IDs eligible for all three | ≤2048 | ≤3072 | ≤3584 | ≤3833 | ≤3961 | Smallest common length cap for 128 |
|---|---:|---:|---:|---:|---:|---:|
| LCC | 81 | 280 | 337 | 348 | 360 | 2249 |
| MultiNews | 71 | 123 | 143 | 148 | 151 | 3154 |

The common sets are intersections by original `source_index`, preserving original JSONL order. `feasibility.json` stores the first 128 common indices at each sufficient cap; there is no reference or output filter. A 3072-cap MultiNews population is five records short and must not be silently padded. At cap 3584, the first 128 common LCC records have actual maximum token lengths OLMoE/Qwen/Granite **3559/3170/3458**; the corresponding MultiNews maxima are **3168/2988/3572**. At the smaller sufficient LCC cap 3072, maxima are **3059/2838/2950**.

## Model and rendering boundaries

| Target and neural source | Target / draft `max_position_embeddings` | Target rendering | Token compatibility observed on the complete 700-record source scan |
|---|---|---|---|
| OLMoE-0924 / OLMo-1B-0724 | 4096 / 4096 | Existing target chat template, one user message, generation prefix enabled. | The draft has no chat template, but encoding the already target-rendered text produces identical IDs for all 700 inputs. The target vocabulary has one extra padding entry; no scanned prompt uses it. Maximum observed ID 50279; both model input vocabularies are 50304. |
| Qwen3 AWQ / Qwen3-a3B EAGLE3 | 40960 / 40960 | Existing target `chat_template.jinja`, same one-user/generation-prefix call; no `enable_thinking` override. | EAGLE3 has no independent tokenizer. Its full input vocabulary is 151936; its 32000-entry draft output vocabulary is a separate native mapping boundary. Maximum scanned target ID 151645, with no out-of-input-vocabulary IDs. This scan did not load or revalidate EAGLE weights/mappings. |
| Granite3B base / Granite1B base | 131072 / 131072 | Neither tokenizer supplies a chat template. The same full content prompt is encoded as explicit raw continuation, without inserted chat tags/BOS. | All target vocabulary token-to-ID mappings equal the draft's 49152-entry vocabulary. All 700 rendered inputs re-encode identically. Maximum scanned ID 49151; no vocabulary-boundary violations. |

These are limits from the actual local model configs, not the tokenizer metadata's large sentinel lengths (or Qwen tokenizer's 131072). No RoPE extension is assumed. Target/draft context must both accommodate the prompt, output budget and draft lookahead; vLLM's installed `SpeculativeConfig._maybe_override_draft_max_model_len` uses the smaller target/draft limit unless a smaller explicit override is supplied (`runs/environment_20261016/source/config/speculative.py`, lines 1089–1124). Granite's distinct target/draft RoPE theta values are preserved in the report; they are separate model configurations, not grounds to rewrite either asset.

All families receive exactly the same task instruction and source context before rendering. OLMoE and Qwen add different native chat wrappers; Granite's base-model raw continuation is **not chat/instruction-following equivalence**. The Qwen default template behavior remains intact rather than silently disabling thinking. Within a target family, n-gram and neural-source runs should reuse the same target-tokenized records and split. Across families, common original source IDs preserve content identity, not token counts or model behavior.

## Minimal preparation choice

Use **128 common records per task**: first 64 development, next 64 held out, preserving original order after the common length filter. Reuse those exact IDs for all three targets and both draft sources. Among the prespecified caps, use LCC 3072 and MultiNews 3584; using 3584 for both is also feasible but retains longer LCC prompts. The exact mathematical minimum caps 2249 and 3154 are reported as input-length feasibility facts, not silently applied as new selection rules.

A future **4096-token engine context** accommodates either cap with 128 generated tokens plus G7; it also accommodates 3584 + 256 + 7 = 3847 for a separate natural-stop cap. The OLMoE/OLMo pair is the restrictive 4096 pair, so the larger Qwen/Granite advertised limits need not be enabled merely to prepare this common population. Existing ShareGPT plans at max_model_len=1024 cannot be reused unchanged for these long prompts. No assertion is made that the current fixed KV budget supports every nominal batch without admission effects; GPU memory/serving behavior was not measured here.

`prepare_aux_workloads.py` currently hard-requires max_prompt≤2048 and a chat template. Consequently these inputs cannot be reproduced through that unmodified CLI at 3072/3584, and its chat-only path would reject Granite. The saved candidates below avoid any second tokenizer scan: a later, explicitly scoped population builder can filter by the recorded token lengths and common indices, retain original order, and write the final workload files. No running configuration, queue, or existing preparer was changed in this task.

## Saved inputs and provenance

Six `*_complete_candidates.json.gz` files contain **every eligible complete record up to 3584**, not only the final 128: LCC OLMoE/Qwen/Granite = 337/385/354; MultiNews = 157/158/143. They retain the existing preparer schema (`source_index`, `source_id`, `task`, `source_member`, `prompt`, `prompt_token_ids`, `rendered_sha256`, `context_sha256`, `reference`, `references`, language/source-length metadata). References are copied for later task evaluation; their text is not tokenized or used to determine eligibility. Gzip level 1 limits preparation CPU work. Each artifact SHA and size is recorded in `feasibility.json` and matched after copying locally.

`population_lengths.csv` contains 2100 per-tokenizer/task/source rows, including ineligible inputs and their full lengths; `counts.csv` and `feasibility.json` retain threshold counts and selection indices. `scan.py` is the single executed study-specific CPU entry point. The source is the existing read-only `research_C_format_sched/datasets/serving_sources/longbench/data.zip`, 113932529 bytes, SHA256 `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`, pinned revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`. Actual model/tokenizer/config and script fingerprints are retained in the report. Original source files were not modified; there were no downloads, model-weight loads, GPU commands or generated answers. These are LongBench-derived serving inputs, not an official LongBench score.

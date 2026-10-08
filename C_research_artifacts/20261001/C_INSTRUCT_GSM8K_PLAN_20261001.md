# Instruct GSM8K development qualification

## Decision and hypothesis

The previous base-model BBH pilot produced 6/27 exact answers, with no EOS endings. The native APC pair removed its observed pressure (4 to 0 preemptions) but yielded only 1/32 correct answers. There is no justified new controller in that case. The current weakest link is useful, naturally terminating work.

Run one fixed native qualification with official `allenai/OLMoE-1B-7B-0924-Instruct`, revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`. The hypothesis is that instruction tuning plus the model's official chat format yields meaningful answer-bearing outputs and EOS variation on a fixed public reasoning cohort. This is not a policy comparison or a claim that changing the model improves scheduling.

## Fixed inputs and runtime

- Official GSM8K test source order, indices 0 through 15, all retained. Official eight-shot CoT exemplars in their original order. Historical Open-Instruct single-user chat construction and appended `Answer:` cue. No input filtering, truncation, prompt search, seed change or replacement requests.
- Greedy generation, seed 20260905, maximum 1024 new tokens, minimum 0, EOS enabled, no text stops. All arrive at time zero and drain. Require every prompt plus 1024 to fit the model's 4096 context before launching.
- Native vLLM 0.26.0 in C's existing private environment; BF16; native APC on; 4097 allocated KV blocks including one null block; 32 sequence limit, 1024 batched tokens. Two separate 16-token warmups on the first and last requests, then reset APC before measurement.
- Validate local prompt token IDs and exact chat rendering with the runtime tokenizer before CUDA initialization. Save all output token IDs, text, gold answers, finish/stop reasons, host delivery times, EOS configuration, preemptions and drain state.

The OLMoE evaluation driver used a random sample of 200 items. Its exact installed Open-Instruct commit is not recorded; our pinned historical source is the last upstream commit before that driver. The historical HF path caps generation at 512 tokens and uses its own text stop logic. Our first16, 1024-cap, EOS-only vLLM pilot is therefore not the published model score or a full GSM8K benchmark. Inputs are development data once inspected.

## Analysis and decision

Use the historical GSM last-number extraction after removing commas between digits, and report exact numeric-string match across all 16 requests. Separately report natural EOS, cap hits, generated lengths, repetitive suffix diagnostics and full per-item outputs. Inspect answer-bearing text as well as the automatic score. Runtime is descriptive only: initialization, download, warmup and the small instrumented episode are separate costs.

Correct answers with genuine EOS and varied lengths would support using this model/workload as a development basis for a subsequent serving experiment. Incorrect, repetitive or cap-dominated outputs weaken that basis. This small fixed qualification cannot establish general quality, inference efficiency, a pressure regime, or algorithm novelty. Do not rescue poor results by deleting requests or tuning prompts/caps/seeds. No new controller is selected by this plan.

## One bounded resource unit

Use the existing shared lock `/root/autodl-tmp/moe-research-gpu.lock`, inode `2304:29005388732`, nonblockingly, on GPU `GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36`. One owned child covers model staging, load, warmup, measurement, drain and exit; timeout 1560 seconds. No queued retry. Download has a 1200-second total deadline and one attempt per file.

The data disk lacks room for a second checkpoint. Stage the 13,841,132,070-byte checkpoint only in the newly created private directory `/dev/shm/c-olmoe-instruct-20261001-pilot-v1`, with cgroup and tmpfs headroom checks. Small metadata comes from exact fixed-revision Git objects retained in the frozen package; the HTTPS resolve endpoint for small metadata returns 403. Weight shards use the working fixed-revision HTTPS endpoint, with official LFS SHA-256 and byte-size verification. Keep the lock through child reap, GPU release, and removal of only this newly created stage. Preserve logs and failed results on disk. Existing base-model cache and other sessions' files are untouched.

Remote result root: `/root/autodl-tmp/c-research-20260930/c-instruct-gsm8k-qualification-dev-v1`. This file records the plan before the first model download or GPU execution of this qualification. The overall paper goal remains incomplete.

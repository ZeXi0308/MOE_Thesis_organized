# BBH base-model workload qualification

## Question and fixed choice

Current article continuations often hit max_tokens and repeat. Before selecting a further admission mechanism, check whether the existing base OLMoE produces interpretable task answers with observable stopping on a standard task family. This is one development qualification cell, not a controller or a full BBH evaluation.

Use all 27 BBH tasks in alphabetical order, example index 0 of each. No outcome-dependent choice, dropping, retry or prompt change. Official source commit: `9ee07bd481feebf959a6b59d61ea57bdcf30964d`. Prompts follow the historical [open-instruct evaluator](https://github.com/allenai/open-instruct/blob/d05effeb4df018dd82c15a956a4a58da82547eb5/eval/bbh/run_eval.py): remove the first two prompt-file lines, strip, then append `\n\nQ: {input}\nA:`. No chat template. Greedy, max_tokens **512**, stop `\n\n`, EOS enabled; 512 replaces the earlier tentative 1024 before any GPU run or output inspection.

The [official model evaluation snapshot](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct#evaluation-snapshot) lists the base model's 3-shot BBH score as 33.6. That motivates this task family; the 27 first examples, runtime and aggregation here cannot reproduce or validate that benchmark number.

## Execution and accounting

Pinned base model/revision and existing private vLLM 0.26 runtime; RTX 5090 UUID `GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36`. Native FCFS, max_num_seqs32, batch1024, context4096, 4097 physical KV blocks including the null block, prefix caching/offload/async disabled. No PF or C admission gate. All 27 external arrivals at t=0; no exclusion and full drain. Prompt lengths 230–2003, total24644; all prompt+512 <=4096.

One two-request application warmup using the first and last task, 16 output tokens, recorded separately. Host-observed token delivery and complete episode time retained; text-stop buffering makes these unsuitable for an unqualified comparison to earlier detokenize=False experiments. Gold answers are only stored for offline scoring, never passed to inference or the scheduler.

Run a single immutable root `c-bbh-qualification-dev-v1` through the existing shared lock inode `2304:29005388732`, nonblocking. Lock spans initialization, warmup, generation, drain and owned child exit/GPU release. No retry queue. Source/input hashes are in `C_BBH_QUALIFICATION_FREEZE_V1.json`.

### v1 premeasurement failure and v2 compatibility correction

At 2026-10-01 12:00:27 UTC, v1 exited in its first warmup engine step: vLLM calls `scheduler.schedule(throttle_prefills)`, but the new observational wrapper accepted no arguments. No measured 27-task output was produced. Child21721 was reaped and GPU returned to0MiB/no compute processes; SSH71195exit70. The immutable v1 failure root is retained locally and remotely.

v2 changes that wrapper to forward `*args, **kwargs` to the original scheduler and uses a new immutable root `c-bbh-qualification-dev-v2`; every prompt, sampling setting, scoring rule and request remains identical. Sources are `C_BBH_QUALIFICATION_CELL_V2.py`, `C_BBH_QUALIFICATION_LAUNCHER_V2.py` and `C_BBH_QUALIFICATION_FREEZE_V2.json`. This corrects an unmeasured implementation failure, not a repeat selected from task outcomes.

## Analysis and decision

Keep every output string, output token ID, answer, finish reason, actual stop reason, and prompt. Score the first regex match `[t|T]he answer is (.*?)\.` or the full output fallback; exact match ignores case and ASCII punctuation, following the historical script and HF metric. No alternative parser selected after seeing answers. Denominator always27; report every task. Incorrect or unparsable answers remain failures.

Report length-limit vs explicit stop-string vs EOS-token termination separately. A text stop is task completion syntax, not proof of natural EOS. Predeclared repetition diagnostic: longest periodic output suffix with period<=16, flag when >=256 tokens; it never filters scores. This threshold differs from the prior 1024-token article diagnostic and is not a matched quality comparison.

Answers with nontrivial correctness, task completion syntax and varied output lengths would establish a usable development task source, without proving representative production service or model accuracy. Predominantly failed/repeated/capped output would weaken this choice; preserve the failure rather than change the first-example selection or stopping rule. Even a useful qualification does not restore the retired peak-formula novelty. Further serving work needs separately frozen requests, arrivals, fair baselines and an actionable mechanism gap.

## Preceding chunk hypothesis

The two actual PF traces at batch4096 had zero whole-prefill budget or slot rejections; maximum joint budget need3095/3094. A CPU diagnostic at a hypothetical batch1024 found only4/4910 and5/4925 recorded states where progressive chunk growth fit4096 blocks but an immediate full-prefill occupancy envelope did not. All9 were actual AE-peak rejects; seven unique heads total. This is a bound on those states, not a reachable native counterfactual or a service benefit. No chunk controller is selected from it.

LightLLM already includes chunk-delayed lifetimes in [`ChunkedPrefillReq`](https://github.com/ModelTC/LightLLM/blob/main/lightllm/server/core/objs/req.py), used by its [chunked prefill queue](https://github.com/ModelTC/LightLLM/blob/main/lightllm/server/router/req_queue/chunked_prefill/impl.py). [Sarathi-Serve Algorithm3](https://apanwariisc.github.io/publications/osdi-2024-sarathi-serve/osdi24-sarathiserve.pdf) prioritizes decodes and schedules partial/new prefills from the remaining token budget; its abstract allocation predicate does not establish absence of a temporal KV check. Retain this narrow residual as unselected.

## Full-corpus CPU geometry while v2 was deferred

All6511 official examples fit prompt+512<=4096. Prompt median895, p951709, max2035. Largest32 fullcap reservations total5053blocks; smallest32 total1504. For geometric_shapes and salient_translation_error_detection, even the smallest32 reservations sum4904/4367blocks, but the task-wide exact common prefixes span1851/1577tokens (115/98 whole16-token blocks). These are structural counts with no APC execution evidence. They show why a same-task pressure experiment with APC disabled would omit a strong simple capacity mechanism; no such performance experiment or new controller is selected. The already-fixed27 input set remains unchanged.

# KV-budget experiment integrity audit

**Date:** 2026-09-12  
**Review class:** fresh Codex reviewer, same-family, provisional  
**Scope:** one narrow, read-only pass over the frozen four-cell KV-budget bundle, retained GPU results, analyzers, and current claim files. No remote access, no new GPU work, no `.aris` trace.

本文件审查中的未加前缀路径相对于 `../20260912_kv_budget_r01/`；主交付报告和独立重算在本目录。

## Overall verdict: WARN

The retained four-cell result is a real native vLLM request measurement, not a synthetic result. Its hashes, inputs, source, configuration, request denominators, saved metrics, and headline numbers close under an independent CPU rerun. The only material warning is environmental: GPU-process checks are point snapshots before initialization, before the measured episode, and after it; they do not prove continuous physical isolation. Both current reports state this boundary, so the warning limits generalization but does not invalidate the finite paired measurement.

## A. Provenance / real measurement: PASS

- The frozen runner creates real `LLMEngine` instances, submits the retained prompt token IDs with `engine.add_request`, advances them with `engine.step`, and records cumulative native outputs and host receipt times: `frozen/run_probe.py:91-99,128-144`; `frozen/native_capture.py:115-177,192-204`.
- The workload is source-ordered WikiText-103 train data with dataset revision, Arrow SHA256, tokenizer hashes, and a predeclared first-32-complete-articles rule: `inputs_preparation/prepared/long/config.json:17-38`. It is not generated from model outputs.
- The four launches all exited 0 and were read back as COMPLETE with 32 requests each: `execution.json:8-72`. Software, model-shard hashes, GPU UUID, and preflight process snapshot exist: `preflight.json:2-23`.
- Evaluation type is **native request measurement / real system telemetry**. There is no external task ground truth because no task-accuracy claim is evaluated.

## B. Denominators and normalization: PASS

- The metric path retains every arrived request at the actual observation end and records planned/not-yet-arrived counts: `frozen/metrics.py:10-29`. In all four complete cells, the denominator is 32 arrived and 32 completed requests.
- TTFT is arrival-to-first-token; mean TPOT is first-to-last token divided by `n-1`; ITL uses adjacent receipt times; SLO attainment is `passes / arrived`; throughput is `completed / observation_duration`: `frozen/metrics.py:62-155`. Percentiles use explicit linear interpolation and mark samples below 100 as small: `frozen/metrics.py:40-59`.
- Each primary raw has 32 requests, 1,024 returned tokens per request, 32,768 one-token output events, and no multi-token chunks. Thus TPOT/ITL are not inflated by hidden within-chunk interpolation. The report correctly separates pooled-token ITL from request-max ITL: `RESEARCH_REPORT.md:155`.
- Relative changes divide by the matched 0.90 baseline, not by a statistic of the proposed output: `analyze_kv_budget.py:74-90`. No self-max/min score normalization was found. The inherited 5 s / 200 ms SLO is explicitly secondary and all four cells pass it, so it is not used to manufacture separation: `RESEARCH_REPORT.md:155`.

## C. Files, hashes, commands, and numbers: PASS

- The local execution bundle SHA256 is `e9649d8314d593d9c0b5a879483e471d8ddda7bd57482aa3acf9ead6d9c2880b`, matching `execution.json:75`. All four retained cell archives also match their recorded SHA256 values at `execution.json:15,31,47,63`.
- All eleven frozen bundle files checked byte-for-byte against `execution.tar.gz`; the five executed source hashes match every cell environment. One direct example is `gpu_results/repeat0-budget95/environment.json:12-23`; the aggregate source/software/warmup result is reported at `analysis/report.md:30-35`.
- The four launcher metadata files preserve executable, arguments, cwd, child PID, launcher hashes, timestamps, and return code. Example: `gpu_results/repeat0-budget95-execution.json:2-26`. Per-cell `commands.txt:1` agrees with the metadata.
- `engine_args.json` is identical across all four cells after removing only `gpu_memory_utilization`; the analyzer enforces this at `analyze_kv_budget.py:135-140`. Config/raw/qualification identity, pool conservation, and saved-metric equality are enforced at `analyze_kv_budget.py:33-68`.
- A fresh CPU invocation of the frozen analyzer reproduced the same report byte-for-byte. Its `analysis.json` differed from the retained JSON only in absolute versus relative path strings; scientific fields were identical. The two retained analysis copies are identical, and `RESULTS.json:2-103` matches the canonical analysis values.
- The current headline values match the retained raw-derived analysis: 32/32 per cell; 0.95 versus 0.90 throughput changes +3.9928% and +4.6985%; wall changes -3.8395% and -4.4876%; 0 versus 2 preemptions; 0 versus 7,685 recomputed tokens; max ITL below 0.1 s versus about 4.6 s: `analysis/report.md:9-26`; `RESEARCH_REPORT.md:144-157`.

## D. Metric call path: PASS

- `summarize_episode_requests` is called when each GPU cell writes `metrics.json`: `frozen/run_probe.py:139-144`.
- The analyzer calls the same summary again from raw requests, compares every recomputed top-level metric value with the saved metrics, then runs interval-based recomputation/preemption accounting: `analyze_kv_budget.py:44-58`.
- `_distribution` and `summarize_requests` are reached by `summarize_episode_requests`: `frozen/metrics.py:24-25,40-59,62-155`. No defined headline metric was found to be dead or populated without a call path.

## E. Scope and GPU isolation: WARN

- Exact measured scope is one RTX 5090, pinned OLMoE BF16, native vLLM 0.26.0, fixed cap32, 32 repeated WikiText articles, fixed 3,072-token prompts and 1,024-token outputs, two counterordered repeats: `frozen/DECISIONS.md:15-23,46-50`; `RESEARCH_REPORT.md:144-163`.
- `gpu_state()` rejects another visible compute PID, but it is invoked only before engine initialization, before the primary episode, and after the primary episode: `frozen/run_probe.py:27-34,74,129,138`. For example, `gpu_results/repeat0-budget95/environment.json:7-10` is empty before initialization, while `gpu_results/repeat0-budget95/gpu-after.json:2-3` contains only that cell's Python PID after measurement.
- There is no continuous process/utilization trace. Therefore **continuous physical GPU isolation is unverified**. The claim files correctly retain that limitation: `RESEARCH_REPORT.md:159`; `../20260912_admission_paging_research_r01/REPORT.md:178`.
- The 128 completed request executions are four measurements of the same 32 documents, not 128 independent texts. `RESEARCH_REPORT.md:144` states this correctly. Request p99 values are descriptive, with no population significance or production-tail claim.

## F. Evaluation type and state/output boundary: PASS

- Classification: **native request measurement; external task GT not applicable**. Exact generated-token agreement is a paired diagnostic, not a quality score.
- All 32 output-token sequences match between 0.90 and 0.95 within each repeat; the retained check says 32/32 for both repeats and labels it non-quality evidence: `cpu_analysis/output_agreement.json:1-13`. Independent hashes over request ID plus output-token arrays also matched across all four cells.
- This supports only exact output-token agreement for this fixed cohort/configuration. It does **not** establish same pre-action state, KV-state equality, route equality, hidden-state equality, or policy-specific state equivalence: each arm is a fresh engine with a different KV pool and, in the 0.90 arm, a different preemption/recomputation trajectory. The report claims output identity and explicitly stops at that boundary: `RESEARCH_REPORT.md:159,198-202`.

## Claim impact

**Supported within the stated finite scope**

- Increasing ordinary vLLM KV budget from 0.90 to 0.95 at fixed cap32 increased the realized KV pool, removed the observed native preemptions/second-scale request pauses, improved episode wall time and throughput in both repeats, and slightly worsened mean completion latency and median mean-TPOT. This is a configuration intervention with a measured tradeoff.
- The four-cell source/configuration/input/output and request-accounting claims in `RESEARCH_REPORT.md:144-163` and `../20260912_admission_paging_research_r01/REPORT.md:167-193` are supported.

**Unsupported if asserted beyond the current wording**

- Continuous GPU exclusivity; statistical significance; 128 independent documents; all requests or all metrics improving; population/production p99; task quality; natural-EOS behavior; second-model or multi-GPU/EP transfer.
- Same-memory scheduling gain, same-state causal policy comparison, exact Oracle/headroom, route/KV/hidden-state equivalence, expert paging/H2D benefit, expert-aware controller value, or method GO.

The current reports already avoid these upgrades. No full Oracle or fixed effect threshold is required to retain this `MEASUREMENT_ONLY` result.

# Experiment Audit Report

**Date:** 2026-09-14  
**Auditor:** fresh same-family Codex reviewer, read-only, provisional  
**Scope:** lifecycle accounting only: `analyze_effective_recovery_service.py`, its four targeted tests, `lifecycle/analysis.json`, the eight enumerated raw captures, the frozen capture/telemetry producers, and lifecycle statements in `REPORT.md`. The service-window model, neighbors, prior performance claims, and the later six-cell run are out of scope.

## Overall verdict: PASS (provisional)

No P0 or P1 integrity finding was found. This audit does not raise the scientific evidence tier.

## A. Reference provenance: PASS

The reference is observed native-serving execution, not a replayed or model-generated proxy. The frozen capture wraps each call, invokes synchronous `engine.step()`, and marks the call complete only after it returns (`native_capture.py:133-144`). Executed high-water advances only after that return (`native_capture.py:145-148`), and output tokens receive the same host-return timestamp (`native_capture.py:149-177`). The telemetry producer explicitly says schedule completion alone is not execution confirmation (`memory_telemetry.py:109-153`). The analyzer therefore requires one aligned successful engine-call receipt for every scheduler step (`analyze_effective_recovery_service.py:55-66`).

Independent raw verification found 14,984/14,984 aligned successful call receipts. Every legacy `model_execution_confirmed` value was `None`; none was changed to `True`, and no `False` receipt was accepted. Actual preemption calls returned and recorded `computed_tokens=0`, `block_counts=[0]`, with output history unchanged (`memory_telemetry.py:83-107`; `analyze_effective_recovery_service.py:73-77`).

## B. Accounting: PASS

The outcome buckets are disjoint (`analyze_effective_recovery_service.py:26-50`). A residency starts at one actual preemption and closes immediately before the same request's next preemption; terminal residencies close at capture completion (`analyze_effective_recovery_service.py:79-104`). Per-cell recomputation conserves exactly:

- fit-scan: `5,973 + 48,329 + 21,440 = 75,742`
- rank-prefix: `6,978 + 42,578 + 25,426 = 74,982`
- first-output guard: `0 + 67,554 + 29,401 = 96,955`

The analyzer also checks the bucket total against every scheduler-step recompute count (`analyze_effective_recovery_service.py:129-134`). Short service is a subset of `served_then_discarded`, and the report correctly avoids adding it again (`REPORT.md:19-25`). Inclusive call time, invalidated state, and next-residency reexecution are explicitly non-additive diagnostics (`lifecycle/analysis.json:7-10`).

The next-residency prefix intersection is valid for this adapter. Every recovery begins at computed position zero, advances contiguously without a hidden adjustment/load, and the runtime preconditions reject prefix caching and connectors (`analyze_effective_recovery_service.py:70-71,91-101`; frozen `ltr_recompute_native.py:131-144` in the packing package and `:122-135` in the restore package). Thus adjacent recovery prefixes are both `[0,R)`, so their observed intersection is `min(R_i,R_{i+1})` (`analyze_effective_recovery_service.py:124-127`). Independently, every 1-2-output short row had its full recovery prefix reexecuted next residency: 37,851 positions for fit-scan and 13,767 for the guard. This remains an overlapping work-position diagnostic, not recoverable time.

## C. Result existence and numbers: PASS

All eight absolute raw paths exist, and all eight independently computed SHA-256 values match `cells[*].raw_sha256`. A separate implementation that did not import the analyzer reproduced every cell summary, closed-service histogram, and every residency field exactly. It also confirmed 32/32 completed requests and 32,768 outputs per cell. The lifecycle table and worked example match those records (`REPORT.md:17-29`).

The output-time boundary is honest: token times are host receipt immediately after synchronous `engine.step()` return, with no client/network receipt claim (`lifecycle/analysis.json:2-6`; `REPORT.md:33`). All 262,144 token timestamps matched successful engine-call return timestamps; every raw reports single-token chunks without interpolation. This is not a GPU event-duration measurement.

## D. Called metric code: PASS

`main()` reads each raw and calls `analyze()`, which calls `summarize()`, then requires exactly eight cells before writing the artifact (`analyze_effective_recovery_service.py:129-170`). No lifecycle metric function is dead. The four targeted tests exercise served-then-discarded accounting, internal reuse versus terminal loss, failed execution receipts, future-output leakage, and hidden prefix loads (`test_effective_recovery_service.py:33-61`); all 4/4 passed.

## E. Scope and repeat independence: PASS (limited scope)

The artifact contains eight completed executions over one shared cohort of 32 unique request IDs, 32 unique documents, and 32 unique prompt hashes. All cells had the same request/workload signature. Block repeats for each rule also had identical output signatures; the rank-prefix and restore-off groups add execution repeats, not new request or document samples. `REPORT.md:17` states this boundary directly, and `REPORT.md:49-57` keeps later runs and untested regimes outside the lifecycle result. The evidence supports bounded existence/accounting statements only; it does not establish cross-workload frequency or robustness.

## F. Evidence type: PASS

Classification: **offline reanalysis of real native-serving in-process host captures** (`REANALYSIS_OF_NATIVE_SERVING`), with status `MEASUREMENT_ONLY` (`lifecycle/analysis.json:2-6`; `REPORT.md:3,13`). The supported ceiling is observed recovery/output/invalidation lifecycle accounting at the engine host-return boundary. Client/network timing, isolated GPU recompute time, quality, and performance-method benefit were not measured by this analysis.

## P0/P1 action items

None.

## Claim impact

- The three per-rule lifecycle ledgers and the concrete request `0003640` examples are supported.
- Reexecution of the reported next-residency prefix is supported only under the checked no-APC/no-connector, computed-reset adapter.
- Any conversion of positions or inclusive call time into saved milliseconds is unsupported.
- No quality or performance-method claim was audited or established.


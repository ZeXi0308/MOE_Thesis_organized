# N0b Native Telemetry Qualification Report

Date: 2026-08-23  
Repository HEAD: `b141c1d587fe2c918643c3c7c3a8f5f5157d4c8a` (dirty; the complete
`refine-logs/expert_saturation/` campaign remains uncommitted)  
Canonical campaign: `n0b-valid-window-20260823-westd-r01`

## Verdict

`CONDITIONAL_NO_GO` for the tested lossless, host-visible routed-expert
telemetry formulation. Corrected campaign status:

```text
VALID_WINDOW_NOT_TRANSPARENT
TELEMETRY_TOKEN_DRIFT
```

The sealed v2 reducer originally surfaced repeat 0's timing failure before
repeat 1's stronger token-drift failure. The raw artifact remains unchanged;
the append-only evaluator-v3 replay corrects the failure attribution.

## Evidence type

`NATIVE_OFFLINE_FIXED_BATCH` on one RTX 5090 using vLLM 0.26.0, Python 3.12.3,
Torch 2.11.0+cu130, OLMoE-1B-7B at revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`, BF16, eager execution.

The campaign retained two counterbalanced process repeats of four fresh-process
arms: stock/valid-window source crossed with route telemetry OFF/ON. All eight
bundles passed source identity, producer identity, artifact integrity,
exclusive-GPU, process cleanup, and repeat-coverage validation. An independent
read recomputed all 464 manifest entries with zero mismatches.

## What was measured

| Repeat | Optimized token parity | Cross-runtime OFF parity | Lossless route comparison | Optimized P95 absolute TPOT / wall deviation |
|---|---|---|---|---|
| 0 | 36/36 cells exact | PASS | 36/36 comparable and exact | 28.26% / 43.77% |
| 1 | FAIL at `[512,16,1,0]` | PASS | 34/36 comparable; 34/34 exact among comparable | 20.34% / 28.24%, diagnostic only |

The stock ON/OFF control also drifted in repeat 1 at `[512,8,2,0]`. The stock
and optimized route-OFF arms were output-identical across implementations and
across their two same-arm repeats in all 36 cells. The two route-ON arms were
same-arm stable in 35/36 cells each; their unique drift cell was also the unique
cell with a changed top-k route set across repeats.

The corrected repeat-localization diagnostic retains an important limit: the
saved route artifact drops the prompt-tail forward. Saved route step `s`
produces generated token `s+1`; after correcting this alignment, captured route
divergence is no later than output divergence for only 3 of 5 drifted requests.
The result is therefore `PARTIALLY_LOCALIZED`, not proof of a KV propagation
chain.

## What was not measured

- no continuous-arrival serving, queue, request-level SLO, TPOT tail under load,
  or SLO-goodput;
- no pressure-to-latency relationship, saturation knee, action Oracle, batch
  formation, admission policy, or Controller;
- no same-prestate causal split and no pre-router hidden/router-logit/final-logit
  first-divergence capture;
- no CUDA-graph, async scheduling, speculative decode, multi-GPU, Expert
  Parallel, A2A, or rank-straggler result.

## Strongest baseline

The strongest simple baseline is stock vLLM 0.26 full routed-expert export. The
tested optimization clears only the valid route window while preserving the
same full-history API. Route-OFF under the same source is the paired execution
baseline. The optimization did not qualify: one retained repeat changed output
tokens and the parity-valid repeat exceeded the frozen two-sided 5% timing Gate.

## Oracle/headroom status

`UNRUN / NOT AUTHORIZED`. N0b is a measurement-path qualification Gate, not a
capacity or action-headroom experiment. Decode-cap and Controller experiments
remain blocked.

## Claim ceiling

```text
NATIVE_OFFLINE_FIXED_BATCH_TELEMETRY_IMPLEMENTATION_ONLY
```

Defensible claim: in this exact single-GPU eager regime, the lossless
host-visible route-export path is neither repeat-transparent nor bounded within
5% timing deviation. This does not establish that telemetry caused every drift
event, because arms are separate fresh processes.

## Failure category

- Primary: `TELEMETRY_TOKEN_DRIFT`.
- Secondary: `TELEMETRY_TIMING_DEVIATION_ABOVE_THRESHOLD` in the parity-valid
  repeat.
- Evaluator correction: v2 campaign reduction was repeat-order dependent; v3
  uses explicit cross-repeat failure precedence and reports every repeat
  failure.
- Localization limit: `PROMPT_TAIL_ROUTE_MISSING_OR_ROUTE_AFTER_OUTPUT_FOR_SUBSET`.

## Resurrection condition

Do not repeat the same eight-arm formulation to seek a favorable run. Reopen
lossless telemetry only if a new implementation or deterministic runtime mode
passes exact output parity in every retained repeat and the frozen 5% bound. A
compact GPU pressure sketch is a different telemetry contract and requires its
own semantic and overhead Gate.

## One next smallest experiment

Run N0c on exactly the two preselected drift cells, retaining the prompt-tail
forward and explicit `forward_input -> produced_output_token` indices. Use
independent route-OFF repeats as the negative control and capture the minimum
chain:

```text
pre-router hidden
-> router logits and top-k boundary
-> selected expert set
-> expert output
-> final logits / sampled token
```

The experiment is diagnostic only and does not time the instrumented arms. If
OFF/OFF also diverges, classify baseline runtime nondeterminism. If OFF/OFF is
stable but ON repeatedly diverges, localize the first differing operator before
choosing between global batch-invariant execution and a compact telemetry
contract.

## Evidence pointers

- Raw terminal: `outputs/native_route_shape/n0b_valid_window_20260823_westd_r01/CAMPAIGN_COMPLETE.json`
- Raw sealed result: `outputs/native_route_shape/n0b_valid_window_20260823_westd_r01/valid-window-gate.json`
- Evaluator correction: `outputs/native_route_shape/n0b_valid_window_20260823_westd_r01/ADDENDUM.md`
- Corrected replay: `outputs/native_route_shape/n0b_valid_window_20260823_westd_r01/valid-window-gate.evaluator-v3-r2.addendum.json`
- Alignment correction: `outputs/native_route_shape/n0b_valid_window_20260823_westd_r01/REPEAT_LOCALIZATION_ADDENDUM.md`
- Corrected localization: `outputs/native_route_shape/n0b_valid_window_20260823_westd_r01/repeat-divergence-localization-v2.json`

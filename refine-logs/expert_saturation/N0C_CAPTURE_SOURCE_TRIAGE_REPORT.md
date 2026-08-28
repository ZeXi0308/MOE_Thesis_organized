# N0c Capture-Source Triage Report

Date: 2026-08-23  
Repository HEAD: `b141c1d587fe2c918643c3c7c3a8f5f5157d4c8a` (dirty; the
`refine-logs/expert_saturation/` line remains uncommitted)  
Canonical campaign: `n0c-capture-stage-20260823-westd-r01`

## Verdict

```text
NOT_REPRODUCED
NO_CAPTURE_OR_EXPORT_SOURCE_LOCALIZED
```

Both preregistered N0b drift cells were token-identical across OFF,
capture-enabled/no-export, and full-export execution in all four fresh-process
rounds. Full-export route arrays were also repeat-identical. N0c therefore did
not reproduce a stable telemetry-path perturbation source.

This is a precise negative result for the two frozen cells and replay contract.
It does not invalidate N0b's retained raw mismatch, prove that all vLLM route
telemetry is transparent, or falsify the broader batch-dependent execution
conformance family.

## Evidence type

`NATIVE_SINGLE_GPU_FRESH_PROCESS_ASSOCIATIONAL` on one RTX 5090 with vLLM
0.26.0, Python 3.12.3, Torch 2.11.0+cu130, OLMoE-1B-7B revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`, BF16 and eager execution.

The campaign contains 32 write-once arms: two exact targets, four process
rounds and four counterbalanced arms per target. Every arm replayed the same
six warm-up shapes and the exact frozen N0b semantic prefix. All 32 arm process
groups exited cleanly without retry, and the GPU was idle after every arm and
after the campaign.

## What was measured

For each target and round:

- `n_a` and `n_b`: sealed base runtime with route capture OFF;
- `capture_only`: capture enabled while sync D2H, output wrapping and async
  snapshot/export were suppressed;
- `full_export`: complete route export including the prompt-tail forward.

The locked decision unit was the first generated-token divergence signature
`(output_token_index, request_row, baseline_token, observed_token)`. Results:

| Target | OFF mismatch | Capture-only drift rounds | Full-export drift rounds | Full-export route drift | Status |
|---|---:|---:|---:|---:|---|
| stock P512/B8/G2/W0 | 0/8 OFF arms | 0/4 | 0/4 | 0/3 repeats vs r0 | `NOT_REPRODUCED` |
| valid-window P512/B16/G1/W0 | 0/8 OFF arms | 0/4 | 0/4 | 0/3 repeats vs r0 | `NOT_REPRODUCED` |

The 16 arm outputs per target retain 6,144 generated token IDs in total; every
matrix equals its target's canonical OFF output. The four full-export arms per
target retain 196,608 routed expert IDs in total; exact arrays and top-k sets
are repeat-identical.

The terminal binds:

```text
run_plan_sha256                  a3bda498cdcef5d2d32f4da1d0eeb1927a2d6975e5f25df8cd632336e705f93e
runtime_package_manifest_sha256 c847378c007f87333a622a7fdb2adac31f8654c00f71a01078559bd461e6a823
verdict_sha256                   cb4f39473da67c4982c657fbefbc98277c38aa05e429e0424ccd36328dcdfff1
```

The local lightweight replay independently rechecked the three terminal
hashes, 32 bundle seals/payloads, 58 frozen files, prompt and route NPZs,
schedule, both target reports and 12 key runtime source files. It reproduced
the sealed `NOT_REPRODUCED` verdict exactly. The four full 2.9 GiB runtime
snapshots remain sealed remotely; their complete package manifests were
recomputed by the campaign evaluator but were intentionally not downloaded.

## What was not measured

- no same-physical-prestate branch or in-process fork;
- no pre-router hidden state, router logits, top-k margin, expert output or
  final-logit first-divergence observer;
- no continuous-arrival queue, native online batching, TTFT/TPOT/P99,
  request-level SLO or SLO-goodput;
- no action-conditioned capacity Oracle, admission policy or Controller;
- no CUDA graph, async scheduler, speculative decode, multi-GPU EP, A2A or
  rank-straggler result.

`capture_only` also retains scheduler route-manager bookkeeping, slot mapping
and the experiment patch/config. Even a positive result would not have isolated
a pure device kernel.

## Strongest baseline

The strongest execution baseline is full host-visible route export under the
sealed stock and valid-window vLLM packages. Two independent capture-OFF arms
per round guard baseline nondeterminism; the capture-enabled/no-export arm
separates the common capture/bookkeeping path from the broader export path.

All three paths were output-identical in this campaign.

## Oracle/headroom status

`NOT_APPLICABLE / LOCKED`. N0c is source triage, not an action experiment.
`controller_unlocked=false` and `action_oracle_unlocked=false` are sealed in the
verdict.

## Claim ceiling

```text
FRESH_PROCESS_ASSOCIATIONAL_CAPTURE_TRIAGE_ONLY
```

Defensible claim: the two historical N0b route-ON drift cells did not recur in
four counterbalanced fresh-process replays under OFF, capture/no-export or
full-export paths. No stable telemetry-path source was identified.

## Failure category

`HISTORICAL_EVENT_NOT_REPRODUCED_UNDER_FROZEN_REPLAY`.

The first remote source bundle passed its uploaded-file hashes but failed
preflight before GPU execution because the upload omitted an import-time helper
dependency. The attempt is preserved as `src-r01`; it created no campaign
output. The corrected `src-r02` added an isolated import-closure regression,
passed independent P0/P1 review and produced the sole scientific campaign.

## Resurrection condition

Do not repeat N0c with more seeds, cells or favorable arm orders. Reopen this
exact vLLM telemetry-source formulation only if a new deterministic runtime
mode or a preregistered same-physical-prestate mechanism makes the historical
event directly testable. A new naturally observed divergence must be frozen
before inspecting its telemetry source.

## One next smallest experiment

Run the already prepared custom-runtime serial-vs-batch-4 router-logit
conformance Gate on the frozen four-request/8-step trace. Its single question
is whether the repeat-stable custom-runtime expert-assignment divergence is
already present in pre-top-k router logits or is confined to near-tie top-k
amplification.

This is a new execution-conformance diagnostic, not an N0c retry. It must use
matched request/decode/layer identity, three repeats per width, token parity,
within-arm stability and an idle GPU. It remains below native serving and does
not authorize a capacity action.

## Evidence pointers

- Protocol: `experiments/vllm_patches/N0C_CAPTURE_SOURCE_TRIAGE.md`
- Terminal: `outputs/native_route_shape/n0c_capture_stage_20260823_westd_r01/CAMPAIGN_COMPLETE.json`
- Verdict: `outputs/native_route_shape/n0c_capture_stage_20260823_westd_r01/n0c-stage-a-verdict.json`
- Frozen plan/source/input bundle: `outputs/native_route_shape/n0c_capture_stage_20260823_westd_r01/frozen/`
- Arm evidence: `outputs/native_route_shape/n0c_capture_stage_20260823_westd_r01/bundles/`
- Key runtime sources: `outputs/native_route_shape/n0c_capture_stage_20260823_westd_r01/runtime-key-files/`

# Logical alignment validation r01 — narrow new-data audit

## Verdict

**PASS_NEW_DATA_INTEGRITY / STOP_STABLE_Y_INCREMENTAL_CLAIM — P0: 0, P1: 0.** V has no hidden execution failure, missing request, dropped output, or unaccounted cell. The second cohort does **not** reproduce the P result that Y beats both F and X in both blocks: Y is 1.768370% slower than X in the forward block and only 0.064751% faster in the reverse block, while Y's complete subprocess cost is 4.491727% and 3.542291% higher than X. The fixed stable-Y-over-X performance claim must stop, with no additional cohort or threshold tuning.

This is a same-reviewer, non-fresh follow-up. It inherits the accepted P implementation audit at `logical_alignment_performance_r01/EXPERIMENT_AUDIT.json` (SHA-256 `08a8b47ad9056ac85c1adcf12fa942ee7e4320bf30b59c6ac351444e74d46d9a`) and does not re-audit the 11 runtime sources or the prior numerical qualification. The inherited review is same-family provisional, so this V acceptance remains **provisional**. No `.aris` trace was created.

## Frozen-input delta from P

P and V each freeze the same 40 path names. All 40 V files rehash against both the root and attempt manifests. Thirty-three files are byte-identical to P, including `run_attempt.py`, `driver_support.py`, `finite_metrics.py`, `analyze_covered.py`, all runtime sources, all instrumentation, all analysis-source dependencies and both allocation files.

Exactly seven inputs changed:

- `protocol.json`: cohort identity, timestamps, reversed arm order and validation-scope text;
- `run_cells.json`: `F/X/Y/Y/X/F` becomes `Y/X/F/F/X/Y`;
- `input_provenance.json`, `prepare_inputs.py`, and `prepared/workload.json`: exact reconstruction extends from eligible 1..112 and selects the next source-ordered documents 113..128, adding P as a checked prior input;
- `COMMANDS.md`: validation-cohort commands and documentation;
- `analyze_logical_performance.py`: fixed expected labels, block-pair indices and same-arm repeat indices.

The V and P document IDs are disjoint, the 16 V prompts are all 128 tokens, and the arrival schedule remains `0..3.75 s` at `0.25 s`. No mechanism, execution driver, resource setting, metric formula or shared analysis dependency changed.

Evidence hashes:

- V 40-input manifest: `f9d71b756a5a7aa33bfeefcc2127bfd202056062024e9c91b765df98406efbe9`;
- attempt readback manifest: `72cdf9bdc0a88d0093a0822d96dedc7069e3a3829bfd93c16f1739767ec52f5c`;
- retained execution: `67ee4e0a146b48d18f6f701f2621cc88004f4bf2c4bc36c7b517bffc35b121dc`;
- retained remote analysis: `4cc9c6b3f93e52e72ba74b9605ee4c6b67f5f689744f59755021848cbccaa456`.

## Independent raw reconstruction

All six declared cells are COMPLETE with return code 0, raw error `null`, coverage valid, and 16 of 16 requests completed. Every request produced all 32 requested tokens; every cell therefore retains 512 output tokens and 512 one-token output events with valid cumulative prefixes. No cell records preemption or recomputed tokens.

| Cell | Capture (s) | Last completion (s) | Max request ITL (s) | Full process (s) | Requests | Output tokens/events | Steps | Scheduled positions | Measurement pager calls / layer rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0_Y | 8.086751 | 8.086633 | 0.184204 | 60.794342 | 16/16 | 512/512 | 57 | 2544 | 912 / 40704 |
| 1_X | 7.946232 | 7.946117 | 0.181690 | 58.181009 | 16/16 | 512/512 | 57 | 2544 | 912 / 40704 |
| 2_F | 8.307977 | 8.307679 | 0.201219 | 60.849539 | 16/16 | 512/512 | 56 | 2544 | 896 / 40704 |
| 3_F | 8.359879 | 8.359544 | 0.205079 | 60.707074 | 16/16 | 512/512 | 56 | 2544 | 896 / 40704 |
| 4_X | 8.120389 | 8.120245 | 0.191748 | 58.569496 | 16/16 | 512/512 | 57 | 2544 | 912 / 40704 |
| 5_Y | 8.115131 | 8.114824 | 0.186053 | 60.644198 | 16/16 | 512/512 | 57 | 2544 | 912 / 40704 |

The apparent call-count difference is fully explained by the actual scheduler-step count: each nonempty step has 16 layer calls. All cells schedule the same 2544 token positions and therefore have the same 40704 layer-token rows. This is retained trajectory behavior, not a missing denominator.

## Frozen comparisons

Percentages below are target relative to baseline; negative capture/process values are faster.

| Block | Comparison | Capture | Full process | Max request ITL |
|---|---|---:|---:|---:|
| forward | Y vs F | -2.662817% | -0.090711% | -8.455830% |
| forward | Y vs X | **+1.768370%** | **+4.491727%** | +1.383613% |
| forward | X vs F | -4.354189% | -4.385455% | -9.705161% |
| reverse | Y vs F | -2.927652% | -0.103572% | -9.277033% |
| reverse | Y vs X | **-0.064751%** | **+3.542291%** | -2.969797% |
| reverse | X vs F | -2.864756% | -3.521135% | -6.500281% |

The two-engine descriptive means give Y vs X `+0.841874%` capture and `+4.015430%` full process. X beats F in both blocks; Y does not beat X across both blocks and has higher full-process costs in both. Thus V's strongest observed completion baseline is X, and Y does not reproduce an incremental full-request advantage over that baseline.

P reported Y/X capture changes of `-3.196768%` and `-2.408519%`. V changes these to `+1.768370%` and `-0.064751%`. The forward comparison reverses sign, and the reverse result shrinks to a near-zero descriptive difference. At the same time X/F changes from positive in both P blocks to negative in both V blocks. These results rule out a stable two-cohort ordering; they do not define a population noise floor.

## Same-arm repeats and trajectory boundary

| Arm | Second vs first capture | Second vs first process | Second vs first max ITL | Equal request outputs | Route hash equal |
|---|---:|---:|---:|---:|---|
| F | +0.624720% | -0.234126% | +1.918209% | 7/16 | false |
| X | +2.191687% | +0.667721% | +5.535643% | 10/16 | false |
| Y | +0.350941% | -0.246969% | +1.003946% | 9/16 | false |

The route digests were independently rebuilt from each measurement pager trace using layer, step, row identity/position and top-k expert IDs. Every same-arm pair differs. These are two ordered fresh engines sharing one workload cohort, so neither the output/route variation nor the timing differences can be promoted into an overall noise estimate or statistical sample size. The performance rows remain valid policy-specific request trajectories, but causal attribution to a trajectory-invariant alignment-only saving is unsupported.

## Failure and reporting check

No raw or execution failure was found, and the retained `attempt01/analysis.json` has no issue or missing-cell entry. Its fixed cell values and paired comparisons match the independent reconstruction.

The material new result is the Y/X directional non-replication above. At the audit snapshot, top-level `REPORT.md` still described GPU as UNRUN and contained no post-run values; it must not be cited as the V result until updated. This is an in-progress reporting state, not a defect in the retained raw data. The direction flip is explicitly retained in this audit and must remain in the final report.

## Claim ceiling

- **Verdict:** `PASS_NEW_DATA_INTEGRITY / STOP_STABLE_Y_INCREMENTAL_CLAIM`, provisional; P0=0, P1=0.
- **Evidence type:** second independent-document `REQUEST_LEVEL / NATIVE_SERVING_INPROCESS_HOST_CAPTURE` cohort, with implementation integrity inherited from P.
- **What was measured:** six fresh engines on one new 16-document finite-arrival cohort, including capture, completion, max ITL and full subprocess cost.
- **What was not measured:** a population noise floor, statistical significance, task quality, deterministic same-arm trajectories, production SLO, steady state, multi-GPU EP or general model/runtime benefit.
- **Strongest baseline:** X in V; it was faster than F in both blocks and Y failed to beat it consistently.
- **Oracle/headroom status:** no oracle was defined or measured.
- **Failure category:** `DIRECTIONAL_NONREPLICATION_AGAINST_SAME_BUDGET_X`.
- **Allowed claim:** two cohorts show Y faster than F on capture, while the proposed incremental Y-over-X benefit does not replicate and full-process Y/X is adverse in both V blocks.
- **Disallowed claim:** stable logical64 performance benefit, significance/noise bound, quality equivalence, method GO, or paging-family NO-GO.
- **Next step:** stop this stable-benefit claim and do not add another cohort or tune thresholds from P/V.

The second cohort therefore answers the research question negatively at the requested ceiling: **it does not support continuation of a stable Y improvement against both baselines.**

# LTR component longest-gap localization

MEASUREMENT_ONLY_EVENT_LOCALIZATION

Observed same-input native-recompute component trajectories only; no counterfactual, new scheduler, full LTR or method failure. Gap output window is (previous receipt,next receipt]; scheduler steps and completed engine calls are joined by step index. Scheduler/decision times are nested in engine time, never added or subtracted as savings. Future receipts are outcomes only.

| cell | max-gap request | gap s | calls | max consecutive unselected | idle resets without output | boosted calls in max gap |
|---|---|---:|---:|---:|---:|---:|
| block0-d6-boost-on | memory-train-article-0003571 | 4.928370 | 245 | 125 | 6 | 0 |
| block1-d6-boost-on | memory-train-article-0003571 | 4.867612 | 245 | 125 | 6 | 0 |

block0-d6-boost-on: output 311→312, steps 405–649. Target partial-service steps [518, 519, 520, 646, 647, 648] assign idle=0 without a new token; positive idle values reset are [113, 125]. Total target recomputation 6367 positions; held calls 14; other requests return 7122 tokens in the same gap. First boost at step 609 selects memory-train-article-0003640 and evicts ['memory-train-article-0003345']; its 10-call epoch returns 7 new outputs. Quantum exhaustion without new output: 0. Gap engine calls 4.812745s, outside-engine host intervals 0.115625s; nested scheduler 0.212898s.

block1-d6-boost-on: output 311→312, steps 405–649. Target partial-service steps [518, 519, 520, 646, 647, 648] assign idle=0 without a new token; positive idle values reset are [113, 125]. Total target recomputation 6367 positions; held calls 14; other requests return 7122 tokens in the same gap. First boost at step 609 selects memory-train-article-0003640 and evicts ['memory-train-article-0003345']; its 10-call epoch returns 7 new outputs. Quantum exhaustion without new output: 0. Gap engine calls 4.772722s, outside-engine host intervals 0.094889s; nested scheduler 0.202754s.

Both max-gap step paths equal: True.
The measured residual is an output gap split by partial recompute service: selected-call idleness can reset while output absence continues. The successful quantum serves a different request; it does not establish a bound for every request. Matched request costs and timing may change across repeats; full engine/compiler/GPU cause and a better policy remain untested.
Next smallest step: check whether the existing funded-resume proposal preserves this interrupted recovery through its first new output; benefits remain unmeasured until an independently executed comparison.

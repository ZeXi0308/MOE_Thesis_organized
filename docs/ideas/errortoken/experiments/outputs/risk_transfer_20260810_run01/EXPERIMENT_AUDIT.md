# ErrorToken Risk-Transfer CPU Pilot Audit

**Overall verdict:** `WARN`  
**Evidence status:** `partially_valid`  
**Acceptance:** `provisional_same_family`  
**Mechanical experiment verdict:** `WEAKEN_STATIC_KEYED_RISK_TRANSFER` is correct.

## Independent recomputation

- Natural B-arm `M>1`: 904 calls / 8,116 rows.
- Eligible power-of-two grid: 240 calls / 1,264 rows, only 15.57% of natural `M>1` rows.
- Matched calibration keys: 235 calls / 1,232 rows; conditional matched coverage 97.47%.
- Unknown: 5 calls / 32 rows, including 30 mismatch rows.
- Matched row labels: 1,123 mismatch / 109 exact.
- Row-level AUC:
  - `(layer, expert, M)`: `0.538943851250337`
  - `(layer, M)`: `0.5204154991136127`
  - `M-only`: `0.5313870938753502`
  - keyed gain over `M-only`: `0.007556757374986733`
- At the frozen primary threshold `0.25`, no call is admitted; mismatch exposure and launch-reduction proxy are both zero.

The frozen weakening rule is therefore mechanically satisfied. Input hashes and the full reported policy curve independently match the bound artifacts; 4/4 unit tests pass.

## Material limitations

- This is a post-aggregate-unblinding retrospective analysis, not a preregistered confirmation.
- Rows in one call share one score and are not independent. All 235 matched calls contain at least one mismatch, so a call-level safe/unsafe AUC is undefined.
- The primary threshold is a degenerate no-op because the minimum observed calibration risk is 0.5.
- Only 191/235 matched calls satisfy the original SemanticFence pack/document support rule; restricting to them reduces keyed AUC to `0.5229225016037927`.
- `launch_count_proxy` is not measured CUDA launches, latency, throughput, or cost.
- The test suite covers pure calculations but not the complete `build_analysis` integration path.

## Claim boundary

Allowed: on this one retrospective eligible-grid analysis, static `(layer, expert, M)` calibration risk ranks fresh raw-BF16 mismatch close to random, and the frozen rule returns `WEAKEN`.

Not allowed: generalizing to other models/stacks/workloads; claiming dynamic ErrorToken or every possible selector is disproved; or claiming any latency, throughput, quality, serving, EP, or production effect.

## Next step

Do not tune this table or threshold. If the broader direction continues, freeze a genuinely new held-out run before observing labels, use call-level zero-mismatch safety with call-clustered uncertainty, and separately execute/measure the actual GPU pack-splitting action.

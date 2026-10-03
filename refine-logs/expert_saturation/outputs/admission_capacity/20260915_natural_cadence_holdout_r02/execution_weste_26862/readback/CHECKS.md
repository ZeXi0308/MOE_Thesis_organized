# r02: one targeted qualifier repair

**CPU CHECK PASS / GPU UNRUN.** Reuse r01's input provenance/identity, source-order independence, six-cell order, resource invariants, sparse measurement and3%/5% budget checks. No tokenizer replay, G qualification or broad test matrix was rerun.

The sole runtime diff is `config['requests'] == 64`→`==128` in `pkg/safe_static.py`. The actual r01 nested error proves this stale bound blocked measurement; real GPU/host allocation and full-history facts were already recorded and matched. The exact r01 package and its zero-measurement failure remain immutable.

`check_request_bound.py` verifies the entire qualifier source equals accepted r01 with that one literal replacement, then calls the real qualification function with a narrowly defined CPU engine/layout substitute.128 qualifies and64/127/129 are rejected by the original upper-bounds error. This checks the migrated guard, not native GPU serving. All native resource checks remain present and unchanged; actual allocations must still be verified at r02 startup.

From repository root:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r02/check_request_bound.py
```

The existing broader preparation-check entry is retained for reproducibility with the single changed source excluded from byte-identical G expectations. Its old input/EOS/budget checks are reused and not repeated. README changes only the new identity and the retained failure explanation; inputs/runner/controller/analyzer/qualification-reuse receipt are identical to r01. No CURRENT or remote files were edited during preparation; no r02 upload, initialization, queue or execution occurred.

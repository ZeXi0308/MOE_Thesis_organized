# Bounded preparation checks

PASS: original D input and warmup bytes, EOS capture and shared selector/native scheduling patch/recovery contract remain identical. The actual staged begin/hold/limiter closures are unchanged; only the scope-specific installation and diagnostic range observation differ.

The one CPU scope check executes the pinned native `_calc_num_offloadable_tokens` and `storable_chunks` bodies against synthetic metadata. It accepts a native-full block that the selected-prefix checker rejects; rejects double-counting scheduled tokens; retains the native known-finished-request range; rejects source/key mismatch; and verifies actual install/uninstall closures leave the original native calculation method untouched forfull and restore it forselected. Syntax/shell parsing also passed. No EOS rerun, GPU, network, parameter scan or broad audit.

New/added executable lines: 263, below500. Existing generic natural analysis is reused unchanged; scope evidence does not change store jobs or policy actions.

Reproduce from repository root:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/check_scope.py
```

GPU UNRUN. Root must accept the archive and confirm prior whole-host release; runtime source hashes, GPU occupancy and actual4096usable+1null GPU/16GiBhost allocation remain launch gates. One native diagnostic must establish real full-range accepted/completed stores, loads, flush and final-drain behavior. No action or pending-state incompatibility is retained; no common policy guard is relaxed.

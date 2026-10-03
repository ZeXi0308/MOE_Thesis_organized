Final verdict: **PASS for Gate 0-A producer-code qualification**. Safe to retain: **YES**. No concurrent mutation at final hash check.

The new formal-workload binding correctly requires:

- Canonical model-specific manifest path.
- Runtime bytes matching the preregistered SHA-256.
- Committed `HEAD` bytes matching that same SHA-256.
- Invocation before any formal model execution.
- Tests covering valid binding, noncanonical path, runtime-hash mismatch, and committed-hash mismatch.

Evidence: targeted **7/7 PASS**; full suite **69/69 PASS**.

Final reviewed hashes:

- Producer: `f37899f24b49fc3cc46bd5105b5a977502ca171561f37e319c3fa611920d4fb7`
- Test: `9f7e9d0d94a2aa80bdcf7051aa39abb09d79461ab308b9ebe79f1fd08f2ff36e`
- Preregistration: `8310589e6a8f1d7a091f73ca1fd4fcd9db6a951f737ba46f2131383d3d2c51a9`
- Experiment card: `aa0b73a27c54b22b9f8e3bb4d3888b45d2077da9d622f014165e8a0efa57cba0`

Formal CUDA execution authorized: **NO**. The preregistration still has `formal_execution_authorized: false`, unresolved workload hashes/dataset/arrival trace, and the CUDA blocker.

Gate-weight/full-router-logit parity remains a separate exact-output/full-Gate-0 blocker; it does not invalidate this Gate 0-A producer-code PASS. No files were edited.

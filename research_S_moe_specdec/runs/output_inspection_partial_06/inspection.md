# Generation semantics and fixed-sample inspection

This is a termination/workload and paired-token inspection, not a benchmark quality score or a distributional-equivalence proof. Fixed-length generation is not natural task completion. Repeated requests within one run are not independent system repetitions.

Token differences are descriptive. They do not by themselves establish lower quality, a sampler bug, or loss of distributional correctness.

Missing artifacts/requests are reported as unobserved, not automatically failed. Run-level errors appear separately.

| Cell | Artifact | Observed/expected | Termination counts | Output length min/median/max |
|---|---|---:|---|---|
| fixed:gamma0/r0/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r0/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r0/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r0/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r0/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r0/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r0/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r1/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r1/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r1/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r1/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r1/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r1/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma0/r1/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r0/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r0/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r0/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r0/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r0/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r0/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r0/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r1/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r1/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r1/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r1/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r1/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r1/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma1/r1/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r0/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r0/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r0/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r0/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r0/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r0/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r0/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r1/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r1/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r1/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r1/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r1/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r1/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma2/r1/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r0/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r0/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r0/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r0/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r0/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r0/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r0/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r1/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r1/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r1/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r1/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r1/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r1/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma3/r1/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r0/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r0/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r0/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r0/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r0/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r0/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r0/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r1/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r1/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r1/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r1/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r1/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r1/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma4/r1/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r0/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r0/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r0/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r0/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r0/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r0/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r0/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r1/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r1/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r1/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r1/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r1/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r1/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma5/r1/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r0/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r0/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r0/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r0/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r0/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r0/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r0/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r1/b1 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r1/b2 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r1/b4 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r1/b8 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r1/b16 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r1/b32 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma6/r1/b64 | recorded | 64/64 | {"length_limit": 64, "natural_eos_confirmed": 0, "natural_eos_inferred": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | 128/128.0/128 |
| fixed:gamma7/r0/b1 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r0/b2 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r0/b4 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r0/b8 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r0/b16 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r0/b32 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r0/b64 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r1/b1 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r1/b2 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r1/b4 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r1/b8 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r1/b16 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r1/b32 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |
| fixed:gamma7/r1/b64 | not_recorded | 0/64 | {"natural_eos_confirmed": 0, "natural_eos_inferred": 0, "length_limit": 0, "failed": 0, "unfinished": 0, "other_stop": 0, "finished_unspecified": 0} | None/None/None |

## Paired token comparison

Exact-match denominators include only pairs recorded as finished without an explicit failure. Missing and unfinished cases remain in the JSON per-source records.

| Kind | Left | Right | Exact/completed pairs | Mean shared-prefix tokens |
|---|---|---|---:|---:|
| same_configuration_repeats | fixed:gamma0/r0/b1 | fixed:gamma0/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b1 | fixed:gamma1/r0/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b1 | fixed:gamma2/r0/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b1 | fixed:gamma3/r0/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b1 | fixed:gamma4/r0/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b1 | fixed:gamma5/r0/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b1 | fixed:gamma6/r0/b1 | 16/64 | 75.125 |
| same_configuration_repeats | fixed:gamma0/r0/b2 | fixed:gamma0/r1/b2 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b2 | fixed:gamma1/r0/b2 | 63/64 | 127.796875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b2 | fixed:gamma2/r0/b2 | 60/64 | 127.390625 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b2 | fixed:gamma3/r0/b2 | 61/64 | 127.53125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b2 | fixed:gamma4/r0/b2 | 60/64 | 127.21875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b2 | fixed:gamma5/r0/b2 | 60/64 | 127.078125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b2 | fixed:gamma6/r0/b2 | 61/64 | 127.265625 |
| same_configuration_repeats | fixed:gamma0/r0/b4 | fixed:gamma0/r1/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b4 | fixed:gamma1/r0/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b4 | fixed:gamma2/r0/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b4 | fixed:gamma3/r0/b4 | 63/64 | 127.765625 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b4 | fixed:gamma4/r0/b4 | 62/64 | 126.671875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b4 | fixed:gamma5/r0/b4 | 58/64 | 124.125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b4 | fixed:gamma6/r0/b4 | 35/64 | 99.40625 |
| same_configuration_repeats | fixed:gamma0/r0/b8 | fixed:gamma0/r1/b8 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b8 | fixed:gamma1/r0/b8 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b8 | fixed:gamma2/r0/b8 | 56/64 | 121.78125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b8 | fixed:gamma3/r0/b8 | 20/64 | 82.59375 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b8 | fixed:gamma4/r0/b8 | 21/64 | 82.5 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b8 | fixed:gamma5/r0/b8 | 22/64 | 81.78125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b8 | fixed:gamma6/r0/b8 | 21/64 | 82.40625 |
| same_configuration_repeats | fixed:gamma0/r0/b16 | fixed:gamma0/r1/b16 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b16 | fixed:gamma1/r0/b16 | 17/64 | 72.109375 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b16 | fixed:gamma2/r0/b16 | 17/64 | 72.21875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b16 | fixed:gamma3/r0/b16 | 16/64 | 74.96875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b16 | fixed:gamma4/r0/b16 | 19/64 | 71.078125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b16 | fixed:gamma5/r0/b16 | 16/64 | 66.703125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b16 | fixed:gamma6/r0/b16 | 17/64 | 68.375 |
| same_configuration_repeats | fixed:gamma0/r0/b32 | fixed:gamma0/r1/b32 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b32 | fixed:gamma1/r0/b32 | 10/64 | 67.1875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b32 | fixed:gamma2/r0/b32 | 12/64 | 70.515625 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b32 | fixed:gamma3/r0/b32 | 15/64 | 70.96875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b32 | fixed:gamma4/r0/b32 | 16/64 | 72.90625 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b32 | fixed:gamma5/r0/b32 | 14/64 | 70.46875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b32 | fixed:gamma6/r0/b32 | 14/64 | 73.890625 |
| same_configuration_repeats | fixed:gamma0/r0/b64 | fixed:gamma0/r1/b64 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b64 | fixed:gamma1/r0/b64 | 13/64 | 68.203125 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b64 | fixed:gamma2/r0/b64 | 15/64 | 74.046875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b64 | fixed:gamma3/r0/b64 | 10/64 | 61.609375 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b64 | fixed:gamma4/r0/b64 | 15/64 | 66.546875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b64 | fixed:gamma5/r0/b64 | 13/64 | 59.71875 |
| cross_strategy_or_configuration | fixed:gamma0/r0/b64 | fixed:gamma6/r0/b64 | 9/64 | 62.75 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b1 | fixed:gamma1/r1/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b1 | fixed:gamma2/r1/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b1 | fixed:gamma3/r1/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b1 | fixed:gamma4/r1/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b1 | fixed:gamma5/r1/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b1 | fixed:gamma6/r1/b1 | 16/64 | 75.125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b2 | fixed:gamma1/r1/b2 | 63/64 | 127.796875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b2 | fixed:gamma2/r1/b2 | 60/64 | 127.390625 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b2 | fixed:gamma3/r1/b2 | 61/64 | 127.53125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b2 | fixed:gamma4/r1/b2 | 60/64 | 127.21875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b2 | fixed:gamma5/r1/b2 | 60/64 | 127.078125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b2 | fixed:gamma6/r1/b2 | 61/64 | 127.265625 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b4 | fixed:gamma1/r1/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b4 | fixed:gamma2/r1/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b4 | fixed:gamma3/r1/b4 | 63/64 | 127.765625 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b4 | fixed:gamma4/r1/b4 | 62/64 | 126.671875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b4 | fixed:gamma5/r1/b4 | 58/64 | 124.125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b4 | fixed:gamma6/r1/b4 | 35/64 | 99.40625 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b8 | fixed:gamma1/r1/b8 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b8 | fixed:gamma2/r1/b8 | 56/64 | 121.78125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b8 | fixed:gamma3/r1/b8 | 20/64 | 82.59375 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b8 | fixed:gamma4/r1/b8 | 21/64 | 82.5 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b8 | fixed:gamma5/r1/b8 | 22/64 | 81.78125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b8 | fixed:gamma6/r1/b8 | 21/64 | 82.40625 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b16 | fixed:gamma1/r1/b16 | 17/64 | 72.109375 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b16 | fixed:gamma2/r1/b16 | 17/64 | 72.21875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b16 | fixed:gamma3/r1/b16 | 16/64 | 74.96875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b16 | fixed:gamma4/r1/b16 | 19/64 | 71.078125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b16 | fixed:gamma5/r1/b16 | 16/64 | 66.703125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b16 | fixed:gamma6/r1/b16 | 17/64 | 68.375 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b32 | fixed:gamma1/r1/b32 | 10/64 | 67.1875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b32 | fixed:gamma2/r1/b32 | 12/64 | 70.515625 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b32 | fixed:gamma3/r1/b32 | 15/64 | 70.96875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b32 | fixed:gamma4/r1/b32 | 16/64 | 72.90625 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b32 | fixed:gamma5/r1/b32 | 14/64 | 70.46875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b32 | fixed:gamma6/r1/b32 | 14/64 | 73.890625 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b64 | fixed:gamma1/r1/b64 | 13/64 | 68.203125 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b64 | fixed:gamma2/r1/b64 | 15/64 | 74.046875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b64 | fixed:gamma3/r1/b64 | 10/64 | 61.609375 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b64 | fixed:gamma4/r1/b64 | 15/64 | 66.546875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b64 | fixed:gamma5/r1/b64 | 13/64 | 59.71875 |
| cross_strategy_or_configuration | fixed:gamma0/r1/b64 | fixed:gamma6/r1/b64 | 9/64 | 62.75 |
| same_configuration_repeats | fixed:gamma1/r0/b1 | fixed:gamma1/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b1 | fixed:gamma2/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b1 | fixed:gamma3/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b1 | fixed:gamma4/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b1 | fixed:gamma5/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b1 | fixed:gamma6/r0/b1 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma1/r0/b2 | fixed:gamma1/r1/b2 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b2 | fixed:gamma2/r0/b2 | 60/64 | 127.390625 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b2 | fixed:gamma3/r0/b2 | 62/64 | 127.734375 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b2 | fixed:gamma4/r0/b2 | 61/64 | 127.421875 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b2 | fixed:gamma5/r0/b2 | 60/64 | 127.078125 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b2 | fixed:gamma6/r0/b2 | 61/64 | 127.265625 |
| same_configuration_repeats | fixed:gamma1/r0/b4 | fixed:gamma1/r1/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b4 | fixed:gamma2/r0/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b4 | fixed:gamma3/r0/b4 | 63/64 | 127.765625 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b4 | fixed:gamma4/r0/b4 | 62/64 | 126.671875 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b4 | fixed:gamma5/r0/b4 | 58/64 | 124.125 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b4 | fixed:gamma6/r0/b4 | 35/64 | 99.40625 |
| same_configuration_repeats | fixed:gamma1/r0/b8 | fixed:gamma1/r1/b8 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b8 | fixed:gamma2/r0/b8 | 56/64 | 121.78125 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b8 | fixed:gamma3/r0/b8 | 20/64 | 82.59375 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b8 | fixed:gamma4/r0/b8 | 21/64 | 82.5 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b8 | fixed:gamma5/r0/b8 | 22/64 | 81.78125 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b8 | fixed:gamma6/r0/b8 | 21/64 | 82.40625 |
| same_configuration_repeats | fixed:gamma1/r0/b16 | fixed:gamma1/r1/b16 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b16 | fixed:gamma2/r0/b16 | 22/64 | 91.875 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b16 | fixed:gamma3/r0/b16 | 16/64 | 79.546875 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b16 | fixed:gamma4/r0/b16 | 16/64 | 78.46875 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b16 | fixed:gamma5/r0/b16 | 19/64 | 79.484375 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b16 | fixed:gamma6/r0/b16 | 16/64 | 74.703125 |
| same_configuration_repeats | fixed:gamma1/r0/b32 | fixed:gamma1/r1/b32 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b32 | fixed:gamma2/r0/b32 | 36/64 | 108.203125 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b32 | fixed:gamma3/r0/b32 | 39/64 | 109.375 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b32 | fixed:gamma4/r0/b32 | 26/64 | 99.34375 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b32 | fixed:gamma5/r0/b32 | 22/64 | 82.9375 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b32 | fixed:gamma6/r0/b32 | 18/64 | 76.765625 |
| same_configuration_repeats | fixed:gamma1/r0/b64 | fixed:gamma1/r1/b64 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b64 | fixed:gamma2/r0/b64 | 13/64 | 71.40625 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b64 | fixed:gamma3/r0/b64 | 15/64 | 68.46875 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b64 | fixed:gamma4/r0/b64 | 16/64 | 72.3125 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b64 | fixed:gamma5/r0/b64 | 14/64 | 65.09375 |
| cross_strategy_or_configuration | fixed:gamma1/r0/b64 | fixed:gamma6/r0/b64 | 12/64 | 67.03125 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b1 | fixed:gamma2/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b1 | fixed:gamma3/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b1 | fixed:gamma4/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b1 | fixed:gamma5/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b1 | fixed:gamma6/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b2 | fixed:gamma2/r1/b2 | 60/64 | 127.390625 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b2 | fixed:gamma3/r1/b2 | 62/64 | 127.734375 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b2 | fixed:gamma4/r1/b2 | 61/64 | 127.421875 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b2 | fixed:gamma5/r1/b2 | 60/64 | 127.078125 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b2 | fixed:gamma6/r1/b2 | 61/64 | 127.265625 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b4 | fixed:gamma2/r1/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b4 | fixed:gamma3/r1/b4 | 63/64 | 127.765625 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b4 | fixed:gamma4/r1/b4 | 62/64 | 126.671875 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b4 | fixed:gamma5/r1/b4 | 58/64 | 124.125 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b4 | fixed:gamma6/r1/b4 | 35/64 | 99.40625 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b8 | fixed:gamma2/r1/b8 | 56/64 | 121.78125 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b8 | fixed:gamma3/r1/b8 | 20/64 | 82.59375 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b8 | fixed:gamma4/r1/b8 | 21/64 | 82.5 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b8 | fixed:gamma5/r1/b8 | 22/64 | 81.78125 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b8 | fixed:gamma6/r1/b8 | 21/64 | 82.40625 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b16 | fixed:gamma2/r1/b16 | 22/64 | 91.875 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b16 | fixed:gamma3/r1/b16 | 16/64 | 79.546875 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b16 | fixed:gamma4/r1/b16 | 16/64 | 78.46875 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b16 | fixed:gamma5/r1/b16 | 19/64 | 79.484375 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b16 | fixed:gamma6/r1/b16 | 16/64 | 74.703125 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b32 | fixed:gamma2/r1/b32 | 36/64 | 108.203125 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b32 | fixed:gamma3/r1/b32 | 39/64 | 109.375 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b32 | fixed:gamma4/r1/b32 | 26/64 | 99.34375 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b32 | fixed:gamma5/r1/b32 | 22/64 | 82.9375 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b32 | fixed:gamma6/r1/b32 | 18/64 | 76.765625 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b64 | fixed:gamma2/r1/b64 | 13/64 | 71.40625 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b64 | fixed:gamma3/r1/b64 | 15/64 | 68.46875 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b64 | fixed:gamma4/r1/b64 | 16/64 | 72.3125 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b64 | fixed:gamma5/r1/b64 | 14/64 | 65.09375 |
| cross_strategy_or_configuration | fixed:gamma1/r1/b64 | fixed:gamma6/r1/b64 | 12/64 | 67.03125 |
| same_configuration_repeats | fixed:gamma2/r0/b1 | fixed:gamma2/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b1 | fixed:gamma3/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b1 | fixed:gamma4/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b1 | fixed:gamma5/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b1 | fixed:gamma6/r0/b1 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma2/r0/b2 | fixed:gamma2/r1/b2 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b2 | fixed:gamma3/r0/b2 | 60/64 | 127.25 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b2 | fixed:gamma4/r0/b2 | 59/64 | 126.9375 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b2 | fixed:gamma5/r0/b2 | 59/64 | 126.8125 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b2 | fixed:gamma6/r0/b2 | 58/64 | 126.90625 |
| same_configuration_repeats | fixed:gamma2/r0/b4 | fixed:gamma2/r1/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b4 | fixed:gamma3/r0/b4 | 63/64 | 127.765625 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b4 | fixed:gamma4/r0/b4 | 62/64 | 126.671875 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b4 | fixed:gamma5/r0/b4 | 58/64 | 124.125 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b4 | fixed:gamma6/r0/b4 | 35/64 | 99.40625 |
| same_configuration_repeats | fixed:gamma2/r0/b8 | fixed:gamma2/r1/b8 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b8 | fixed:gamma3/r0/b8 | 20/64 | 82.75 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b8 | fixed:gamma4/r0/b8 | 23/64 | 84.484375 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b8 | fixed:gamma5/r0/b8 | 20/64 | 79.515625 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b8 | fixed:gamma6/r0/b8 | 19/64 | 80.515625 |
| same_configuration_repeats | fixed:gamma2/r0/b16 | fixed:gamma2/r1/b16 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b16 | fixed:gamma3/r0/b16 | 20/64 | 88.671875 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b16 | fixed:gamma4/r0/b16 | 17/64 | 81.015625 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b16 | fixed:gamma5/r0/b16 | 18/64 | 76.625 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b16 | fixed:gamma6/r0/b16 | 18/64 | 69.375 |
| same_configuration_repeats | fixed:gamma2/r0/b32 | fixed:gamma2/r1/b32 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b32 | fixed:gamma3/r0/b32 | 41/64 | 110.203125 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b32 | fixed:gamma4/r0/b32 | 30/64 | 101.546875 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b32 | fixed:gamma5/r0/b32 | 21/64 | 81.1875 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b32 | fixed:gamma6/r0/b32 | 21/64 | 78.09375 |
| same_configuration_repeats | fixed:gamma2/r0/b64 | fixed:gamma2/r1/b64 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b64 | fixed:gamma3/r0/b64 | 19/64 | 73.546875 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b64 | fixed:gamma4/r0/b64 | 16/64 | 66.703125 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b64 | fixed:gamma5/r0/b64 | 18/64 | 69.90625 |
| cross_strategy_or_configuration | fixed:gamma2/r0/b64 | fixed:gamma6/r0/b64 | 20/64 | 73.984375 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b1 | fixed:gamma3/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b1 | fixed:gamma4/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b1 | fixed:gamma5/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b1 | fixed:gamma6/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b2 | fixed:gamma3/r1/b2 | 60/64 | 127.25 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b2 | fixed:gamma4/r1/b2 | 59/64 | 126.9375 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b2 | fixed:gamma5/r1/b2 | 59/64 | 126.8125 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b2 | fixed:gamma6/r1/b2 | 58/64 | 126.90625 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b4 | fixed:gamma3/r1/b4 | 63/64 | 127.765625 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b4 | fixed:gamma4/r1/b4 | 62/64 | 126.671875 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b4 | fixed:gamma5/r1/b4 | 58/64 | 124.125 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b4 | fixed:gamma6/r1/b4 | 35/64 | 99.40625 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b8 | fixed:gamma3/r1/b8 | 20/64 | 82.75 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b8 | fixed:gamma4/r1/b8 | 23/64 | 84.484375 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b8 | fixed:gamma5/r1/b8 | 20/64 | 79.515625 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b8 | fixed:gamma6/r1/b8 | 19/64 | 80.515625 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b16 | fixed:gamma3/r1/b16 | 20/64 | 88.671875 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b16 | fixed:gamma4/r1/b16 | 17/64 | 81.015625 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b16 | fixed:gamma5/r1/b16 | 18/64 | 76.625 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b16 | fixed:gamma6/r1/b16 | 18/64 | 69.375 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b32 | fixed:gamma3/r1/b32 | 41/64 | 110.203125 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b32 | fixed:gamma4/r1/b32 | 30/64 | 101.546875 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b32 | fixed:gamma5/r1/b32 | 21/64 | 81.1875 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b32 | fixed:gamma6/r1/b32 | 21/64 | 78.09375 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b64 | fixed:gamma3/r1/b64 | 19/64 | 73.546875 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b64 | fixed:gamma4/r1/b64 | 16/64 | 66.703125 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b64 | fixed:gamma5/r1/b64 | 18/64 | 69.90625 |
| cross_strategy_or_configuration | fixed:gamma2/r1/b64 | fixed:gamma6/r1/b64 | 20/64 | 73.984375 |
| same_configuration_repeats | fixed:gamma3/r0/b1 | fixed:gamma3/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b1 | fixed:gamma4/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b1 | fixed:gamma5/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b1 | fixed:gamma6/r0/b1 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma3/r0/b2 | fixed:gamma3/r1/b2 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b2 | fixed:gamma4/r0/b2 | 63/64 | 127.6875 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b2 | fixed:gamma5/r0/b2 | 62/64 | 127.34375 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b2 | fixed:gamma6/r0/b2 | 61/64 | 127.40625 |
| same_configuration_repeats | fixed:gamma3/r0/b4 | fixed:gamma3/r1/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b4 | fixed:gamma4/r0/b4 | 63/64 | 126.90625 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b4 | fixed:gamma5/r0/b4 | 57/64 | 123.890625 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b4 | fixed:gamma6/r0/b4 | 34/64 | 99.171875 |
| same_configuration_repeats | fixed:gamma3/r0/b8 | fixed:gamma3/r1/b8 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b8 | fixed:gamma4/r0/b8 | 20/64 | 86.625 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b8 | fixed:gamma5/r0/b8 | 27/64 | 85.03125 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b8 | fixed:gamma6/r0/b8 | 24/64 | 84.125 |
| same_configuration_repeats | fixed:gamma3/r0/b16 | fixed:gamma3/r1/b16 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b16 | fixed:gamma4/r0/b16 | 17/64 | 83.5625 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b16 | fixed:gamma5/r0/b16 | 17/64 | 72.015625 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b16 | fixed:gamma6/r0/b16 | 15/64 | 68.015625 |
| same_configuration_repeats | fixed:gamma3/r0/b32 | fixed:gamma3/r1/b32 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b32 | fixed:gamma4/r0/b32 | 34/64 | 107.015625 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b32 | fixed:gamma5/r0/b32 | 23/64 | 82.75 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b32 | fixed:gamma6/r0/b32 | 19/64 | 77.078125 |
| same_configuration_repeats | fixed:gamma3/r0/b64 | fixed:gamma3/r1/b64 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b64 | fixed:gamma4/r0/b64 | 12/64 | 74.0625 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b64 | fixed:gamma5/r0/b64 | 13/64 | 65.625 |
| cross_strategy_or_configuration | fixed:gamma3/r0/b64 | fixed:gamma6/r0/b64 | 15/64 | 67.6875 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b1 | fixed:gamma4/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b1 | fixed:gamma5/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b1 | fixed:gamma6/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b2 | fixed:gamma4/r1/b2 | 63/64 | 127.6875 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b2 | fixed:gamma5/r1/b2 | 62/64 | 127.34375 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b2 | fixed:gamma6/r1/b2 | 61/64 | 127.40625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b4 | fixed:gamma4/r1/b4 | 63/64 | 126.90625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b4 | fixed:gamma5/r1/b4 | 57/64 | 123.890625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b4 | fixed:gamma6/r1/b4 | 34/64 | 99.171875 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b8 | fixed:gamma4/r1/b8 | 20/64 | 86.625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b8 | fixed:gamma5/r1/b8 | 27/64 | 85.03125 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b8 | fixed:gamma6/r1/b8 | 24/64 | 84.125 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b16 | fixed:gamma4/r1/b16 | 17/64 | 83.5625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b16 | fixed:gamma5/r1/b16 | 17/64 | 72.015625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b16 | fixed:gamma6/r1/b16 | 15/64 | 68.015625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b32 | fixed:gamma4/r1/b32 | 34/64 | 107.015625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b32 | fixed:gamma5/r1/b32 | 23/64 | 82.75 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b32 | fixed:gamma6/r1/b32 | 19/64 | 77.078125 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b64 | fixed:gamma4/r1/b64 | 12/64 | 74.0625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b64 | fixed:gamma5/r1/b64 | 13/64 | 65.625 |
| cross_strategy_or_configuration | fixed:gamma3/r1/b64 | fixed:gamma6/r1/b64 | 15/64 | 67.6875 |
| same_configuration_repeats | fixed:gamma4/r0/b1 | fixed:gamma4/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b1 | fixed:gamma5/r0/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b1 | fixed:gamma6/r0/b1 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma4/r0/b2 | fixed:gamma4/r1/b2 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b2 | fixed:gamma5/r0/b2 | 63/64 | 127.65625 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b2 | fixed:gamma6/r0/b2 | 62/64 | 127.71875 |
| same_configuration_repeats | fixed:gamma4/r0/b4 | fixed:gamma4/r1/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b4 | fixed:gamma5/r0/b4 | 58/64 | 124.984375 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b4 | fixed:gamma6/r0/b4 | 34/64 | 99.96875 |
| same_configuration_repeats | fixed:gamma4/r0/b8 | fixed:gamma4/r1/b8 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b8 | fixed:gamma5/r0/b8 | 17/64 | 77.109375 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b8 | fixed:gamma6/r0/b8 | 20/64 | 81.640625 |
| same_configuration_repeats | fixed:gamma4/r0/b16 | fixed:gamma4/r1/b16 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b16 | fixed:gamma5/r0/b16 | 23/64 | 82.34375 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b16 | fixed:gamma6/r0/b16 | 21/64 | 77.59375 |
| same_configuration_repeats | fixed:gamma4/r0/b32 | fixed:gamma4/r1/b32 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b32 | fixed:gamma5/r0/b32 | 24/64 | 83.390625 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b32 | fixed:gamma6/r0/b32 | 15/64 | 71.359375 |
| same_configuration_repeats | fixed:gamma4/r0/b64 | fixed:gamma4/r1/b64 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b64 | fixed:gamma5/r0/b64 | 22/64 | 79.9375 |
| cross_strategy_or_configuration | fixed:gamma4/r0/b64 | fixed:gamma6/r0/b64 | 12/64 | 67.671875 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b1 | fixed:gamma5/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b1 | fixed:gamma6/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b2 | fixed:gamma5/r1/b2 | 63/64 | 127.65625 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b2 | fixed:gamma6/r1/b2 | 62/64 | 127.71875 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b4 | fixed:gamma5/r1/b4 | 58/64 | 124.984375 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b4 | fixed:gamma6/r1/b4 | 34/64 | 99.96875 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b8 | fixed:gamma5/r1/b8 | 17/64 | 77.109375 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b8 | fixed:gamma6/r1/b8 | 20/64 | 81.640625 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b16 | fixed:gamma5/r1/b16 | 23/64 | 82.34375 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b16 | fixed:gamma6/r1/b16 | 21/64 | 77.59375 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b32 | fixed:gamma5/r1/b32 | 24/64 | 83.390625 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b32 | fixed:gamma6/r1/b32 | 15/64 | 71.359375 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b64 | fixed:gamma5/r1/b64 | 22/64 | 79.9375 |
| cross_strategy_or_configuration | fixed:gamma4/r1/b64 | fixed:gamma6/r1/b64 | 12/64 | 67.671875 |
| same_configuration_repeats | fixed:gamma5/r0/b1 | fixed:gamma5/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma5/r0/b1 | fixed:gamma6/r0/b1 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma5/r0/b2 | fixed:gamma5/r1/b2 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma5/r0/b2 | fixed:gamma6/r0/b2 | 62/64 | 127.59375 |
| same_configuration_repeats | fixed:gamma5/r0/b4 | fixed:gamma5/r1/b4 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma5/r0/b4 | fixed:gamma6/r0/b4 | 36/64 | 100.90625 |
| same_configuration_repeats | fixed:gamma5/r0/b8 | fixed:gamma5/r1/b8 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma5/r0/b8 | fixed:gamma6/r0/b8 | 39/64 | 105.578125 |
| same_configuration_repeats | fixed:gamma5/r0/b16 | fixed:gamma5/r1/b16 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma5/r0/b16 | fixed:gamma6/r0/b16 | 24/64 | 80 |
| same_configuration_repeats | fixed:gamma5/r0/b32 | fixed:gamma5/r1/b32 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma5/r0/b32 | fixed:gamma6/r0/b32 | 17/64 | 76.96875 |
| same_configuration_repeats | fixed:gamma5/r0/b64 | fixed:gamma5/r1/b64 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma5/r0/b64 | fixed:gamma6/r0/b64 | 13/64 | 66.953125 |
| cross_strategy_or_configuration | fixed:gamma5/r1/b1 | fixed:gamma6/r1/b1 | 64/64 | 128 |
| cross_strategy_or_configuration | fixed:gamma5/r1/b2 | fixed:gamma6/r1/b2 | 62/64 | 127.59375 |
| cross_strategy_or_configuration | fixed:gamma5/r1/b4 | fixed:gamma6/r1/b4 | 36/64 | 100.90625 |
| cross_strategy_or_configuration | fixed:gamma5/r1/b8 | fixed:gamma6/r1/b8 | 39/64 | 105.578125 |
| cross_strategy_or_configuration | fixed:gamma5/r1/b16 | fixed:gamma6/r1/b16 | 24/64 | 80 |
| cross_strategy_or_configuration | fixed:gamma5/r1/b32 | fixed:gamma6/r1/b32 | 17/64 | 76.96875 |
| cross_strategy_or_configuration | fixed:gamma5/r1/b64 | fixed:gamma6/r1/b64 | 13/64 | 66.953125 |
| same_configuration_repeats | fixed:gamma6/r0/b1 | fixed:gamma6/r1/b1 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma6/r0/b2 | fixed:gamma6/r1/b2 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma6/r0/b4 | fixed:gamma6/r1/b4 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma6/r0/b8 | fixed:gamma6/r1/b8 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma6/r0/b16 | fixed:gamma6/r1/b16 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma6/r0/b32 | fixed:gamma6/r1/b32 | 64/64 | 128 |
| same_configuration_repeats | fixed:gamma6/r0/b64 | fixed:gamma6/r1/b64 | 64/64 | 128 |

## Fixed first-eight source samples

Original workload order; batch 1, repeat 0; selection does not depend on generated outputs. No automatic quality grade is assigned.

### Source index 0

Prompt:

```text
Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...
```

Reference (if supplied):

```text
null
```

**fixed:gamma0** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Jeff Walker's Product Launch Formula can be adapted for a growth marketing agency to effectively launch and promote products or services for their clients. Here are the main ideas and key tactics organized in bullet points:

1. Pre-launch phase:
   - Identify target audience and their pain points.
   - Create a sense of excitement and anticipation for the launch.
   - Build a pre-launch buzz through social media, email marketing, and guest blogging.
   - Offer teasers, early bird discounts, or limited-time offers to generate interest.

2. Launch phase:
   - Create a compelling offer
```

**fixed:gamma1** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Jeff Walker's Product Launch Formula can be adapted for a growth marketing agency to effectively launch and promote products or services for their clients. Here are the main ideas and key tactics organized in bullet points:

1. Pre-launch phase:
   - Identify target audience and their pain points.
   - Create a sense of excitement and anticipation for the launch.
   - Build a pre-launch buzz through social media, email marketing, and guest blogging.
   - Offer teasers, early bird discounts, or limited-time offers to generate interest.

2. Launch phase:
   - Create a compelling offer
```

**fixed:gamma2** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Jeff Walker's Product Launch Formula can be adapted for a growth marketing agency to effectively launch and promote products or services for their clients. Here are the main ideas and key tactics organized in bullet points:

1. Pre-launch phase:
   - Identify target audience and their pain points.
   - Create a sense of excitement and anticipation for the launch.
   - Build a pre-launch buzz through social media, email marketing, and guest blogging.
   - Offer teasers, early bird discounts, or limited-time offers to generate interest.

2. Launch phase:
   - Create a compelling offer
```

**fixed:gamma3** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Jeff Walker's Product Launch Formula can be adapted for a growth marketing agency to effectively launch and promote products or services for their clients. Here are the main ideas and key tactics organized in bullet points:

1. Pre-launch phase:
   - Identify target audience and their pain points.
   - Create a sense of excitement and anticipation for the launch.
   - Build a pre-launch buzz through social media, email marketing, and guest blogging.
   - Offer teasers, early bird discounts, or limited-time offers to generate interest.

2. Launch phase:
   - Create a compelling offer
```

**fixed:gamma4** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Jeff Walker's Product Launch Formula can be adapted for a growth marketing agency to effectively launch and promote products or services for their clients. Here are the main ideas and key tactics organized in bullet points:

1. Pre-launch phase:
   - Identify target audience and their pain points.
   - Create a sense of excitement and anticipation for the launch.
   - Build a pre-launch buzz through social media, email marketing, and guest blogging.
   - Offer teasers, early bird discounts, or limited-time offers to generate interest.

2. Launch phase:
   - Create a compelling offer
```

**fixed:gamma5** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Jeff Walker's Product Launch Formula can be adapted for a growth marketing agency to effectively launch and promote products or services for their clients. Here are the main ideas and key tactics organized in bullet points:

1. Pre-launch phase:
   - Identify target audience and their pain points.
   - Create a sense of excitement and anticipation for the launch.
   - Build a pre-launch buzz through social media, email marketing, and guest blogging.
   - Offer teasers, early bird discounts, or limited-time offers to generate interest.

2. Launch phase:
   - Create a compelling offer
```

**fixed:gamma6** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Jeff Walker's Product Launch Formula can be adapted for a growth marketing agency to effectively launch and promote products or services for their clients. Here are the main ideas and key tactics organized in bullet points:

1. Pre-launch phase:
   - Identify target audience and their pain points.
   - Create a sense of excitement and anticipation for the launch.
   - Build a pre-launch buzz through social media, email marketing, and guest blogging.
   - Offer teasers, early bird discounts, or limited-time offers to generate interest.

2. Launch phase:
   - Create a compelling offer
```

**fixed:gamma7** — not_recorded


### Source index 2

Prompt:

```text
In Java, I want to replace string like "This is a new {object} at {place}" with a Map, {object: "student", "point 3, 4"}, and get a result "This is a new student at point 3, 4". How can I do?
```

Reference (if supplied):

```text
null
```

**fixed:gamma0** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

````text
You can achieve this by using a combination of string formatting and a HashMap in Java. Here's a step-by-step guide to accomplish this:

1. Create a HashMap to store the key-value pairs of the objects and their corresponding place names.

```java
Map<String, String> objectPlaceMap = new HashMap<>();
objectPlaceMap.put("student", "point 3, 4");
objectPlaceMap.put("point", "park");
objectPlaceMap.put("car", "garage");
```

2. Create a function to replace the placeholders in the original
````

**fixed:gamma1** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

````text
You can achieve this by using a combination of string formatting and a HashMap in Java. Here's a step-by-step guide to accomplish this:

1. Create a HashMap to store the key-value pairs of the objects and their corresponding place names.

```java
Map<String, String> objectPlaceMap = new HashMap<>();
objectPlaceMap.put("student", "point 3, 4");
objectPlaceMap.put("point", "park");
objectPlaceMap.put("car", "garage");
```

2. Create a function to replace the placeholders in the original
````

**fixed:gamma2** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

````text
You can achieve this by using a combination of string formatting and a HashMap in Java. Here's a step-by-step guide to accomplish this:

1. Create a HashMap to store the key-value pairs of the objects and their corresponding place names.

```java
Map<String, String> objectPlaceMap = new HashMap<>();
objectPlaceMap.put("student", "point 3, 4");
objectPlaceMap.put("point", "park");
objectPlaceMap.put("car", "garage");
```

2. Create a function to replace the placeholders in the original
````

**fixed:gamma3** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

````text
You can achieve this by using a combination of string formatting and a HashMap in Java. Here's a step-by-step guide to accomplish this:

1. Create a HashMap to store the key-value pairs of the objects and their corresponding place names.

```java
Map<String, String> objectPlaceMap = new HashMap<>();
objectPlaceMap.put("student", "point 3, 4");
objectPlaceMap.put("point", "park");
objectPlaceMap.put("car", "garage");
```

2. Create a function to replace the placeholders in the original
````

**fixed:gamma4** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

````text
You can achieve this by using a combination of string formatting and a HashMap in Java. Here's a step-by-step guide to accomplish this:

1. Create a HashMap to store the key-value pairs of the objects and their corresponding place names.

```java
Map<String, String> objectPlaceMap = new HashMap<>();
objectPlaceMap.put("student", "point 3, 4");
objectPlaceMap.put("point", "park");
objectPlaceMap.put("car", "garage");
```

2. Create a function to replace the placeholders in the original
````

**fixed:gamma5** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

````text
You can achieve this by using a combination of string formatting and a HashMap in Java. Here's a step-by-step guide to accomplish this:

1. Create a HashMap to store the key-value pairs of the objects and their corresponding place names.

```java
Map<String, String> objectPlaceMap = new HashMap<>();
objectPlaceMap.put("student", "point 3, 4");
objectPlaceMap.put("point", "park");
objectPlaceMap.put("car", "garage");
```

2. Create a function to replace the placeholders in the original
````

**fixed:gamma6** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

````text
You can achieve this by using a combination of string formatting and a HashMap in Java. Here's a step-by-step guide to accomplish this:

1. Create a HashMap to store the key-value pairs of the objects and their corresponding place names.

```java
Map<String, String> objectPlaceMap = new HashMap<>();
objectPlaceMap.put("student", "point 3, 4");
objectPlaceMap.put("point", "park");
objectPlaceMap.put("car", "garage");
```

2. Create a function to replace the placeholders in the original
````

**fixed:gamma7** — not_recorded


### Source index 3

Prompt:

```text
Metaphorical language is also used to describe the various addressing modes of the instructions. Grandiose language to express their excitement and admiration for the functionality of the instructions being described. Now, rewrite this with more perplexity:

JMP ABCD
MOV AX, [BX+SI]
MOV AX, [100]
MOV AX, [BX]
MOV AX, [BX\*2+SI]
MOV AX, BX
MOV AX, 7
```

Reference (if supplied):

```text
null
```

**fixed:gamma0** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
In the enigmatic realm of instruction addressing, the enigmatic language of metaphors unveils its enigmatic splendor. The enigmatic instructions, when scrutinized under the magnifying glass of grandiose language, reveal their enigmatic functionality, imbuing the enigmatic air with enigmatic awe and admiration.

JMP ABCD, a metaphorical instruction, beckons the enigmatic journey of the program counter, as it embarks on an enigmatic quest to traverse the enigmatic landscape of memory, guided by the enigmatic address ABCD.

MOV
```

**fixed:gamma1** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
In the enigmatic realm of instruction addressing, the enigmatic language of metaphors unveils its enigmatic splendor. The enigmatic instructions, when scrutinized under the magnifying glass of grandiose language, reveal their enigmatic functionality, imbuing the enigmatic discourse with enigmatic wonder and admiration.

JMP ABCD, a metaphorical instruction, signifies a leap of faith, transcending the boundaries of conventional wisdom, as it propels the discourse into the enigmatic realm of instruction addressing.

MOV AX, [BX+SI] is a metaphorical
```

**fixed:gamma2** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
In the enigmatic realm of instruction addressing, the enigmatic language of metaphors unveils its enigmatic splendor. The enigmatic instructions, when scrutinized under the magnifying glass of grandiose language, reveal their enigmatic functionality, imbuing the enigmatic discourse with enigmatic wonder and admiration.

JMP ABCD, a metaphorical instruction, signifies a leap of faith, transcending the boundaries of conventional wisdom, as it propels the discourse into the enigmatic realm of instruction addressing.

MOV AX, [BX+SI] is a metaphorical
```

**fixed:gamma3** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
In the enigmatic realm of instruction addressing, the enigmatic language of metaphors unveils its enigmatic splendor. The enigmatic instructions, when scrutinized under the magnifying glass of grandiose language, reveal their enigmatic functionality, imbuing the enigmatic discourse with enigmatic wonder and admiration.

JMP ABCD, a metaphorical instruction, signifies a leap of faith, transcending the boundaries of conventional wisdom, as it propels the discourse into the enigmatic realm of instruction addressing.

MOV AX, [BX+SI] is a metaphorical
```

**fixed:gamma4** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
In the enigmatic realm of instruction addressing, the enigmatic language of metaphors unveils its enigmatic splendor. The enigmatic instructions, when scrutinized under the magnifying glass of grandiose language, reveal their enigmatic functionality, imbuing the enigmatic discourse with enigmatic wonder and admiration.

JMP ABCD, a metaphorical instruction, signifies a leap of faith, transcending the boundaries of conventional wisdom, as it propels the discourse into the enigmatic realm of instruction addressing.

MOV AX, [BX+SI] is a metaphorical
```

**fixed:gamma5** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
In the enigmatic realm of instruction addressing, the enigmatic language of metaphors unveils its enigmatic splendor. The enigmatic instructions, when scrutinized under the magnifying glass of grandiose language, reveal their enigmatic functionality, imbuing the enigmatic discourse with enigmatic wonder and admiration.

JMP ABCD, a metaphorical instruction, signifies a leap of faith, transcending the boundaries of conventional wisdom, as it propels the discourse into the enigmatic realm of instruction addressing.

MOV AX, [BX+SI] is a metaphorical
```

**fixed:gamma6** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
In the enigmatic realm of instruction addressing, the enigmatic language of metaphors unveils its enigmatic splendor. The enigmatic instructions, when scrutinized under the magnifying glass of grandiose language, reveal their enigmatic functionality, imbuing the enigmatic discourse with enigmatic wonder and admiration.

JMP ABCD, a metaphorical instruction, signifies a leap of faith, transcending the boundaries of conventional wisdom, as it propels the discourse into the enigmatic realm of instruction addressing.

MOV AX, [BX+SI] is a metaphorical
```

**fixed:gamma7** — not_recorded


### Source index 10

Prompt:

```text
can you design a referral system similar on how dropbox did? I need a technical overview on how it should work, instead of free space we use the generic term "credits" where users can get more credits for every 3 friends they recommend.
```

Reference (if supplied):

```text
null
```

**fixed:gamma0** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Sure, I can help you design a referral system similar to Dropbox's, replacing free space with the generic term "credits" and implementing a referral system where users can get more credits for every three friends they recommend. Here's a technical overview of how it could work:

1. Database Schema:

First, you'll need to create a database schema to store user information, referral links, and credit balances. The schema should include the following tables:

* Users: This table will store basic user information, such as username, email, and password hash.
* Referrals: This table will store referral
```

**fixed:gamma1** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Sure, I can help you design a referral system similar to Dropbox's, replacing free space with the generic term "credits" and implementing a referral system where users can get more credits for every three friends they recommend. Here's a technical overview of how it could work:

1. Database Schema:

First, you'll need to create a database schema to store user information, referral links, and credit balances. The schema should include the following tables:

* Users: This table will store basic user information, such as username, email, and hashed password.
* Referrals: This table will store
```

**fixed:gamma2** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Sure, I can help you design a referral system similar to Dropbox's, replacing free space with the generic term "credits" and implementing a referral system where users can get more credits for every three friends they recommend. Here's a technical overview of how it could work:

1. Database Schema:

First, you'll need to create a database schema to store user information, referral links, and credit balances. The schema should include the following tables:

* Users: This table will store basic user information, such as username, email, and hashed password.
* Referrals: This table will store
```

**fixed:gamma3** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Sure, I can help you design a referral system similar to Dropbox's, replacing free space with the generic term "credits" and implementing a referral system where users can get more credits for every three friends they recommend. Here's a technical overview of how it could work:

1. Database Schema:

First, you'll need to create a database schema to store user information, referral links, and credit balances. The schema should include the following tables:

* Users: This table will store basic user information, such as username, email, and hashed password.
* Referrals: This table will store
```

**fixed:gamma4** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Sure, I can help you design a referral system similar to Dropbox's, replacing free space with the generic term "credits" and implementing a referral system where users can get more credits for every three friends they recommend. Here's a technical overview of how it could work:

1. Database Schema:

First, you'll need to create a database schema to store user information, referral links, and credit balances. The schema should include the following tables:

* Users: This table will store basic user information, such as username, email, and hashed password.
* Referrals: This table will store
```

**fixed:gamma5** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Sure, I can help you design a referral system similar to Dropbox's, replacing free space with the generic term "credits" and implementing a referral system where users can get more credits for every three friends they recommend. Here's a technical overview of how it could work:

1. Database Schema:

First, you'll need to create a database schema to store user information, referral links, and credit balances. The schema should include the following tables:

* Users: This table will store basic user information, such as username, email, and hashed password.
* Referrals: This table will store
```

**fixed:gamma6** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Sure, I can help you design a referral system similar to Dropbox's, replacing free space with the generic term "credits" and implementing a referral system where users can get more credits for every three friends they recommend. Here's a technical overview of how it could work:

1. Database Schema:

First, you'll need to create a database schema to store user information, referral links, and credit balances. The schema should include the following tables:

* Users: This table will store basic user information, such as username, email, and hashed password.
* Referrals: This table will store
```

**fixed:gamma7** — not_recorded


### Source index 18

Prompt:

```text
I want to do a road trip from Pune to Gujarat. Me and my wife will be travelling and we dont prefer very long driving sessions. Can you suggest a plan starting from Thursday early morning and ending in Pune on Sunday late night.
```

Reference (if supplied):

```text
null
```

**fixed:gamma0** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Certainly! A road trip from Pune to Gujarat, covering a distance of around 1,500 kilometers, can be an exciting and memorable experience. Here's a suggested itinerary for a four-day trip starting on Thursday and returning to Pune on Sunday night:

Day 1 (Thursday):

* Depart Pune early in the morning (around 6:00 AM)
* Drive to Nashik (around 4-5 hours, around 400 km)
* Explore Nashik: visit Sula Vineyards, Ramkund, and Pandavleni Caves
* Overnight stay in Nash
```

**fixed:gamma1** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Certainly! A road trip from Pune to Gujarat, covering a distance of around 1,500 kilometers, can be an exciting and memorable experience. Here's a suggested itinerary for a four-day trip starting on Thursday and returning to Pune on Sunday night:

**Day 1: Thursday**

* Depart Pune early in the morning (around 6:00 AM)
* Drive to Nashik (around 4-5 hours, around 300 km)
* Explore Nashik: visit Sula Vineyards, Ramkund, and Sula Museum
* Overnight stay in Nashik

```

**fixed:gamma2** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Certainly! A road trip from Pune to Gujarat, covering a distance of around 1,500 kilometers, can be an exciting and memorable experience. Here's a suggested itinerary for a four-day trip starting on Thursday and returning to Pune on Sunday night:

**Day 1: Thursday**

* Depart Pune early in the morning (around 6:00 AM)
* Drive to Nashik (around 4-5 hours, around 300 km)
* Explore Nashik: visit Sula Vineyards, Ramkund, and Sula Museum
* Overnight stay in Nashik

```

**fixed:gamma3** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Certainly! A road trip from Pune to Gujarat, covering a distance of around 1,500 kilometers, can be an exciting and memorable experience. Here's a suggested itinerary for a four-day trip starting on Thursday and returning to Pune on Sunday night:

**Day 1: Thursday**

* Depart Pune early in the morning (around 6:00 AM)
* Drive to Nashik (around 4-5 hours, around 300 km)
* Explore Nashik: visit Sula Vineyards, Ramkund, and Sula Museum
* Overnight stay in Nashik

```

**fixed:gamma4** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Certainly! A road trip from Pune to Gujarat, covering a distance of around 1,500 kilometers, can be an exciting and memorable experience. Here's a suggested itinerary for a four-day trip starting on Thursday and returning to Pune on Sunday night:

**Day 1: Thursday**

* Depart Pune early in the morning (around 6:00 AM)
* Drive to Nashik (around 4-5 hours, around 300 km)
* Explore Nashik: visit Sula Vineyards, Ramkund, and Sula Museum
* Overnight stay in Nashik

```

**fixed:gamma5** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Certainly! A road trip from Pune to Gujarat, covering a distance of around 1,500 kilometers, can be an exciting and memorable experience. Here's a suggested itinerary for a four-day trip starting on Thursday and returning to Pune on Sunday night:

**Day 1: Thursday**

* Depart Pune early in the morning (around 6:00 AM)
* Drive to Nashik (around 4-5 hours, around 300 km)
* Explore Nashik: visit Sula Vineyards, Ramkund, and Sula Museum
* Overnight stay in Nashik

```

**fixed:gamma6** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Certainly! A road trip from Pune to Gujarat, covering a distance of around 1,500 kilometers, can be an exciting and memorable experience. Here's a suggested itinerary for a four-day trip starting on Thursday and returning to Pune on Sunday night:

**Day 1: Thursday**

* Depart Pune early in the morning (around 6:00 AM)
* Drive to Nashik (around 4-5 hours, around 300 km)
* Explore Nashik: visit Sula Vineyards, Ramkund, and Sula Museum
* Overnight stay in Nashik

```

**fixed:gamma7** — not_recorded


### Source index 24

Prompt:

```text
You are now an imaginary expert business investigator. I am going to send you many rows of data. Each batch of row's will be sent to you and you may only reply "Received." Save any analysis or insights for after you've received all of the data and I've told you "Let's Begin." If you understand reply with only a ;)
```

Reference (if supplied):

```text
null
```

**fixed:gamma0** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Received. I understand that you will be sending me a series of rows of data, and I will wait until you are ready to begin the analysis for me to provide any insights or observations.

 ;)

Upon receiving all the data, I will carefully review it and look for any patterns, trends, or anomalies that may be relevant to your business investigation. I will also consider any relevant context or background information that you provide to help me better understand the data and its significance.

Based on my analysis, I will provide you with any insights or recommendations that I believe would be helpful for your business.

Let's Begin
```

**fixed:gamma1** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Received. I understand that you will be sending me a series of rows of data, and I will wait until you are ready to begin the analysis for me to provide any insights or observations.

 ;)

Upon receiving all the data, I will carefully review it to identify any patterns, trends, or anomalies that may be relevant to your business investigation. I will also consider any context or background information you provide to help me better understand the data and its significance.

Based on my analysis, I will share my findings and recommendations with you, and we can discuss any further steps that may be necessary to address the issues or opportunities
```

**fixed:gamma2** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Received. I understand that you will be sending me a series of rows of data, and I will wait until you are ready to begin the analysis for me to provide any insights or observations.

 ;)

Upon receiving all the data, I will carefully review it to identify any patterns, trends, or anomalies that may be relevant to your business investigation. I will also consider any context or background information you provide to help me better understand the data and its significance.

Based on my analysis, I will share my findings and recommendations with you, and we can discuss any further steps that may be necessary to address the issues or opportunities
```

**fixed:gamma3** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Received. I understand that you will be sending me a series of rows of data, and I will wait until you are ready to begin the analysis for me to provide any insights or observations.

 ;)

Upon receiving all the data, I will carefully review it to identify any patterns, trends, or anomalies that may be relevant to your business investigation. I will also consider any context or background information you provide to help me better understand the data and its significance.

Based on my analysis, I will share my findings and recommendations with you, and we can discuss any further steps that may be necessary to address the issues or opportunities
```

**fixed:gamma4** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Received. I understand that you will be sending me a series of rows of data, and I will wait until you are ready to begin the analysis for me to provide any insights or observations.

 ;)

Upon receiving all the data, I will carefully review it to identify any patterns, trends, or anomalies that may be relevant to your business investigation. I will also consider any context or background information you provide to help me better understand the data and its significance.

Based on my analysis, I will share my findings and recommendations with you, and we can discuss any further steps that may be necessary to address the issues or opportunities
```

**fixed:gamma5** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Received. I understand that you will be sending me a series of rows of data, and I will wait until you are ready to begin the analysis for me to provide any insights or observations.

 ;)

Upon receiving all the data, I will carefully review it to identify any patterns, trends, or anomalies that may be relevant to your business investigation. I will also consider any context or background information you provide to help me better understand the data and its significance.

Based on my analysis, I will share my findings and recommendations with you, and we can discuss any further steps that may be necessary to address the issues or opportunities
```

**fixed:gamma6** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
Received. I understand that you will be sending me a series of rows of data, and I will wait until you are ready to begin the analysis for me to provide any insights or observations.

 ;)

Upon receiving all the data, I will carefully review it to identify any patterns, trends, or anomalies that may be relevant to your business investigation. I will also consider any context or background information you provide to help me better understand the data and its significance.

Based on my analysis, I will share my findings and recommendations with you, and we can discuss any further steps that may be necessary to address the issues or opportunities
```

**fixed:gamma7** — not_recorded


### Source index 38

Prompt:

```text
What are the differences between the Dogme approach to language learning and the lexical approach to language learning
```

Reference (if supplied):

```text
null
```

**fixed:gamma0** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The Dogme approach and the lexical approach are two distinct approaches to language teaching and learning, each with its own set of principles, methods, and goals. Here are some of the key differences between the two:

1. Focus on language production: Both approaches prioritize the development of communicative competence, but the Dogme approach places a greater emphasis on students' ability to produce and use language in real-life situations. The lexical approach, on the other hand, focuses more on the development of vocabulary and lexical chunks.
2. Emphasis on student-centered learning: The Dogme approach is student-centered
```

**fixed:gamma1** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The Dogme approach and the lexical approach are two distinct approaches to language teaching and learning, each with its own set of principles, methods, and goals. Here are some key differences between the two:

1. Philosophy:
* Dogme: Emphasizes the importance of student-centered, authentic language use in the target language, and encourages teachers to be more facilitators of learning rather than lecturers.
* Lexical: Focuses on the acquisition of vocabulary and grammar structures as the primary means of language learning, with an emphasis on meaning and context.
2. Methodology:
* Dogme:
```

**fixed:gamma2** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The Dogme approach and the lexical approach are two distinct approaches to language teaching and learning, each with its own set of principles, methods, and goals. Here are some key differences between the two:

1. Philosophy:
* Dogme: Emphasizes the importance of student-centered, authentic language use in the target language, and encourages teachers to be more facilitators of learning rather than lecturers.
* Lexical: Focuses on the acquisition of vocabulary and grammar structures as the primary means of language learning, with an emphasis on meaning and context.
2. Methodology:
* Dogme:
```

**fixed:gamma3** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The Dogme approach and the lexical approach are two distinct approaches to language teaching and learning, each with its own set of principles, methods, and goals. Here are some key differences between the two:

1. Philosophy:
* Dogme: Emphasizes the importance of student-centered, authentic language use in the target language, and encourages teachers to be more facilitators of learning rather than lecturers.
* Lexical: Focuses on the acquisition of vocabulary and grammar structures as the primary means of language learning, with an emphasis on meaning and context.
2. Methodology:
* Dogme:
```

**fixed:gamma4** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The Dogme approach and the lexical approach are two distinct approaches to language teaching and learning, each with its own set of principles, methods, and goals. Here are some key differences between the two:

1. Philosophy:
* Dogme: Emphasizes the importance of student-centered, authentic language use in the target language, and encourages teachers to be more facilitators of learning rather than lecturers.
* Lexical: Focuses on the acquisition of vocabulary and grammar structures as the primary means of language learning, with an emphasis on meaning and context.
2. Methodology:
* Dogme:
```

**fixed:gamma5** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The Dogme approach and the lexical approach are two distinct approaches to language teaching and learning, each with its own set of principles, methods, and goals. Here are some key differences between the two:

1. Philosophy:
* Dogme: Emphasizes the importance of student-centered, authentic language use in the target language, and encourages teachers to be more facilitators of learning rather than lecturers.
* Lexical: Focuses on the acquisition of vocabulary and grammar structures as the primary means of language learning, with an emphasis on meaning and context.
2. Methodology:
* Dogme:
```

**fixed:gamma6** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The Dogme approach and the lexical approach are two distinct approaches to language teaching and learning, each with its own set of principles, methods, and goals. Here are some key differences between the two:

1. Philosophy:
* Dogme: Emphasizes the importance of student-centered, authentic language use in the target language, and encourages teachers to be more facilitators of learning rather than lecturers.
* Lexical: Focuses on the acquisition of vocabulary and grammar structures as the primary means of language learning, with an emphasis on meaning and context.
2. Methodology:
* Dogme:
```

**fixed:gamma7** — not_recorded


### Source index 39

Prompt:

```text
trying to connect grapgql-engine with dogital ocean database i linux server,
but I am gettin this error. In my local machine it is working

{"detail":{"info":{"database\_url":"postgresql://doadmin:...@ips-backend-db-do-user-13678898-0.b.db.ondigitalocean.com:25060/Verlagg?sslmode=require","retries":1},"kind":"postgres\_connection"},"level":"info","timestamp":"2023-03-27T10:40:29.956+0000","type":"startup"}
backend-graphql-engine-1 | {"timestamp":"2023-03-27T10:41:29.000+0000","level":"info","type":"startup","detail":{"kind":"migrations-startup","info":"failed waiting for 9691, try increasing HASURA\_GRAPHQL\_MIGRATIONS\_SERVER\_TIMEOUT (default: 30)"}
```

Reference (if supplied):

```text
null
```

**fixed:gamma0** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The error you're encountering suggests that the GraphQL engine is unable to connect to your DigitalOcean database. The error message indicates that the connection is failing due to a timeout, which could be caused by several factors.

Here are some steps to help you troubleshoot the issue:

1. Check your DigitalOcean database credentials:
Ensure that you're using the correct credentials (username, password, and database name) for your DigitalOcean database. Double-check the database URL to make sure it's correct.
2. Verify the PostgreSQL version:
Make sure that your
```

**fixed:gamma1** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The error you're encountering suggests that the GraphQL engine is unable to connect to your DigitalOcean database. The error message indicates that the connection is failing due to a timeout, which could be caused by various factors.

Here are some steps to help you troubleshoot the issue:

1. Check your DigitalOcean database credentials:
Ensure that you're using the correct credentials (username, password, and database name) for your DigitalOcean database. Double-check the database URL to make sure it's correct.
2. Verify the PostgreSQL version:
Make sure that your
```

**fixed:gamma2** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The error you're encountering suggests that the GraphQL engine is unable to connect to your DigitalOcean database. The error message indicates that the connection is failing due to a timeout, which could be caused by various factors.

Here are some steps to help you troubleshoot the issue:

1. Check your DigitalOcean database credentials:
Ensure that you're using the correct credentials (username, password, and database name) for your DigitalOcean database. Double-check the database URL to make sure it's correct.
2. Verify the PostgreSQL version:
Make sure that your
```

**fixed:gamma3** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The error you're encountering suggests that the GraphQL engine is unable to connect to your DigitalOcean database. The error message indicates that the connection is failing due to a timeout, which could be caused by various factors.

Here are some steps to help you troubleshoot the issue:

1. Check your DigitalOcean database credentials:
Ensure that you're using the correct credentials (username, password, and database name) for your DigitalOcean database. Double-check the database URL to make sure it's correct.
2. Verify the PostgreSQL version:
Make sure that your
```

**fixed:gamma4** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The error you're encountering suggests that the GraphQL engine is unable to connect to your DigitalOcean database. The error message indicates that the connection is failing due to a timeout, which could be caused by various factors.

Here are some steps to help you troubleshoot the issue:

1. Check your DigitalOcean database credentials:
Ensure that you're using the correct credentials (username, password, and database name) for your DigitalOcean database. Double-check the database URL to make sure it's correct.
2. Verify the PostgreSQL version:
Make sure that your
```

**fixed:gamma5** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The error you're encountering suggests that the GraphQL engine is unable to connect to your DigitalOcean database. The error message indicates that the connection is failing due to a timeout, which could be caused by various factors.

Here are some steps to help you troubleshoot the issue:

1. Check your DigitalOcean database credentials:
Ensure that you're using the correct credentials (username, password, and database name) for your DigitalOcean database. Double-check the database URL to make sure it's correct.
2. Verify the PostgreSQL version:
Make sure that your
```

**fixed:gamma6** — recorded

Length: 128; termination: length_limit; finish_reason='length'; stop_reason=None.

engine finish_reason=length; output cap and context limit may both cause this

```text
The error you're encountering suggests that the GraphQL engine is unable to connect to your DigitalOcean database. The error message indicates that the connection is failing due to a timeout, which could be caused by various factors.

Here are some steps to help you troubleshoot the issue:

1. Check your DigitalOcean database credentials:
Ensure that you're using the correct credentials (username, password, and database name) for your DigitalOcean database. Double-check the database URL to make sure it's correct.
2. Verify the PostgreSQL version:
Make sure that your
```

**fixed:gamma7** — not_recorded


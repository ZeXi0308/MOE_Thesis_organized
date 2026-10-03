Gate 0-A producer candidate: **PASS**. Safe to retain: **YES**. This is not a full Gate 0 or scientific-result PASS.

Evidence:

- Natural native router-logit capture, mutable active-set cached decode, cache merge/split, request identity conservation, serial token/route parity, CUDA synchronization, and non-overwriting partial-output handling are coherent.
- EOS correctly takes precedence over `max_decode_steps` ([producer](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_continuous_decode.py:581), [deterministic test](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/test_continuous_decode.py:216)).
- Completion accounting now exposes expected/admitted/completed/failed/filtered requests, stop reasons, decode steps, router invocations, and contribution denominators ([producer](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/capture_continuous_decode.py:1193)).
- Tests: targeted producer tests **6/6 PASS**; complete local BCRD suite **68/68 PASS**.
- Files remained byte-stable before and after testing:
  - `capture_continuous_decode.py`: `1840b760d181414751ebed37a9ee5014b10b8cbf7ddc99453ac73098fbf5624b`
  - `test_continuous_decode.py`: `1452eb9d6b102237f76454a1e42acca0b96a28d1cbdffaf273c7549f9d5d8bfb`

Separate blocker: serial equivalence validates selected expert identities and tokens, not gate-weight/full-router-logit parity. That blocks stronger exact-output/full-Gate-0 claims, not retention or Gate 0-A candidate qualification.

Formal CUDA execution authorized: **NO**. The preregistration hash `7c66a63bc9d74d318d80fe3f3cb3c3da060ac0eea183b02d28fcd9db7b4fbe7d` still has `formal_execution_authorized: false` and unresolved workload/CUDA blockers ([config](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json:44)). No files were edited.

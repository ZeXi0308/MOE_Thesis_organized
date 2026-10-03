# Admission-aware early release of Q10 protection

Three arms use the same package: q1, q10_yield, q10. Only q10_yield enables --yield-to-ready-head. At the actual non-target waiting-loop gate after first output, an extended target can release early if the PREEMPTED queue head has known empty KV ownership, no pending transfers/queue/push work, an available sequence slot, known native R=0, and free blocks sufficient for its full history plus protected target growth. The existing read-only direct-admission eligibility helper checks actual ownership. Unknown or insufficient state preserves the Q10 break.

Release is an intentional policy action and is not rolled back when subsequent native admission is absent. The receipt explicitly reports SCHEDULED_TOKENS, ASYNC_LOAD_ADMITTED, or NO_NATIVE_ADMISSION. Early release does not guarantee ten target outputs. All complete-request target/peer consequences remain part of evaluation. Sparse capture retained; no full diagnostics.

Four CPU closure fixtures cover positive release/admission receipt, intentional release without admission, unfunded head and pending work. They do not establish native lifecycle or performance. Native triplet plan: PROTECTION_YIELD_TRIPLET_PLAN_R01_20261001.json. This is a simple baseline extension, not an independent novelty claim.

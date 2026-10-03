# Final same-family integrity audit

**Verdict:** `WARN / SAME_FAMILY_PROVISIONAL`
**Deterministic integrity:** `PASS`
**Scientific status:** `BLOCKED_RUNTIME_NOT_REPRESENTATIVE`

No P0 or P1 integrity defect remains. The current snapshot consistently freezes
`next_window_active_token_budget`; the max-running variant appears only as a
rejected alternative. Canonical reproduction exited zero, all 23 tests passed,
all four derived outputs matched byte-for-byte, P2/P3 failed closed, and the
canonical artifact directory contained exactly six files.

Request/document identity overlap is zero, while reuse of one arrival episode
is explicitly recorded as an eligibility blocker. M4 is a future-route latency
diagnostic only; it is not counterfactual capacity ground truth. The full
1,337,795,800-byte source expert-call ledger was rehashed and matched
`799e9799270e5b9aa267e0ff72dce80e10650ed39d236ea6e184fa99d6747e99`.

The result remains `WARN`, not an independent `PASS`, because the reviewer is
same-family/provisional and the scientific input remains a one-model,
one-arrival-replay, non-serving smoke artifact. The next authorized work is one
bounded OLMoE P0 producer capture with explicit GO/STOP criteria.

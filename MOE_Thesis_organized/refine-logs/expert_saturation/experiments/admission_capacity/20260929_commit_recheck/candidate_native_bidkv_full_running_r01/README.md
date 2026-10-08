CPU-only, unrun candidate for a stronger adapted BidKV victim baseline.

This is a private copy of `candidate_native_current_guard_r01`. Set
`A_NATIVE_VICTIM_RULE=bidkv_score` and `A_NATIVE_VICTIM_FULL_RUNNING=on` to
include qualified pure-decode requests already scheduled earlier in the same
native running loop, as well as the original unprocessed suffix. A prefix
request must be present in `scheduled_running_reqs`; skipped prefix requests
are ineligible. Active protection or preparation retains the old suffix path.
Unknown ownership, residence, or decode state falls back to the native tail.

When a scheduled prefix wins, the patched pinned FCFS branch mirrors the
native PRIORITY rollback: it removes that request from the step's scheduled
list, refunds its token budget, removes its KV/spec/encoder output plan,
restores any encoder compute budget, shifts the loop index, then uses native
`_preempt_request` and retries the failed allocation. Per-decision records
identify prefix choices and actual rollback. After native scheduling, the
selected prefix must be absent from the output plan and present in the native
preempted IDs. Current-request guard logic uses the explicit failed-request
index, including when candidate rows start with a scheduled prefix.

Default `A_NATIVE_VICTIM_FULL_RUNNING=off` preserves the existing suffix-only
selection. The new mode is limited to adapted BidKV scoring and the existing
native-full ordinary-only execution domain. It is a baseline candidate, not a
full upstream BidKV reproduction or a GPU result. The focused CPU closure is
`test_bidkv_full_running_cpu.py`; it covers rollback, suffix/default behavior,
and explicit current/unknown fallback on the pinned scheduler AST.

# Selector Failure Decomposition — Execution tracker

- `S0 DEFINE`: DONE — decomposition, heads, exact baselines, and two-branch policy decision frozen.
- `S1 IMPLEMENT`: DONE — pure-CPU runner plus 6/6 synthetic contract tests PASS; bounded code review found no P0/P1.
- `S2 LOCK`: DONE — v2 re-lock completed after the serialization-only fix and explicit finite-result checks; model, features, thresholds, inputs, and policy decision are unchanged.
- `S3 RUN`: DONE — authoritative CPU-only `run01` completed under v2 lock; primary fresh profile rank gain is negative and the frozen rule selects the supervised-selector stop branch.
- `S4 AUDIT`: DONE — fresh same-family ultra audit PASS, P0=0, P1=0; raw labels, all decisive exact fractions, predictions, V2 lock, output manifest, and failed-attempt isolation independently reproduced.

No GPU or model inference is permitted in this experiment. Both outcome surfaces predate this analysis, so all results remain retrospective and exploratory.

Final selected policy: `STOP_SUPERVISED_SELECTOR_TO_WITNESSPATCH_BUDGETED_PROBING`. The positive fresh harm diagnostic is retained as a finding, not promoted into an unregistered harm-only policy.

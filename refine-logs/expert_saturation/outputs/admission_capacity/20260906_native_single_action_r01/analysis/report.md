# One-shot admission qualification: MEASUREMENT_ONLY

Completed 12/12; qualified 12.

| Cell | Arm | Status | Trigger | Applied actions | Goodput |
|---|---|---|---|---|---|
| forward/cell-000 | single_shadow | COMPLETE | SHADOW_INTENT_ONLY | 0 | 2.705448053404388 |
| forward/cell-001 | single_shadow | COMPLETE | SHADOW_INTENT_ONLY | 0 | 12.522543818482061 |
| forward/cell-002 | single_down | COMPLETE | ONE_ACTION_APPLIED | 1 | 0.3248389329967433 |
| forward/cell-003 | single_down | COMPLETE | ONE_ACTION_APPLIED | 1 | 12.107150513015727 |
| forward/cell-004 | static | COMPLETE | STATIC_BASELINE | 0 | 2.077160099454609 |
| forward/cell-005 | static | COMPLETE | STATIC_BASELINE | 0 | 12.319354479267139 |
| reverse/cell-000 | static | COMPLETE | STATIC_BASELINE | 0 | 12.224756399522985 |
| reverse/cell-001 | static | COMPLETE | STATIC_BASELINE | 0 | 1.3793895299977685 |
| reverse/cell-002 | single_down | COMPLETE | ONE_ACTION_APPLIED | 1 | 12.140560980247155 |
| reverse/cell-003 | single_down | COMPLETE | ONE_ACTION_APPLIED | 1 | 1.3714462854766032 |
| reverse/cell-004 | single_shadow | COMPLETE | SHADOW_INTENT_ONLY | 0 | 12.438141339125341 |
| reverse/cell-005 | single_shadow | COMPLETE | SHADOW_INTENT_ONLY | 0 | 1.538535905904362 |

forward / steady: PREFIX_NOT_ALIGNED.

forward / bursty: PREFIX_NOT_ALIGNED.

reverse / steady: PREFIX_NOT_ALIGNED.

reverse / bursty: PREFIX_NOT_ALIGNED.

Every planned forward/reverse arm is retained; no best-repeat selection.
Hold32 is a shadow observer: a recorded intention is not an applied admission action.
Observed schedule/token/submission prefixes do not establish equal KV tensors or an exact snapshot.
Untriggered and nonaligned arms retain request measurements but cannot qualify a matched one-shot action.
A binding opportunity is not an exact counterfactual divergence; target reach is not request benefit.
No dynamic Oracle, expert-signal increment, production result or method GO is inferred.


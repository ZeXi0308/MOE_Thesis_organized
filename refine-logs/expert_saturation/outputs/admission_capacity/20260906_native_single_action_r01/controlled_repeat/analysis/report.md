# One-shot admission qualification: MEASUREMENT_ONLY

Completed 12/12; qualified 12.

| Cell | Arm | Status | Trigger | Applied actions | Goodput |
|---|---|---|---|---|---|
| forward/cell-000 | single_shadow | COMPLETE | SHADOW_INTENT_ONLY | 0 | 2.2981610546322373 |
| forward/cell-001 | single_shadow | COMPLETE | SHADOW_INTENT_ONLY | 0 | 12.447746768090385 |
| forward/cell-002 | single_down | COMPLETE | ONE_ACTION_APPLIED | 1 | 1.378398099886826 |
| forward/cell-003 | single_down | COMPLETE | ONE_ACTION_APPLIED | 1 | 12.052203770307408 |
| forward/cell-004 | static | COMPLETE | STATIC_BASELINE | 0 | 1.7216028286420988 |
| forward/cell-005 | static | COMPLETE | STATIC_BASELINE | 0 | 12.184745110638753 |
| reverse/cell-000 | static | COMPLETE | STATIC_BASELINE | 0 | 12.228329285227272 |
| reverse/cell-001 | static | COMPLETE | STATIC_BASELINE | 0 | 1.7248213505192977 |
| reverse/cell-002 | single_down | COMPLETE | ONE_ACTION_APPLIED | 1 | 12.211050084269113 |
| reverse/cell-003 | single_down | COMPLETE | ONE_ACTION_APPLIED | 1 | 1.375617337242232 |
| reverse/cell-004 | single_shadow | COMPLETE | SHADOW_INTENT_ONLY | 0 | 12.47193288029101 |
| reverse/cell-005 | single_shadow | COMPLETE | SHADOW_INTENT_ONLY | 0 | 1.535907708670704 |

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


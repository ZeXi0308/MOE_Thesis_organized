# Native simple policy probe: MEASUREMENT_ONLY

Complete 24/24; qualified 24; requests 768.

| Cell | Regime | Arm | Status | Pass | Throughput | Goodput | TTFT / TPOT p50 s | Max active / decode / queue | Applied actions |
|---|---|---|---|---|---|---|---|---|---|
| forward/cell-000 | steady | static8 | COMPLETE | 8/32 | 8.12786 | 2.03196 | 0.81633 / 0.00712 | 8 / 8 / 16 | 0 |
| forward/cell-001 | bursty | static8 | COMPLETE | 8/32 | 8.97514 | 2.24379 | 0.63085 / 0.00676 | 8 / 8 / 16 | 0 |
| forward/cell-002 | steady | static12 | COMPLETE | 7/32 | 9.10713 | 1.99218 | 0.54310 / 0.00891 | 12 / 12 / 12 | 0 |
| forward/cell-003 | bursty | static12 | COMPLETE | 16/32 | 10.08155 | 5.04078 | 0.30108 / 0.00828 | 12 / 12 / 12 | 0 |
| forward/cell-004 | steady | static16 | COMPLETE | 6/32 | 11.00029 | 2.06255 | 0.10913 / 0.00895 | 16 / 16 / 8 | 0 |
| forward/cell-005 | bursty | static16 | COMPLETE | 32/32 | 12.19290 | 12.19290 | 0.05648 / 0.00827 | 16 / 16 / 8 | 0 |
| forward/cell-006 | steady | static32 | COMPLETE | 7/32 | 12.34433 | 2.70032 | 0.01802 / 0.00972 | 26 / 26 / 0 | 0 |
| forward/cell-007 | bursty | static32 | COMPLETE | 32/32 | 12.55180 | 12.55180 | 0.02650 / 0.00823 | 24 / 24 / 0 | 0 |
| forward/cell-008 | steady | shadow32 | COMPLETE | 4/32 | 12.30950 | 1.53869 | 0.01859 / 0.00973 | 26 / 26 / 0 | 0 |
| forward/cell-009 | bursty | shadow32 | COMPLETE | 32/32 | 12.47535 | 12.47535 | 0.02790 / 0.00823 | 24 / 24 / 0 | 0 |
| forward/cell-010 | steady | feedback32 | COMPLETE | 7/32 | 8.67964 | 1.89867 | 0.04208 / 0.00890 | 17 / 17 / 15 | 8 |
| forward/cell-011 | bursty | feedback32 | COMPLETE | 32/32 | 12.23227 | 12.23227 | 0.02649 / 0.00819 | 16 / 16 / 8 | 2 |
| reverse/cell-000 | bursty | feedback32 | COMPLETE | 28/32 | 10.83862 | 9.48379 | 0.02482 / 0.00826 | 16 / 16 / 8 | 3 |
| reverse/cell-001 | steady | feedback32 | COMPLETE | 7/32 | 9.22695 | 2.01840 | 0.19472 / 0.00888 | 16 / 16 / 12 | 4 |
| reverse/cell-002 | bursty | shadow32 | COMPLETE | 32/32 | 12.38604 | 12.38604 | 0.02994 / 0.00844 | 24 / 24 / 0 | 0 |
| reverse/cell-003 | steady | shadow32 | COMPLETE | 5/32 | 12.13590 | 1.89623 | 0.01812 / 0.01002 | 26 / 26 / 0 | 0 |
| reverse/cell-004 | bursty | static32 | COMPLETE | 32/32 | 12.45647 | 12.45647 | 0.02906 / 0.00829 | 24 / 24 / 0 | 0 |
| reverse/cell-005 | steady | static32 | COMPLETE | 4/32 | 12.30911 | 1.53864 | 0.01819 / 0.00989 | 26 / 26 / 0 | 0 |
| reverse/cell-006 | bursty | static16 | COMPLETE | 32/32 | 12.13394 | 12.13394 | 0.05489 / 0.00832 | 16 / 16 / 8 | 0 |
| reverse/cell-007 | steady | static16 | COMPLETE | 6/32 | 10.97036 | 2.05694 | 0.13021 / 0.00901 | 16 / 16 / 8 | 0 |
| reverse/cell-008 | bursty | static12 | COMPLETE | 16/32 | 9.99638 | 4.99819 | 0.33500 / 0.00838 | 12 / 12 / 12 | 0 |
| reverse/cell-009 | steady | static12 | COMPLETE | 7/32 | 9.09843 | 1.99028 | 0.54801 / 0.00897 | 12 / 12 / 12 | 0 |
| reverse/cell-010 | bursty | static8 | COMPLETE | 8/32 | 9.01724 | 2.25431 | 0.61029 / 0.00677 | 8 / 8 / 16 | 0 |
| reverse/cell-011 | steady | static8 | COMPLETE | 8/32 | 7.95279 | 1.98820 | 0.81574 / 0.00721 | 8 / 8 / 16 | 0 |

All arm pair comparisons and descriptive SLO grid are retained in analysis.json.

All main and reference request metrics are recomputed from retained host timestamps; capture and feedback costs remain in their measured wall-time denominator.
Static 8/12/16/32 are all reported. The best observed static point is a descriptive comparison, not a trained or held-out policy.
Shadow uses the feedback observer/decision code but keeps admission at 32; shadow intent changes are not applied actions.
Existing decode sets are reconstructed from prior scheduled tokens and request completion, then checked for advancement at every step.
Recomputed observation windows contain only completed steps available before each decision; action application precedes that scheduler result.
Request-level mean TPOT, step median ITL feedback, and worst token ITL are distinct quantities.
Action-bound timing and binding opportunities do not identify an exact counterfactual admission-delay effect.
All arms independently execute future state; neither identical hidden state nor equal output semantics is asserted.
Two counterbalanced repeats of the same finite input are descriptive; no step-level independent-sample inference, dynamic Oracle, or expert-signal claim.
The frozen nine-point SLO grid cannot select the canonical policy or change the feedback threshold.


# C01 novelty red-team

## Verdict

`KILL_CCFB_METHOD_CURRENT_FORM`.

C01's paired execution, altered-event propagation, common-event cancellation, and exact equivalence to full discrete-event simulation directly collide with Hanai et al.'s **Exact-Differential Simulation**. Product programs, shadow symbolic execution, dynamic slicing, causally consistent slicing, DPOR, timed-zone reduction, and max/min-plus event-graph analysis further occupy the generic product-system, causal-slice, interleaving-reduction, and fork-join mathematics around the proposal.

Modern LLM-serving simulators and schedulers already supply the domain event vocabulary and execution models. Replacing their events with MoE route, dispatch, queue, expert, and combine events does not by itself create a new method.

## Direct collision set

1. Exact-Differential Simulation: <https://kalper.net/kp/publication/2019-06-18-tomacs-exact-differential/2019-06-18-TOMACS-Exact-differential.pdf>
2. P³ product programs: <https://daniel.schemmel.net/publication/2025-product-programs.pdf>
3. Shadow Symbolic Execution: <https://arxiv.org/abs/1802.01714>
4. Dynamic Slicing of Concurrent Programs: <https://arxiv.org/abs/2211.04683>
5. Causally Consistent Dynamic Slicing: <https://arxiv.org/abs/1610.02327>
6. DPOR families: <https://arxiv.org/abs/2111.05290> and <https://arxiv.org/abs/1909.00989>
7. Timed-zone reductions: <https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CONCUR.2019.16> and <https://arxiv.org/abs/2602.15435>
8. Max/min-plus event graphs: <https://arxiv.org/abs/2003.04703>
9. LLM-serving simulators: Vidur, APEX, LLMServingSim2, and Charon.
10. MoE serving/scheduling neighbors: AMoE and FATE.

## What would have to remain

The only admissible paper-only successor is a **Route-Join Quotient** claim with all three obligations met before implementation:

1. a generic exact-differential/POR baseline requires `Theta(n)` retained events on a declared trace family;
2. the MoE quotient retains `o(n)` events on the same family; and
3. a sound observational-bisimulation theorem survives dynamic batching, shared queues, nonlinear service, and resource-side-effect re-entry.

Without this strict separation, C01 is useful exactness infrastructure, not a CCF-B method contribution. The existing resource-side-effect counterexample already invalidates the proposed barrier closure, so the successor is not currently established.

## Evidence boundary

This is a same-family, literature-based provisional audit. It contains no pilot or scientific result and does not authorize implementation.

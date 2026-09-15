# LTR counter component probe

UNRUN

A scheduling-counter component on a shared custom priority-packing/current-history-reservation backend using native recompute; boost-off is not native vLLM and neither arm is full LTR. Measured wall includes decision, instrumentation and runtime costs without subtraction. No SLO, Oracle, quality, significance or method GO. Same-input repeats are correlated. Recovery spans overlap and cannot be summed into wall. Resource checks use qualified APC-off full-attention ownership counts and actual releases, not an independent physical block-ID replay. No full planner replay: the resolved long-prefill chunk threshold is not recorded.

- block0-d6-boost-off: UNRUN
- block0-d6-boost-on: UNRUN
- block1-d6-boost-on: UNRUN
- block1-d6-boost-off: UNRUN

Full per-request changes, output differences, actual recovery spans, resource and nested costs are in analysis.json.

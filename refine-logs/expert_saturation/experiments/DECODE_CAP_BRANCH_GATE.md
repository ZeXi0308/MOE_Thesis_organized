# Native vLLM Initial Active-Sequence-Cap Gate

## Verdict before execution

`UNRUN / REQUEST_LEVEL_INITIAL_SAME_START_BRANCH_EXPLORATORY`

This harness answers one question only:

> For one frozen prompt cohort and order, does changing the native vLLM
> `max_num_seqs` cap produce material request-level SLO-goodput headroom when
> each arm independently generates its KV, route, queue, and completion
> trajectory?

It does **not** implement a later-epoch snapshot branch, a pressure-conditioned
policy, or an online controller.

## Source-grounded actuator semantics

The implementation was checked against vLLM `v0.26.0`, commit
`568afb3a13806beb53bb2e6bd518269357b237c0`:

- `vllm/entrypoints/offline_utils.py:290-349` converts the entire prompt
  sequence and adds all requests before `_run_engine` starts draining them.
- `vllm/v1/core/sched/scheduler.py:109-115` assigns `max_num_seqs` to
  `max_num_running_reqs`.
- `vllm/v1/core/sched/scheduler.py:665-680` leaves requests in the waiting
  queue when the running count reaches that cap.
- `vllm/v1/core/sched/request_queue.py:75-90` implements the default request
  queue as FCFS; `vllm/config/scheduler.py:109` declares `fcfs` as the default.

The runner seals these actuator sources and the analyzer requires the exact
vLLM 0.26.0 SHA-256 identities before any result can qualify:

```text
entrypoints/offline_utils.py        688fbad0af9c2180b83aa77dcd0dbda85ca076a6c72bffa61840896d950cf458
v1/core/sched/scheduler.py          2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941
v1/core/sched/request_queue.py      4b8d938e5fb8152fe61030b8aa991f983fcac9a405c37ed8908045e05e82ae5e
config/scheduler.py                 a816cf79a3e74ffc0984f9bebb274275b26f46be8b28cb77a29388a0996263c8
```

The analyzer also freezes `runtime_patch_id=valid-window-clear-v1` and the two
patched telemetry-source hashes from `VALID_WINDOW_TELEMETRY_GATE.md`; a label
without those exact loaded-source identities is invalid.

Therefore `submitted request count > max_num_seqs` is supported: the complete
cohort is enqueued and only up to the cap is admitted as running requests.
However, `max_num_seqs` bounds all running sequences, including initial
prefill. In this initial-cohort harness it is an **active-sequence cap**, not a
pure decode-only action at an already-materialized KV snapshot. Equal prompt
length, fixed output length, FCFS, disabled prefix caching, sufficient token
budget, and `cohort_size % cap == 0` make consecutive cap-sized route waves a
useful inference, but no scheduler-event trace is claimed.

## Frozen six-run design

Use one process-isolated fresh engine per bundle:

```text
low-OFF  low-ON
mid-OFF  mid-ON
high-OFF high-ON
```

All six runs must require an exclusive GPU, record an empty pre-engine compute
process list, and share model/revision, runtime source hashes, runtime patch
ID, replicate ID, workload bytes, prompt token IDs/order, prompt/output length,
seed, token budget, and engine settings. Only `decode_cap`, budget label, and
route capture switch may differ. Recommended first matrix:

```text
cohort = 48, prompt = 512, output = 32
b_low = 4, b_mid = 8, b_high = 16
max_num_batched_tokens = 8192
```

Run each bundle into a new path; never overwrite a run. Example for one arm:

```bash
python experiments/run_vllm_decode_cap_branch.py \
  --output-dir outputs/decode_cap/r0-low-off \
  --workload-manifest /path/to/workload.json \
  --budget-arm low --decode-cap 4 --replicate-id r0 \
  --runtime-patch-id valid-window-clear-v1 --no-capture-routes

python experiments/run_vllm_decode_cap_branch.py \
  --output-dir outputs/decode_cap/r0-low-on \
  --workload-manifest /path/to/workload.json \
  --budget-arm low --decode-cap 4 --replicate-id r0 \
  --runtime-patch-id valid-window-clear-v1 --capture-routes
```

Repeat with `mid/8` and `high/16`, then analyze with SLOs frozen **before**
looking at results:

```bash
python experiments/analyze_vllm_decode_cap_branches.py \
  --low-off outputs/decode_cap/r0-low-off \
  --low-on outputs/decode_cap/r0-low-on \
  --mid-off outputs/decode_cap/r0-mid-off \
  --mid-on outputs/decode_cap/r0-mid-on \
  --high-off outputs/decode_cap/r0-high-off \
  --high-on outputs/decode_cap/r0-high-on \
  --tpot-slo-ms <FROZEN_TPOT_SLO> \
  --ttft-slo-ms <FROZEN_TTFT_SLO> \
  --output outputs/decode_cap/r0-analysis.json
```

If both SLOs are omitted, analysis returns
`INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_SLO_UNSET` and emits no comparison. If
only one is supplied, or either is non-positive, the input is invalid.

## Measurement and join rules

- Route-OFF is the only performance source: per-request TTFT, queue, prefill,
  TPOT, end-to-end latency, wall time, and output throughput.
- Route-ON supplies routed-expert pressure only.
- Pressure may join OFF timing only when, for **every** budget, OFF/ON generated
  token IDs match exactly and the **absolute signed deviation** of both
  wall-time and request-P95-TPOT is at most 5%. This is deliberately
  two-sided: an ON run that is more than 5% faster is also non-transparent and
  fails closed instead of being treated as "negative overhead."
- Decision timing is reconstructed from the complete `requests.jsonl` ledger:
  every row carries raw branch start/finish counters and raw queued/scheduled/
  first-token/last-token timing. The analyzer validates finiteness, ordering,
  positivity, the derived per-request fields, and then recomputes wall time,
  throughput, and all request quantiles. `summary.json` is only an independent
  consistency check and cannot override the request ledger.
- Route pressure is reconstructed from `routes.npz`, never accepted from the
  summary alone. The analyzer checks the exact archive member, integral dtype,
  frozen `[cohort, decode_steps, layers, top_k]` shape, expert-ID range,
  distinct top-k IDs, and the route topology frozen in `config.json` before
  recomputing inferred FCFS-wave pressure.
- Cross-budget token drift is retained as an action-conditioned outcome. It
  does not fail telemetry transparency because each budget arm owns its future
  KV/route state. When it occurs, the latency delta cannot be described as a
  same-token execution-only effect.
- Every request must finish by `length` with exactly the frozen output count.
  The denominator is the complete cohort; no incomplete request is censored.
- Joint SLO-goodput is the number of output tokens belonging to requests that
  satisfy both frozen TTFT and TPOT thresholds, divided by full branch wall
  time. The exploratory continuation gate is best-cap versus low-cap goodput
  improvement of at least 3%. Both the 5% telemetry threshold and 3% headroom
  threshold are frozen Gate constants; CLI values cannot relax them. If the
  low-cap goodput is zero, relative improvement is undefined and the result is
  valid non-positive rather than scientific success.

Possible final statuses are limited to:

```text
INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_POSITIVE
INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_BELOW_GATE
INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_TELEMETRY_INVALID
INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_SLO_UNSET
INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_UNDEFINED_RELATIVE_BASELINE
INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_INVALID_INPUT
```

Even a positive result authorizes only a later, separate snapshot-cap actuator
experiment. It does not establish a pressure-conditioned effect or Controller.

The analyzer's process exit status is part of the gate contract:

```text
0  INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_POSITIVE only
1  valid but below-gate, undefined-baseline, telemetry-nonqualified, or SLO-unset result
2  invalid input, corrupt/inconsistent evidence, or CLI usage error
```

Automation must therefore not interpret a syntactically valid report with a
non-positive scientific result as success.

## Sealed artifacts

Each runner bundle contains config/environment/workload identity, exact prompt
tokens, all request rows, optional route tensor, summary, artifact hashes, and
`RUN_COMPLETE.json`. It embeds both the exact branch runner as
`producer_source.py` and its imported helper as
`run_vllm_route_shape_probe.py`. Both hashes are bound into config, experiment
identity, the complete artifact map, and the final seal, so the frozen producer
can execute its CLI from the bundle directory without an unsealed local Python
dependency. The analyzer requires the exact artifact set, verifies every hash
and the complete seal map, checks both source identities plus the scheduler
source map, validates the full request denominator, and enforces identical
experiment/runtime/patch/producer/helper/route-topology identities, arm labels,
telemetry polarity, GPU isolation evidence, and strict budget order.

One analysis consumes one six-branch sextet. It is explicitly a single-
replicate exploratory Gate; it does not establish repeat-level stability and
cannot replace the later confirmatory multi-replicate experiment.

## Exploration-code budget self-check

1. **Why the existing probe cannot answer this:**
   `run_vllm_route_shape_probe.py` rejects a submitted batch larger than
   `max_num_seqs`, treats each submitted batch as a fixed shape, and has no
   six-arm SLO-goodput join contract.
2. **Why a smaller A/B is insufficient:**
   low/high alone cannot locate whether the ordinary Token/KV response has a
   knee, while route pressure cannot be joined without an OFF/ON transparency
   control at each changed cap. Three budgets are the smallest knee probe.
3. **Which single uncertainty the new code closes:**
   whether initial native active-sequence-cap action headroom exists on one
   complete frozen cohort. No routing, precision, placement, regrouping, or
   controller mechanism is introduced.

The runner/analyzer are deliberately standalone because this denominator and
six-arm fail-closed contract differs from the fixed-batch measurement probe.
The code should not grow again before real GPU evidence changes this verdict.

# GPU route-pressure sketch for vLLM 0.26 (conditional backup)

## Verdict and claim ceiling

The **Primary** implementation Gate is now the lossless valid-window patch:

`../vllm_patches/vllm-0.26-valid-route-window.patch`

It changes only the per-step clear from the full worst-case route buffer to the
exact token prefix that the sync D2H or async snapshot can expose. It preserves
the public full-route API, so it is a stronger simple baseline than introducing
a second telemetry contract.

The compact sketch in this directory is a **Conditional Backup**. Use it only
if the valid-window path preserves token parity but still exceeds the frozen
overhead threshold. It attaches a lossy contract to the existing logical-ID
router hook and keeps only:

- per-layer `max_expert_load` and `active_experts` for the current step; and
- the exact `[layer, top_k]` route of each request's last scheduled token.

The compact path is feasible for the current single-GPU, eager, synchronous,
normal-decode probe. It is **not** a drop-in replacement for vLLM's
routed-expert API: it cannot reconstruct every token's historical route. No
overhead or serving benefit is claimed until its own paired native benchmark
passes.

## Existing native evidence (before valid-window optimization)

- The four-pair smoke (`P=64`, `B=2/4`) preserved output-token parity. Its
  observed TPOT overhead was 10.14--13.35% and wall overhead 10.07--12.38%, so
  it failed the 5% telemetry-overhead Gate.
- The two full 36-pair matrices did **not** preserve route-ON/OFF output-token
  parity: process repeat 0 had 1 mismatched cell and repeat 1 had 6 mismatched
  cells. Their comparison status is `INVALID_TELEMETRY_PAIR`.
- Consequently, full-matrix ON/OFF timing differences are diagnostic only and
  cannot support an overhead claim. The mismatch does not by itself prove that
  capture caused semantic drift: these are separate-process executions and the
  first-divergence source is not localized.

Evidence files:

- `../../outputs/native_route_shape/remote_snapshot_20260823/runs/smoke-r1-telemetry-comparison.json`
- `../../outputs/native_route_shape/remote_snapshot_20260823/runs/full-r0-telemetry-comparison.json`
- `../../outputs/native_route_shape/remote_snapshot_20260823/runs/full-r1-telemetry-comparison.json`

The valid-window Gate must therefore require token parity before any timing
comparison. If parity fails, mark the pair invalid and localize the first token
divergence; do not interpret the overhead numbers.

## Why the valid-window patch is source-safe

- Both sync D2H (`buf[:total]`) and async private clone (`buf[:total].clone()`)
  already use `scheduler_output.total_num_scheduled_tokens`; the patch clears
  that same observable prefix before forward.
- Dense/non-MoE layers are safe because every layer column in the visible
  prefix is zeroed; only MoE layer callbacks overwrite their columns.
- On a shrink, stale rows remain only beyond the returned prefix. If a later
  step grows, its larger prefix is cleared before any capture, so stale rows
  cannot re-enter output.
- The sync path completes the D2H at its existing sampling synchronization. The
  async path first clones the active prefix on the default stream; a later
  default-stream clear cannot mutate that private snapshot.
- DP/SP ownership remains the existing capturer contract: it writes this DP
  rank's reconstructed local rows at the start of the buffer, and the model
  runner exports its local scheduled-token prefix. The patch does not add a
  collective or change row ownership.

This review does not widen vLLM's support envelope. PP, KV connectors, and
context parallelism already reject full-route return; DBO/V2 limitations remain
unchanged.

## Exact vLLM 0.26 call path

1. `BaseRouter._select_experts()` computes logical `topk_ids` and calls
   `capture_fn` before EPLB remaps logical to physical IDs
   (`vllm/model_executor/layers/fused_moe/router/base_router.py:290-303`).
2. `GPUModelRunner._bind_routed_experts_capturer()` binds a layer-aware closure
   to every `MoERunner` router (`gpu_model_runner.py:7677-7690`).
3. The current capturer writes full
   `[max_num_batched_tokens, layers, top_k]` int32 state and zeroes that whole
   worst-case buffer each step (`routed_experts_capturer.py:93-106,203-212`).
4. Sync execution copies the valid route slice and an int64 slot map to pinned
   CPU buffers, then relies on the sampling `_to_list()` event synchronization
   (`gpu_model_runner.py:3707-3733,4726-4735`).
5. The scheduler persists all routes by physical KV slot and later reconstructs
   request histories (`routed_experts_capturer.py:223-349`,
   `scheduler.py:1626-1645`).

The full historical-return contract explains why vLLM stores raw IDs.  The
compact pressure contract should be a separate feature, not a silent change to
`enable_return_routed_experts`.

## Minimal integration (one new module, three existing files)

The executable device module is `route_pressure_sketch.py`.  For the first
native Gate, wire it in as follows:

1. **`GPUModelRunner`**
   - create `GPURoutePressureSketch(max_num_reqs, L, E, K, device)` after the
     model is loaded;
   - reuse `_bind_routed_experts_capturer()`'s closure pattern;
   - after `_prepare_inputs()`, call `begin_step(req_indices.gpu,
     query_start_loc.gpu, num_reqs, total_num_scheduled_tokens)`;
   - after forward, call `finalize()`;
   - copy only active `summary[L,2]` and `signature[num_reqs,L,K]` into pinned
     host buffers before the existing `_to_list()` synchronization.
2. **`vllm/v1/outputs.py`**
   - add a compact `RoutePressureLists(layer_summary, request_signature)` field
     to `ModelRunnerOutput`; do not attach it to user completion output.
3. **`Scheduler.update_from_output()`**
   - persist each signature by `model_runner_output.req_ids[row]`;
   - expose the preceding step's summary/signatures to the next `schedule()`;
   - remove a signature in `_free_request()`.

For an experiment-only activation, an environment flag can avoid plumbing a
public CLI option.  A production patch should add a typed config and capability
validation instead.

## Device operations and transfer accounting

Per step, on the model's current CUDA stream:

1. zero `[L,E]` int32 counts and fill the active signature rows with `-1`;
2. one fused Triton launch per MoE layer that atomically counts every top-k ID
   and writes only the last scheduled token for each request;
3. one `L`-program reduction launch for max load and active-expert count;
4. enqueue pinned D2H for `[L,2]` int32 and `[R,L,K]` int16;
5. piggyback on the synchronization already required for sampled tokens.

This introduces no *new host synchronization point* in the synchronous path,
but it extends the existing barrier and adds GPU kernels, so its latency cost
must be measured.

For current OLMoE `R=16,L=16,K=8` normal decode:

- compact D2H: `16*2*4 + 16*16*8*2 = 4,224` bytes;
- current full route + slot D2H: `16*16*8*4 + 16*8 = 8,320` bytes;
- current worst-case route buffer cleared at `max_num_batched_tokens=8192`:
  `8192*16*8*4 = 4,194,304` bytes;
- compact persistent counts plus active signatures are 4,096 + 4,096 bytes.

These are byte-accounting facts, not evidence of latency reduction.

## Semantics available to the next action

`request_signature[r]` is the route for the last token of request `r` that was
actually forwarded in step `t`.  `EngineCore.step()` runs
`schedule(t) -> execute(t) -> update_from_output(t)`, so a standard synchronous
engine can use it in `schedule(t+1)` without future leakage
(`vllm/v1/engine/core.py:576-606`).  New requests have no signature and need a
fixed fallback.  The signature remains keyed by request ID, so batch reordering
does not reinterpret it.

This guarantee does not hold for the batch-queue path, which can schedule a
new batch before processing the preceding output (`core.py:617-705`).

## Deliberate blockers for the first Gate

- **TP/SP and DP:** current full capture slices DP layouts and may all-gather SP
  rows.  The sketch currently fails closed instead of guessing ownership.
- **DBO/microbatching:** callback rows and global request offsets need an
  explicit microbatch offset.
- **CUDA graphs:** buffer pointers are persistent, but dynamic valid-token
  scalars and callback capture have not been qualified.
- **Speculative decode:** the last scheduled route need not correspond to the
  last accepted token; normal decode is the first Gate.
- **Async scheduling/batch queue:** pressure can arrive one action late.
- **PP/KV disaggregation:** worker-to-scheduler ownership and timing are not
  qualified.
- **Historical route API:** max/active/signature cannot serve requests for the
  full per-token route tensor.

## Minimal next benchmark

First run two fresh-process arms with identical model revision, dtype, prompt
token IDs, output length, batch order, and GPU isolation:

1. route-off;
2. full routed-expert export with valid-window clear.

For every cell/repeat:

- require exact output-token parity across all arms;
- compare paired TPOT, wall time, and token throughput;
- only compare timing after exact token parity passes;
- require full-export P95 TPOT/wall overhead `<= 5%` versus route-off;
- retain all process repeats and report sign flips rather than selecting a run.

If the lossless valid-window arm passes, keep it and do not deploy the compact
path. If it preserves parity but remains over 5%, then run a third compact arm;
reduce the lossless routes with `aggregate_reference()` and require exact
equality with compact max/active/signatures before considering its timing.

Only after telemetry qualification should the scheduler receive a read-only
previous-step pressure estimator. An admission/batching action still requires
an action-conditioned Oracle/headroom experiment; telemetry alone does not
establish controller benefit.

## Local validation

```bash
python3 -B -m unittest -v \
  refine-logs/expert_saturation/experiments/gpu_pressure_sketch/test_route_pressure_sketch.py

python3 -B \
  refine-logs/expert_saturation/experiments/gpu_pressure_sketch/validate_vllm_026_source.py \
  /private/tmp/vllm-v0.26.0

# In a vLLM development environment after applying the valid-window patch:
pytest -q tests/model_executor/test_routed_experts_capture.py
```

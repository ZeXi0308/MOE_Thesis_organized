# vLLM 0.26 valid-window routed-expert telemetry Gate

## Current evidence

The native fixed-batch probe found that stock full routed-expert export is not
an observationally transparent signal source in the tested OLMoE/vLLM regime:

- route-OFF repeats produced identical tokens in all 36/36 cells;
- route-ON versus route-OFF drifted in 1/36 and 6/36 cells;
- route-ON timing was therefore excluded from the pressure/TPOT decision;
- aggregate route statistics remained repeat-stable, but temporal prediction
  did not qualify for an action experiment.

The durable verdict is
`outputs/native_route_shape/native_route_pivot_analysis_v1.json`:
`WORKING_SET_MEASUREMENT_ONLY / TELEMETRY_TRANSPARENCY_FAILED`.

An independent integrity audit additionally found that the exact producer
source recorded by these historical bundles is no longer present. They remain
internally sealed observational diagnostics, but their reproducibility label is
`producer-source-unverified`; they cannot qualify this implementation Gate.

## Strongest simple intervention

Stock vLLM clears the full routed-expert device buffer at the start of every
step. With the frozen probe configuration this allocation is:

```text
8192 max tokens * 16 layers * 8 experts/token * 4 bytes = 4 MiB
```

The model runner later copies only
`[:scheduler_output.total_num_scheduled_tokens]`. The patch in
`vllm-0.26-valid-route-window.patch` therefore preserves the lossless API but
clears only the slice that can be observed in the current step.

Safety invariant:

```text
clear current observable token window
-> run every router capture for that step
-> copy exactly the same observable token window
```

Changing batch size between steps is safe under this invariant: rows outside
the current window are never copied, and a later larger window is cleared
before it becomes observable. Dense/non-MoE layer positions inside the current
window remain zeroed. The no-argument full-clear behavior is retained for
compatibility.

This patch has not yet passed a GPU runtime Gate. Its only current evidence is
source review, clean apply/reverse-apply against vLLM 0.26, source validation,
and syntax checks. Focused CPU regressions are included in the patch but were
not executed on the local Mac because that interpreter has no PyTorch.

## Four-arm experiment

Use fresh processes, identical model/workload/order, and two controlled
process repeats:

1. stock source, route OFF;
2. stock source, full route ON;
3. valid-window source, route OFF;
4. valid-window source, full route ON.

Every run must embed its exact producer source and retain hashes of the two
loaded vLLM source files. Patch labels are not evidence: stock arms must use
`stock-vllm-0.26.0` and match the validator's exact `original` hashes, while
optimized arms must use `valid-window-clear-v1` and match the exact `patched`
hashes. Compare arms 1/2 and 3/4 separately so each telemetry pair has
identical runtime source hashes.

Qualification requires all of the following:

- arm 3 and arm 4 have exact generated-token parity in every cell/repeat;
- both repeats have complete, nonzero route coverage; stock and patched
  route-ON tokens and route arrays match exactly in every cell; no mismatched
  cell may be skipped from the route-semantics denominator;
- every full route artifact passes manifest/row-hash, NPZ-key, integer-dtype,
  shape, expert-range, and per-token top-k uniqueness validation;
- patched route-ON P95 absolute pairwise deviation in both wall time and
  request-P95-TPOT is at most 5% versus patched route-OFF. Large negative drift
  fails just like large positive drift because either direction breaks the
  timing/pressure exchangeability assumption;
- stock-OFF and patched-OFF retain token parity (the patch path is inactive
  when routed-expert capture is disabled);
- all repeats are retained; a sign flip or isolated mismatch is reported, not
  selected away.

The comparator accepts two or more paths for each arm and returns exit code
`0` only for `VALID_WINDOW_TELEMETRY_QUALIFIED`, `1` for a valid but failed
scientific Gate, and `2` for invalid input, identity, coverage, or artifacts.
The 5% absolute-deviation threshold is a frozen protocol constant: a nonfinite,
negative, or different CLI value is invalid and cannot reuse the qualified
status. One process repeat can never qualify.

If this Gate passes, use the lossless valid-window path as the strongest
telemetry baseline. Only if it fails should the lossy GPU pressure sketch under
`../gpu_pressure_sketch/` be integrated and benchmarked.

## Frozen execution driver

`../run_valid_window_telemetry_gate.py` is the only N0b campaign entrypoint.
It does not expose the model, matrix, seeds, repeat count, patch IDs, capture
flags, workload hash, or 5% threshold as run-time choices. It:

1. verifies the exact stock vLLM 0.26 source and an idle GPU;
2. copies the canonical workload and all control sources into a write-once
   campaign directory;
3. snapshots the entire stock `vllm` package twice inside the campaign,
   patches only the optimized snapshot, and verifies that the two package
   manifests differ only at the two frozen patch targets;
4. runs two process repeats across all four arms, with reverse order in repeat
   1, while retaining every original bundle and log;
5. treats both `SIGTERM` and SSH/session `SIGHUP` as an abort, terminates the
   complete arm process group, escalates to `SIGKILL` if needed, and records a
   post-cleanup idle-GPU check before returning;
6. verifies every bundle and calls the locked comparator exactly once.

Remote preflight:

```bash
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python \
  /root/autodl-tmp/expert-saturation/probe/run_valid_window_telemetry_gate.py \
  preflight \
  --python /root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
```

Recovery note: retained historical bundles show that this host previously ran
OLMoE successfully from the same venv path with Python 3.12.3, vLLM 0.26.0,
Torch 2.11.0+cu130, CUDA 13.0, and `VLLM_USE_FLASHINFER_SAMPLER=0`. This is an
installation hint only; it does not substitute for a current `pip check`,
exact-source validation, model-cache lookup, or idle-GPU preflight.

Fresh campaign (the output root must not already exist):

```bash
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python \
  /root/autodl-tmp/expert-saturation/probe/run_valid_window_telemetry_gate.py \
  run \
  --python /root/autodl-tmp/expert-saturation/vllm-0.26/bin/python \
  --workload-manifest /root/autodl-tmp/expert-saturation/probe/olmoe.formal.json \
  --output-root /root/autodl-tmp/expert-saturation/runs/n0b-valid-window-<unique-id>
```

The driver has `15/15` focused CPU orchestration tests; the complete top-level
experiment suite is `102/102 PASS` and the pressure-sketch suite is `5/5 PASS`.
An independent adversarial P0/P1 review passed at orchestrator SHA-256
`dc1d9731622f99a3108d418a43a4132f284a1c7bc2e24c8fd94456295be6bde3`.
This is execution readiness only; the GPU Gate remains `UNRUN`.

## Claim ceiling

Passing this Gate proves only bounded-deviation, non-perturbative telemetry for the
tested single-GPU eager runtime. It does not prove that route pressure predicts
latency, that a controller has headroom, or that the result transfers to EP.

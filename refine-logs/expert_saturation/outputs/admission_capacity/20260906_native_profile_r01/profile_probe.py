"""A diagnostic hook for the frozen arrival-swap runner; no engine construction.

Integration, after the unchanged engine initialization and three warmups::

    verify_frozen_sources(archive_path, frozen_source_dir)
    # capture_episode is imported from that extracted native_capture.py.
    raw = profile_bursty_episode(engine, capture_episode, workload, config,
        mode="on", run_id="profile_on", trace_path=output_dir / "trace.json")
    # For the paired independent episode use mode="off", run_id="profile_off".
    # Save raw after return. Keep cap8, arrival_scale=.02 and existing SLOs.

OFF includes the same host wrappers/record_function calls, unlike the old baseline.
This module's CLI only checks frozen source bytes; it never initializes CUDA.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tarfile
import time

ARCHIVE_SHA256 = "56498e09b32ab53a101619952f9024df1d663b7e23cafcec6c33d59c2a89834b"
SOURCE_NAMES = ("run_native_capacity.py", "native_capture.py", "metrics.py")


def verify_frozen_sources(archive_path, source_dir=None):
    """Read/compile three source members; optionally compare extracted files."""
    path = Path(archive_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("not the frozen arrival-swap execution archive")
    hashes = {}
    with tarfile.open(path) as archive:
        for name in SOURCE_NAMES:
            code = archive.extractfile(name).read()
            compile(code, name, "exec")
            hashes[name] = hashlib.sha256(code).hexdigest()
            if source_dir is not None and (Path(source_dir) / name).read_bytes() != code:
                raise ValueError(f"extracted source differs: {name}")
    return dict(archive_sha256=ARCHIVE_SHA256, source_sha256=hashes)


def profile_bursty_episode(engine, capture_fn, workload, config, *, mode, run_id,
                          trace_path=None, arrival_scale=0.02, max_seconds=120):
    """Wrap a drained, warmed synchronous engine; return raw plus diagnostic spans."""
    import torch
    from vllm.forward_context import get_forward_context

    if mode not in ("off", "on"):
        raise ValueError("mode must be off or on")
    scheduler_config = engine.vllm_config.scheduler_config
    if (config["cap"] != 8 or config["prompt_tokens"] != 128 or config["output_tokens"] != 16
            or scheduler_config.max_num_seqs != 8 or scheduler_config.async_scheduling is not False):
        raise ValueError("probe requires the frozen cap8 / engine8 / 128+16 synchronous configuration")
    if arrival_scale != 0.02:
        raise ValueError("probe preserves the arrival-swap scale .02")
    core = engine.engine_core.engine_core
    if (engine.has_unfinished_requests() or core.scheduler.requests
            or core.scheduler.get_request_counts() != (0, 0) or core.scheduler.max_num_running_reqs != 8):
        raise ValueError("probe requires a drained engine already set to admission cap8")
    destination = Path(trace_path) if trace_path is not None else None
    if mode == "on" and (destination is None or destination.exists()):
        raise ValueError("ON needs a new, nonexisting trace path")
    executor = core.model_executor
    runner = executor.driver_worker.worker.model_runner
    profiler = None
    if mode == "on":
        if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise RuntimeError("CUDA profiler activity unavailable; no CPU-only substitute")
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False,
            with_stack=False, with_flops=False)
    spans, shapes, originals = [], [], []
    step_id, raw, profiling_error = -1, None, None
    missing = object()

    def install(owner, attribute, label):
        original = getattr(owner, attribute)
        previous_instance_value = vars(owner).get(attribute, missing)

        def wrapped(*args, **kwargs):
            nonlocal step_id
            if label == "engine.step":
                step_id += 1
            current = step_id
            started = time.perf_counter()
            try:
                with torch.profiler.record_function(f"moe_probe/{mode}/step={current}/{label}"):
                    if label == "model.forward":
                        context = get_forward_context()
                        descriptor = context.batch_descriptor
                        shapes.append(dict(step_id=current, mode=context.cudagraph_runtime_mode.name,
                            skip_compiled=context.skip_compiled,
                            padded_tokens=None if descriptor is None else descriptor.num_tokens,
                            physical_requests=None if descriptor is None else descriptor.num_reqs,
                            uniform_decode=None if descriptor is None else descriptor.uniform))
                    return original(*args, **kwargs)
            finally:
                ended = time.perf_counter()
                spans.append(dict(step_id=current, stage=label, start_perf_s=started,
                                  end_perf_s=ended, host_elapsed_s=ended - started))

        setattr(owner, attribute, wrapped)
        originals.append((owner, attribute, previous_instance_value))

    # step_fn is a cached bound method: wrapping core.step alone would miss it.
    targets = [(engine, "step", "engine.step"), (core, "step_fn", "core.step"),
        (executor, "execute_model", "executor.execute_model"),
        (executor, "sample_tokens", "executor.sample_tokens"),
        (runner, "_model_forward", "model.forward"), (runner, "_to_list", "D2H.to_list_wait"),
        (core.scheduler, "update_from_output", "scheduler.update_from_output"),
        (engine.output_processor, "process_outputs", "frontend.process_outputs")]
    call_start = call_end = None
    try:
        for target in targets:
            install(*target)
        # Profiler startup/stop/export are outside capture's request clock.
        with profiler if profiler is not None else nullcontext():
            call_start = time.perf_counter()
            raw = capture_fn(engine, workload, config, regime="bursty", arrival_scale=arrival_scale,
                             run_id=run_id, max_seconds=max_seconds)
            call_end = time.perf_counter()
    except Exception as exc:
        if raw is None:
            raise
        profiling_error = f"{type(exc).__name__}: {exc}"
    finally:
        for owner, attribute, previous in reversed(originals):
            if previous is missing:
                delattr(owner, attribute)
            else:
                setattr(owner, attribute, previous)
    if profiler is not None and profiling_error is None:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(destination))
        except Exception as exc:
            profiling_error = f"trace export: {type(exc).__name__}: {exc}"
    expected_shape_steps = [s["step"] for s in raw["scheduler_steps"]
        if sum(r["decode_tokens"] for r in s["scheduled"]) == 4
        and sum(r["prefill_tokens"] for r in s["scheduled"]) == 512]
    raw["profiling_probe"] = dict(mode=mode, status="PROFILE_FAILED" if profiling_error else "CAPTURED",
        error=profiling_error, trace_path=str(destination) if mode == "on" and profiling_error is None else None,
        call_start_perf_s=call_start, call_end_perf_s=call_end, host_spans=spans, physical_shapes=shapes,
        target_step_ids=expected_shape_steps,
        geometry_status="TARGET_SHAPE_OBSERVED" if expected_shape_steps else "TARGET_SHAPE_NOT_OBSERVED",
        semantics=["Host spans are inclusive/nested; do not sum them as disjoint costs.",
            "step_id joins host spans, profiler labels, and raw scheduler_steps within this episode.",
            "Absolute perf_counter spans and original capture-relative *_s have distinct origins; use step_id to join.",
            "execute_model host return may precede GPU completion; D2H.to_list_wait includes earlier GPU wait.",
            "CUDA graph launch CPU duration is not GPU execution duration; inspect CUDA events on every stream.",
            "Use GPU kernel/memcpy interval union or span, not summed overlapping kernel durations.",
            "No extra per-step CUDA synchronization or per-layer hooks were inserted.",
            "OFF includes host instrumentation; ON may perturb arrival batching. Compare actual identities/counts/modes.",
            "This is profiling diagnosis, not an OFF/ON speedup or counterfactual removable-time claim."])
    return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static archive/source check only; never starts CUDA")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_frozen_sources(args.archive, args.source_dir), indent=2))

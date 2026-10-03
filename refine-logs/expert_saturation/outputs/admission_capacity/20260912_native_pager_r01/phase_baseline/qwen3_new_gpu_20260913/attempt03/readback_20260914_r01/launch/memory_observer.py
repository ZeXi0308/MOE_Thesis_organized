"""Select equivalent allocator fields without changing Torch's global functions."""
import hashlib
import json
from pathlib import Path
import time

TORCH_MEMORY_SHA256 = "71a36cb635c0a076bf8cda0ae08c112d7b0e15d9d2fb21a0dee373e2967c4700"


def observer_read(cuda, device, peak, mode):
    if mode == "flat":
        return (cuda.max_memory_allocated if peak else cuda.memory_allocated)(device)
    if mode == "nested":
        return cuda.memory.memory_stats_as_nested_dict(device).get(
            "allocated_bytes", {}).get("all", {}).get("peak" if peak else "current", 0)
    raise ValueError("unknown memory observer mode")


def verify_values(cuda, device):
    source_sha = hashlib.sha256(Path(cuda.memory.__file__).read_bytes()).hexdigest()
    if source_sha != TORCH_MEMORY_SHA256:
        raise ValueError("installed Torch memory source differs from frozen source")
    samples = {key: [observer_read(cuda, device, peak, mode)
                    for mode in ("flat", "nested", "flat")]
               for key, peak in (("current", False), ("peak", True))}
    return dict(torch_memory_source_sha256=source_sha, samples_flat_nested_flat=samples,
                values_equal=all(len(set(values)) == 1 for values in samples.values()))


def capture_with_observer(runtime, engine, capture, output, *, mode, tag):
    if mode not in ("flat", "nested"):
        raise ValueError("unknown memory observer mode")
    prior = getattr(runtime, "memory_observer_mode", "flat")
    if prior != "flat" or engine.has_unfinished_requests() or engine.engine_core.engine_core.scheduler.requests:
        raise RuntimeError("observer comparison requires flat mode and a drained engine")
    device = next(iter(runtime.layers.values()))["state"].scratch_w13.device
    evidence = dict(tag=tag, mode=mode, prior_mode=prior, warmup_mode="flat",
                    check_start_s=time.perf_counter(), status="checking")
    try:
        evidence.update(verify_values(runtime.torch.cuda, device), check_end_s=time.perf_counter())
        if not evidence["values_equal"]:
            raise RuntimeError("allocator values changed or observer fields differ")
        runtime.memory_observer_mode = mode
        evidence.update(capture_start_s=time.perf_counter(), status="capturing")
        result = capture()
        evidence.update(status="complete", capture_status=result.get("status"))
        return result
    except BaseException as exc:
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        runtime.memory_observer_mode = prior
        evidence.update(wrapper_end_s=time.perf_counter(), restored_mode=prior,
            scope="Only three expert-apply allocator reads switch mode; pre-capture equality queries perform no allocations; raw wall unchanged")
        (output / "memory_observer.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")

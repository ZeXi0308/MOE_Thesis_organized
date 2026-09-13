"""Optional common nested allocator reads after the existing continuous patch."""
import hashlib

CONTINUOUS_RUNNER_SHA256 = "6f37900bc72012391fd5ca30e805d71a625ec093af98c90a7ceb31fcb7b9a722"


def patch_runner(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != CONTINUOUS_RUNNER_SHA256:
        raise ValueError("nested observer requires the existing continuous runner patch")
    edits = [
        ('    args = p.parse_args()',
         '    p.add_argument("--nested-memory-observer", action="store_true")\n    args = p.parse_args()'),
        ('        before = check_gpu(out, "before_initialization")',
         '        if args.nested_memory_observer and not args.continuous_admission:\n            raise ValueError("nested memory observer requires continuous admission")\n        before = check_gpu(out, "before_initialization")'),
        ('        scheduler = core.scheduler\n        original = scheduler.schedule',
         '''        scheduler = core.scheduler
        if args.nested_memory_observer:
            from memory_observer import verify_values
            if engine.has_unfinished_requests() or scheduler.requests:
                raise RuntimeError("common observer selection requires a drained engine")
            check_start_s = time.perf_counter()
            device = next(iter(runtime.layers.values()))["state"].scratch_w13.device
            equality = verify_values(torch.cuda, device)
            equality.update(check_start_s=check_start_s, check_end_s=time.perf_counter(),
                mode="nested", warmup_mode="nested", measurement_mode="nested",
                scope="One source/field check after model load, before all common warmups and measurement")
            dump(out / "memory_observer.json", equality)
            if not equality["values_equal"]:
                raise RuntimeError("allocator values changed or observer fields differ")
            runtime.memory_observer_mode = "nested"
        original = scheduler.schedule'''),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"continuous nested observer anchor count: {old[:70]!r}")
        source = source.replace(old, new)
    return source

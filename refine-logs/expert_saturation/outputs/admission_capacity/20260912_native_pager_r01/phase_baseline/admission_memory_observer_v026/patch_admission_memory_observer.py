"""Four identical D episodes: flat, nested, nested, flat allocator observations."""
import hashlib

RUNNER_SHA256 = "29d16549131480a5bd439f653ed002213b57a316ef32b70ac526497e1b140d7c"
EXPERT_GROUPS_SHA256 = "d22fa5d0b1ced552556fab5209f4e55a4ecae24b658f043ef0bc35e7fd90f6f2"
TAGS = ("flat_before", "nested_1", "nested_2", "flat_after")
MODES = ("flat", "nested", "nested", "flat")


def patch_runner(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != RUNNER_SHA256:
        raise ValueError("memory observer patch requires the executed admission runner")
    source = source.replace('--same-engine-admission-width-baseline', '--same-engine-admission-memory-observer')
    source = source.replace('same_engine_admission_width_baseline', 'same_engine_admission_memory_observer')
    edits = [
        ('p.add_argument("--same-engine-admission-memory-observer", action="store_true",',
         'p.add_argument("--same-engine-admission-memory-observer", action="store_true", required=True,'),
        ('help="Immediate32/admit2_32/admit2_release64 and reverse; all three requests complete"',
         'help="Four identical admit2_32 episodes with flat/nested/nested/flat allocator observation"'),
        ("        args.admission_width_order = ('immediate32', 'admit2_32', 'admit2_release64', 'admit2_release64', 'admit2_32', 'immediate32')",
         f"        args.admission_width_order = ('admit2_32',) * 4\n        args.memory_observer_tags = {TAGS!r}\n        args.memory_observer_modes = {MODES!r}"),
        ('        args.admission_width_ordering = "declared_first_block_then_reverse"',
         '        args.admission_width_ordering = "identical_D_flat_nested_nested_flat"'),
        ('            count = 6', '            count = 4'),
        ('        from admission_width import apply_admission',
         '        from admission_width import apply_admission\n        from memory_observer import capture_with_observer'),
        ('                variant = cell_arm', '                variant = args.memory_observer_tags[repeat]'),
        ('''                raw = (injection_episode(cell_chunk, phase, release_after_old) if injection else
                       capture_episode(engine, workload, config, regime="steady", arrival_scale=1,
                                       run_id="measurement", max_seconds=args.max_seconds, runtime_observer=observer))''',
         '''                raw = capture_with_observer(runtime, engine,
                    lambda: injection_episode(cell_chunk, phase, release_after_old), episode_out,
                    mode=args.memory_observer_modes[repeat], tag=args.memory_observer_tags[repeat])'''),
        ('episodes[-1].update(arm=cell_arm, chunk=cell_chunk, block=repeat // 3, order=repeat,\n                    order_in_block=repeat % 3, ordering=args.admission_width_ordering)',
         'episodes[-1].update(arm=cell_arm, chunk=cell_chunk, block=repeat // 2, order=repeat,\n                    order_in_block=repeat % 2, ordering=args.admission_width_ordering,\n                    memory_observer_tag=args.memory_observer_tags[repeat], memory_observer_mode=args.memory_observer_modes[repeat])'),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"memory observer anchor count: {old[:70]!r}")
        source = source.replace(old, new)
    return source


def patch_expert_groups(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != EXPERT_GROUPS_SHA256:
        raise ValueError("memory observer patch requires the executed expert groups source")
    if source.count("torch.cuda.memory_allocated(x.device)") != 2 or source.count("torch.cuda.max_memory_allocated(x.device)") != 1:
        raise ValueError("expected exactly three expert allocator observations")
    source = source.replace("def _apply(self, method, layer, x, weights, ids):",
        "from memory_observer import observer_read\n\ndef _apply(self, method, layer, x, weights, ids):")
    for name, peak in (("memory_allocated", False), ("max_memory_allocated", True)):
        source = source.replace(f"torch.cuda.{name}(x.device)",
            f"observer_read(torch.cuda, x.device, {peak!r}, getattr(self, 'memory_observer_mode', 'flat'))")
    return source

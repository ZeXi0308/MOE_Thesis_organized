"""Six identical admit2_32 episodes; only the middle four enable cProfile."""
import hashlib

SOURCE_SHA256 = "29d16549131480a5bd439f653ed002213b57a316ef32b70ac526497e1b140d7c"
TAGS = ("plain_before", "cpu_profile_1", "cpu_profile_2", "cpu_profile_3", "cpu_profile_4", "plain_after")


def patch_runner(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("CPU profile patch requires the executed admission runner")
    source = source.replace('--same-engine-admission-width-baseline', '--same-engine-admission-cpu-profile')
    source = source.replace('same_engine_admission_width_baseline', 'same_engine_admission_cpu_profile')
    edits = [
        ('p.add_argument("--same-engine-admission-cpu-profile", action="store_true",',
         'p.add_argument("--same-engine-admission-cpu-profile", action="store_true", required=True,'),
        ('help="Immediate32/admit2_32/admit2_release64 and reverse; all three requests complete"',
         'help="Six identical admit2_32 episodes, middle four with measurement-only current-thread CPU profiling"'),
        ("        args.admission_width_order = ('immediate32', 'admit2_32', 'admit2_release64', 'admit2_release64', 'admit2_32', 'immediate32')",
         f"        args.admission_width_order = ('admit2_32',) * 6\n        args.cpu_profile_tags = {TAGS!r}"),
        ('        args.admission_width_ordering = "declared_first_block_then_reverse"',
         '        args.admission_width_ordering = "identical_D_plain_profile4_plain"'),
        ('        from admission_width import apply_admission',
         '        from admission_width import apply_admission\n        from cpu_capture_profile import profile_capture'),
        ('                variant = cell_arm', '                variant = args.cpu_profile_tags[repeat]'),
        ('''                raw = (injection_episode(cell_chunk, phase, release_after_old) if injection else
                       capture_episode(engine, workload, config, regime="steady", arrival_scale=1,
                                       run_id="measurement", max_seconds=args.max_seconds, runtime_observer=observer))''',
         '''                raw = profile_capture(
                    lambda: injection_episode(cell_chunk, phase, release_after_old), episode_out,
                    enabled=1 <= repeat <= 4, tag=args.cpu_profile_tags[repeat])'''),
        ('                    order_in_block=repeat % 3, ordering=args.admission_width_ordering)',
         '                    order_in_block=repeat % 3, ordering=args.admission_width_ordering,\n                    profile_tag=args.cpu_profile_tags[repeat], cpu_profile_enabled=1 <= repeat <= 4)'),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"CPU profile anchor count: {old[:70]!r}")
        source = source.replace(old, new)
    return source

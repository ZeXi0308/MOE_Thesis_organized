"""Derive an expert-only chunk curve from the executed v0.26 comparison runner."""

RANDOMIZATION_SEED = 2026091303
# random.Random(seed) shuffles the first block; the second reverses it.
CHUNK_BLOCKS = ((32, 16, 8), (8, 16, 32))


def patch_expert_chunk_curve(source: str) -> str:
    """Return new runner source; preserve all map/capture/reset/observer helpers."""
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"expert chunk curve anchor count: {old[:70]!r}")
        source = source.replace(old, new)

    replace('--same-engine-execution-comparison', '--same-engine-expert-chunk-curve')
    if 'same_engine_execution_comparison' not in source:
        raise ValueError("executed comparison runner required")
    source = source.replace('same_engine_execution_comparison', 'same_engine_expert_chunk_curve')
    replace('help="Token/expert/expert/token; common token prefix, fixed chunk16"',
            'help="Six expert chunk8/16/32 episodes; randomized first block and its reverse"')
    order = tuple(chunk for block in CHUNK_BLOCKS for chunk in block)
    replace('    args = p.parse_args()', f'''    args = p.parse_args()
    if args.same_engine_expert_chunk_curve:
        args.chunk_curve_seed = {RANDOMIZATION_SEED}
        args.chunk_curve_ordering = "randomized_first_block_then_reverse"
        args.chunk_curve_order = {order!r}''')
    replace('"execution comparison requires cap16/requests3/tokens64/KV512MiB/chunk16/none/early"',
            '"expert curve requires cap16/requests3/tokens64/KV512MiB/injection-placeholder16/none/early"')
    replace('scope="same-source c10 token/expert execution comparison; natural 32/32/128 prefixes"',
            'scope="same-source c10 expert chunk curve; natural 32/32/128 prefixes"')
    replace('run_id.endswith("/warmup_injection_expert16")',
            'run_id.endswith(("/warmup_injection_expert8", "/warmup_injection_expert16", "/warmup_injection_expert32"))')
    replace('        if args.same_engine_expert_chunk_curve:\n            count = 4',
            '        if args.same_engine_expert_chunk_curve:\n            count = 6')
    replace('            cell_execution = args.execution',
            '            cell_execution = args.execution\n            cell_chunk = args.injection_chunk')
    replace('''                cell_execution = ("token", "expert", "expert", "token")[repeat]
                variant = cell_execution''', '''                cell_execution = "expert"
                cell_chunk = args.chunk_curve_order[repeat]
                variant = f"expert_chunk{cell_chunk}"''')
    replace('''                    warmups = [(16, False, label, "none") for label in
                               ("token16_before", "expert16", "token16_after")]''', '''                    warmups = [(chunk, False, label, "none") for chunk, label in
                               [(16, "token16_before"), (8, "expert8"), (16, "expert16"),
                                (32, "expert32"), (16, "token16_after")]]''')
    replace('"expert:" if label == "expert16" else "token:"',
            '"expert:" if label in ("expert8", "expert16", "expert32") else "token:"')
    replace('raw = (injection_episode(args.injection_chunk, phase, release_after_old) if injection else',
            'raw = (injection_episode(cell_chunk, phase, release_after_old) if injection else')
    replace('                measured_execution=cell_execution, group_retention=retention_mode,',
            '                measured_execution=cell_execution, measured_chunk=cell_chunk, group_retention=retention_mode,')
    replace('''            if variant:
                episodes[-1]["variant"] = variant''', '''            if args.same_engine_expert_chunk_curve:
                episodes[-1].update(chunk=cell_chunk, block=repeat // 3, order=repeat,
                    order_in_block=repeat % 3, randomization_seed=args.chunk_curve_seed)
            if variant:
                episodes[-1]["variant"] = variant''')
    return source

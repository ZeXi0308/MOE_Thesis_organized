"""Prepare a v0.26 token/expert ABBA runner; no execution or shared-source edits."""


def patch_execution_comparison(source: str) -> str:
    """Patch the frozen retention_lifecycle_performance runner, with exact anchors."""
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"execution comparison anchor count: {old[:70]!r}")
        source = source.replace(old, new)

    replace('choices=[24, 64]', 'choices=[16, 24, 64]')
    replace('choices=[0, 8, 128]', 'choices=[0, 8, 16, 128]')
    replace('    args = p.parse_args()', '''    p.add_argument("--same-engine-execution-comparison", action="store_true",
                   help="Token/expert/expert/token; common token prefix, fixed chunk16")
    args = p.parse_args()''')
    replace('        matrices = (args.same_engine_diagnostic,',
            '        matrices = (args.same_engine_execution_comparison, args.same_engine_diagnostic,')
    replace('        observed_run = args.same_engine_runtime_diagnostic or args.same_engine_retention_lifecycle_comparison',
            '        observed_run = args.same_engine_execution_comparison or args.same_engine_runtime_diagnostic or args.same_engine_retention_lifecycle_comparison')
    replace('        if injection and (args.execution != "expert" or args.expert_cap != 24',
            '''        if args.same_engine_execution_comparison and (
                args.expert_cap != 16 or args.requests != 3 or args.token_budget != 64
                or args.kv_bytes != 536870912 or args.injection_chunk != 16
                or args.group_retention != "none" or args.group_retention_order != "early"):
            raise ValueError("execution comparison requires cap16/requests3/tokens64/KV512MiB/chunk16/none/early")
        if injection and not args.same_engine_execution_comparison and (args.execution != "expert" or args.expert_cap != 24''')
    replace('        if len(frozen["source_requests"]) < args.requests:', '''        if args.same_engine_execution_comparison:
            indices = [4, 5, 10]
            frozen = dict(frozen,
                source_requests=[frozen["source_requests"][i] for i in indices],
                actual_prompt_token_ids=[frozen["actual_prompt_token_ids"][i] for i in indices])
        if len(frozen["source_requests"]) < args.requests:''')
    replace('        dump(out / "workload.json", workload)', '''        if args.same_engine_execution_comparison:
            workload.update(source_indices=[4, 5, 10],
                scope="same-source c10 token/expert execution comparison; natural 32/32/128 prefixes")
        dump(out / "workload.json", workload)''')
    replace('                     + (["wisp_expert_groups.py"] if args.execution == "expert"',
            '                     + (["wisp_expert_groups.py"] if args.same_engine_execution_comparison or args.execution == "expert"')
    replace('        if args.execution == "expert" or args.warmup_executions:',
            '        if args.same_engine_execution_comparison or args.execution == "expert" or args.warmup_executions:')
    replace('        runtime.apply = execution_apply["token" if args.warmup_executions and not injection else args.execution]',
            '        runtime.apply = execution_apply["token" if args.same_engine_execution_comparison or (args.warmup_executions and not injection) else args.execution]')
    replace('            max_num_seqs=16, max_num_batched_tokens=args.token_budget,',
            '            max_num_seqs=3 if args.same_engine_execution_comparison else 16, max_num_batched_tokens=args.token_budget,')
    replace('        def injection_episode(chunk, run_id, release_after_old=False):', '''        comparison_prestate = None
        def injection_episode(chunk, run_id, release_after_old=False):
            if args.same_engine_execution_comparison:
                runtime.apply = execution_apply["token"]''')
    replace('            def before_add(scheduler, action, rows, now):', '''            def before_add(scheduler, action, rows, now):
                nonlocal comparison_prestate''')
    replace('                action["snapshot_end_s"] = now()', '''                action["snapshot_end_s"] = now()
                if args.same_engine_execution_comparison and run_id.endswith("/measurement"):
                    assert runtime.apply == execution_apply["token"]
                    ids = {r["internal_request_id"]: rid for rid, r in rows.items()
                           if "internal_request_id" in r}
                    logical = dict(action["before"])
                    logical["requests"] = {ids[rid]: dict(
                        {k: v for k, v in r.items() if k != "kv_block_ids"},
                        kv_block_counts=[len(g) for g in r["kv_block_ids"]])
                        for rid, r in logical["requests"].items()}
                    for key in ("running", "waiting"):
                        logical[key] = [ids[rid] for rid in logical[key]]
                    digest = hashlib.sha256(json.dumps(logical, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest()
                    action.update(prestate_sha256=digest,
                        prestate_scope="Source-ID-normalized tokens/request/KV counts/pager metadata; physical KV IDs retained separately; no KV tensor comparison")
                    if comparison_prestate is not None and digest != comparison_prestate:
                        raise RuntimeError("common token pre-injection metadata differs")
                    comparison_prestate = digest
                    runtime.apply = execution_apply[cell_execution]
                    action.update(execution_before="token", execution_after=cell_execution,
                        execution_applied_s=now(), execution_boundary="before event add and next native schedule/KV allocation")
                elif args.same_engine_execution_comparison:
                    assert runtime.apply == execution_apply["token"]
                    execution = "expert" if run_id.endswith("/warmup_injection_expert16") else "token"
                    runtime.apply = execution_apply[execution]
                    action.update(execution_before="token", execution_after=execution,
                        execution_applied_s=now(), execution_boundary="before event add and next native schedule/KV allocation")''')
    start = source.index('            return capture_episode(engine, workload, config, regime="steady", arrival_scale=1,\n                run_id=run_id,')
    end = source.index('\n        episodes, reference = [], None', start)
    capture = source[start:end]
    replace(capture, '            try:\n' + '\n'.join('    ' + line for line in capture.splitlines())
            + '\n            finally:\n                if args.same_engine_execution_comparison:\n                    runtime.apply = execution_apply["token"]')
    replace('        if observed_run:\n            cycle = dict(', '''        if args.same_engine_execution_comparison:
            count = 4
        if observed_run:
            cycle = dict(''')
    replace('            release_after_old = retention_run or variant == "release8" or args.same_engine_runtime_diagnostic', '''            cell_execution = args.execution
            if args.same_engine_execution_comparison:
                cell_execution = ("token", "expert", "expert", "token")[repeat]
                variant = cell_execution
                runtime.apply = execution_apply["token"]
            release_after_old = retention_run or variant == "release8" or args.same_engine_runtime_diagnostic''')
    replace('                if (args.same_engine_retention_order_comparison or observed_run',
            '                if (args.same_engine_retention_order_comparison or (observed_run and not args.same_engine_execution_comparison)')
    replace('                for chunk, warmup_release, label, mode in warmups:', '''                if args.same_engine_execution_comparison:
                    warmups = [(16, False, label, "none") for label in
                               ("token16_before", "expert16", "token16_after")]
                for chunk, warmup_release, label, mode in warmups:''')
    replace('                injection_warmup_sequence=["expert:" + label for _, _, label, _ in warmups] if injection else None,',
            '                injection_warmup_sequence=[(("expert:" if label == "expert16" else "token:") if args.same_engine_execution_comparison else "expert:") + label for _, _, label, _ in warmups] if injection else None,')
    replace('            runtime.apply = execution_apply[args.execution]',
            '            runtime.apply = execution_apply["token" if args.same_engine_execution_comparison else args.execution]')
    replace('                measured_execution=args.execution, group_retention=retention_mode,',
            '                measured_execution=cell_execution, group_retention=retention_mode,')
    replace('            episodes.append(dict(phase=phase, status=raw["status"], raw_path=str(episode_out / "raw.json"),',
            '            episodes.append(dict(phase=phase, execution=cell_execution, status=raw["status"], raw_path=str(episode_out / "raw.json"),')
    return source

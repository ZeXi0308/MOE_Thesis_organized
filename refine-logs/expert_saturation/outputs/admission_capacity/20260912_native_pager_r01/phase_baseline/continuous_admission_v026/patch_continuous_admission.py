"""Add one configurable clock episode to the frozen admission runner; no matrix."""
import hashlib

RUNNER_SHA256 = "29d16549131480a5bd439f653ed002213b57a316ef32b70ac526497e1b140d7c"
CAPTURE_SHA256 = "54af498cd1482b717846cf61c5b3748ee5a3e6508662e06b5a1aa2aa887b05cc"


def _patch(source, digest, edits):
    if hashlib.sha256(source.encode()).hexdigest() != digest:
        raise ValueError("continuous patch requires the frozen admission source")
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"continuous anchor count: {old[:70]!r}")
        source = source.replace(old, new)
    return source


def patch_native_capture(source: str) -> str:
    return _patch(source, CAPTURE_SHA256, [
        ('from admission_width import observe_waiting', 'from admission_width import observe_waiting\nfrom continuous_admission import choose_phase'),
        ('    if runtime_observer is not None and not cpu_diagnostics:', '''    if config.get("phase_prefill_mode"):
        cpu_diagnostics = True
        if event_arrival or policy_not_static(config) or config.get("allow_preemption", False):
            raise ValueError("continuous phase mode requires clock arrivals and nonpreemptive static admission")
    if runtime_observer is not None and not cpu_diagnostics:'''.replace('policy_not_static(config)', 'config.get("policy", "static") != "static"')),
        ('        if event_arrival and (engine.vllm_config.speculative_config is not None',
         '        if (event_arrival or config.get("phase_prefill_mode")) and (engine.vllm_config.speculative_config is not None'),
        ('            result = original(*args, **kwargs)', '''            phase_decision = None
            if config.get("phase_prefill_mode"):
                phase_decision = choose_phase(scheduler, config["phase_prefill_mode"], now)
                engine_calls[-1]["phase_prefill"] = phase_decision
                existing_decode_ids = phase_decision["ready_decode_ids"]
                scheduler.scheduler_config.long_prefill_token_threshold = phase_decision["chosen_threshold"]
                phase_decision["applied_s"] = now()
            result = original(*args, **kwargs)'''),
        ('            if event_actions and config.get("admission_width_mode"):', '''            if phase_decision is not None:
                scheduler_steps[-1]["phase_prefill"] = phase_decision
                phase_decision["actual_prefill_rows"] = sum(r["prefill_tokens"] for r in scheduled)
                phase_decision["actual_decode_rows"] = sum(r["decode_tokens"] for r in scheduled)
            if event_actions and config.get("admission_width_mode"):'''),
        ('            if decision is not None or event_arrival:',
         '            if decision is not None or event_arrival or phase_decision is not None:'),
        ('                if event_arrival and any(result.num_scheduled_tokens.get(rid) != 1 for rid in existing_decode_ids):',
         '                if (event_arrival or phase_decision is not None) and any(result.num_scheduled_tokens.get(rid) != 1 for rid in existing_decode_ids):'),
        ('            return result\n\n        scheduler.schedule = schedule',
         '            if phase_decision is not None:\n                phase_decision["status"] = "applied"\n            return result\n\n        scheduler.schedule = schedule'),
        ('                row["admission_s"] = now()', '''                row["admission_s"] = now()
                row["submission_lag_s"] = row["admission_s"] - row["arrival_s"]'''),
        ('                if event_arrival and internal_id not in scheduler.requests:',
         '                if (event_arrival or config.get("phase_prefill_mode")) and internal_id not in scheduler.requests:'),
    ])


def patch_runner(source: str) -> str:
    edits = [
        ('    args = p.parse_args()', '''    p.add_argument("--continuous-admission", action="store_true")
    p.add_argument("--admission-limit", type=int, choices=[2, 3], default=3)
    p.add_argument("--phase-policy", choices=["static32", "phase32"], default="static32")
    p.add_argument("--warmup-source-indices", type=int, nargs="+")
    p.add_argument("--warmup-prompt-tokens", type=int, default=128)
    for name, kind in (("source-indices", int), ("prompt-lengths", int), ("output-lengths", int), ("arrival-times", float)):
        p.add_argument("--" + name, type=kind, nargs=8)
    args = p.parse_args()'''),
        ('        matrices = (args.same_engine_admission_width_baseline,', '        matrices = (args.continuous_admission, args.same_engine_admission_width_baseline,'),
        ('        observed_run = args.same_engine_admission_width_baseline or args.same_engine_runtime_diagnostic or args.same_engine_retention_lifecycle_comparison',
         '        observed_run = args.continuous_admission or args.same_engine_admission_width_baseline or args.same_engine_runtime_diagnostic or args.same_engine_retention_lifecycle_comparison'),
        ('        before = check_gpu(out, "before_initialization")', '''        if args.continuous_admission and (args.requests != 8 or injection or args.execution != "expert"
                or args.expert_cap != 16 or args.kv_bytes != 536870912 or args.token_budget != 64
                or args.warmup_full or args.warmup_executions):
            raise ValueError("continuous mode requires eight requests/expert/cap16/KV512MiB/token64/no event")
        before = check_gpu(out, "before_initialization")'''),
        ('        frozen = json.loads((args.prepared / "workload.json").read_text())',
         '        from continuous_admission import prepare_workload, prepare_warmup\n        frozen = json.loads((args.prepared / "workload.json").read_text())'),
        ('        if any(len(ids) < args.prompt_tokens for ids in prompts):',
         '        if not args.continuous_admission and any(len(ids) < args.prompt_tokens for ids in prompts):'),
        ('        dump(out / "workload.json", workload)', '''        if args.continuous_admission:
            workload = prepare_workload(frozen, args)
            geometry_workload = prepare_warmup(frozen, args)
        dump(out / "workload.json", workload)'''),
        ('            max_model_len=max(512, args.prompt_tokens + args.output_tokens),',
         '            max_model_len=max(512, workload.get("max_request_tokens", args.prompt_tokens + args.output_tokens), geometry_workload["max_request_tokens"] if args.continuous_admission else 0),'),
        ('max_num_seqs=3 if args.same_engine_admission_width_baseline else 16',
         'max_num_seqs=3 if args.continuous_admission or args.same_engine_admission_width_baseline else 16'),
        ('        set_empty_admission_cap(engine, args.requests)',
         '        set_empty_admission_cap(engine, 3 if args.continuous_admission else args.requests)'),
        ('        if injection:\n            source_ids =', '''        if args.continuous_admission:
            config.update(cap=3, allow_preemption=False, phase_prefill_mode="static32",
                          output_tokens_by_request=workload["output_tokens_by_request"])
        if injection:
            source_ids ='''),
        ('        if observed_run:\n            cycle = dict(', '        if args.continuous_admission:\n            count = 1\n        if observed_run:\n            cycle = dict('),
        ('            warmup = dict(workload, source_requests=workload["source_requests"][:1],',
         '            warmup = geometry_workload if args.continuous_admission else dict(workload, source_requests=workload["source_requests"][:1],'),
        ('            episode_name = f"repeat_{repeat}" + ("_" + variant if variant else "")', '''            if args.continuous_admission:
                cell_arm = f"cap{args.admission_limit}_{args.phase_policy}"
                variant = cell_arm
            episode_name = f"repeat_{repeat}" + ("_" + variant if variant else "")'''),
        ('                        runtime_observer=observer)\n                    name = phase + ".json"',
         '                        runtime_observer=observer, cpu_diagnostics=True)\n                    name = phase + ".json"'),
        ('            phase, step = prefix + "measurement", 0', '''            if args.continuous_admission:
                phase, step = prefix + "warmup_phase32", 0
                warm_raw = capture_episode(engine, geometry_workload,
                    dict(config, output_tokens=2, output_tokens_by_request={}, phase_prefill_mode="phase32"),
                    regime="steady", arrival_scale=1, run_id=phase, max_seconds=args.max_seconds,
                    cpu_diagnostics=True, runtime_observer=observer)
                dump(episode_out / "warmup_phase32.json", warm_raw)
                if warm_raw["status"] != "COMPLETE":
                    raise RuntimeError("common phase32 warmup did not complete")
                dump(episode_out / "post_warmup_reset.json", reset_drained_pager(engine, runner, runtime))
                set_empty_admission_cap(engine, args.admission_limit)
                config.update(cap=args.admission_limit, phase_prefill_mode=args.phase_policy)
            phase, step = prefix + "measurement", 0'''),
        ('                                       run_id="measurement", max_seconds=args.max_seconds, runtime_observer=observer))',
         '                                       run_id=phase, max_seconds=args.max_seconds, runtime_observer=observer, cpu_diagnostics=True))'),
    ]
    return _patch(source, RUNNER_SHA256, edits)

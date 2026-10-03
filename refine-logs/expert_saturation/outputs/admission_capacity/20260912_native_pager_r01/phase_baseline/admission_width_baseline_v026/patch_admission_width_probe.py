"""Derive I,D,R,R,D,I admission baselines from the executed expert curve."""
import hashlib

RUNNER_SHA256 = "555fb35fe5fd97eb4e8080804f3da14db2099c8733d1a649b06c5df8cf0e8160"
CAPTURE_SHA256 = "7e981d3ea2d0d206908ceaab568471d7923e2db0fc1b4f38fe6b2a73ebc5d9da"
ARMS = ("immediate32", "admit2_32", "admit2_release64", "admit2_release64", "admit2_32", "immediate32")


def _patch(source, digest, edits):
    if hashlib.sha256(source.encode()).hexdigest() != digest:
        raise ValueError("admission patch requires the executed frozen source")
    for old, new, count in edits:
        if source.count(old) != count:
            raise ValueError(f"admission anchor count: {old[:70]!r}")
        source = source.replace(old, new)
    return source


def patch_native_capture(source: str) -> str:
    return _patch(source, CAPTURE_SHA256, [
        ('from admission_feedback import AdmissionFeedback, apply_nonpreemptive_limit',
         'from admission_feedback import AdmissionFeedback, apply_nonpreemptive_limit\nfrom admission_width import observe_waiting', 1),
        ('                target_cap=target, running_before=running_before, waiting_before=waiting_before,',
         '                target_cap=target, actual_admission_cap=scheduler.max_num_running_reqs, running_before=running_before, waiting_before=waiting_before,', 1),
        ('            if decision is not None:\n                step = scheduler_steps[-1]', '''            if event_actions and config.get("admission_width_mode"):
                observe_waiting(scheduler, config["admission_width_mode"], event_id, targets,
                    rows, result.num_scheduled_tokens, scheduler_steps[-1])
            if decision is not None:
                step = scheduler_steps[-1]''', 1),
        ('                if action["threshold_before"] != 8 or action["remaining_prefill_tokens"] <= 0:', '''                expected_threshold = 32 if config.get("admission_width_mode") == "admit2_release64" else 8
                if action["threshold_before"] != expected_threshold or action["remaining_prefill_tokens"] <= 0:''', 1),
        ('raise ValueError("release baseline requires a remaining prefill under chunk8")',
         'raise ValueError("release requires its declared threshold and remaining prefill")', 1),
        ('                action.update(threshold_after=scheduler.scheduler_config.long_prefill_token_threshold,\n                              applied_s=now(), end_s=now())',
         '                action.update(threshold_after=scheduler.scheduler_config.long_prefill_token_threshold,\n                              effective_token_budget=scheduler.max_num_scheduled_tokens, applied_s=now(), end_s=now())', 1),
    ])


def patch_runner(source: str) -> str:
    oldflag = 'same_engine_expert_chunk_curve'
    return _patch(source, RUNNER_SHA256, [
        ('--same-engine-expert-chunk-curve', '--same-engine-admission-width-baseline', 1),
        (oldflag, 'same_engine_admission_width_baseline', source.count(oldflag)),
        ('help="Six expert chunk8/16/32 episodes; randomized first block and its reverse"',
         'help="Immediate32/admit2_32/admit2_release64 and reverse; all three requests complete"', 1),
        ('''        args.chunk_curve_seed = 2026091303
        args.chunk_curve_ordering = "randomized_first_block_then_reverse"
        args.chunk_curve_order = (32, 16, 8, 8, 16, 32)''', f'''        args.admission_width_order = {ARMS!r}
        args.admission_width_ordering = "declared_first_block_then_reverse"''', 1),
        ('from native_capture import capture_episode, set_empty_admission_cap',
         'from native_capture import capture_episode, set_empty_admission_cap\n        from admission_width import apply_admission', 1),
        ('scope="same-source c10 expert chunk curve; natural 32/32/128 prefixes"',
         'scope="same-source c10 native admission baseline; all three requests enter at the unchanged event time"', 1),
        ('        def injection_episode(chunk, run_id, release_after_old=False):', '''        def injection_episode(chunk, run_id, release_after_old=False):
            admission_mode = cell_arm if run_id.endswith("/measurement") else next(
                (m for m in ("immediate32", "admit2_32", "admit2_release64")
                 if run_id.endswith("/warmup_injection_" + m)), "immediate32")
            episode_raw = None
            assert scheduler.max_num_running_reqs == 3''', 1),
        ('run_id.endswith(("/warmup_injection_expert8", "/warmup_injection_expert16", "/warmup_injection_expert32"))',
         'run_id.endswith(("/warmup_injection_immediate32", "/warmup_injection_admit2_32", "/warmup_injection_admit2_release64"))', 1),
        ('                action["threshold_before"] = scheduler.scheduler_config.long_prefill_token_threshold',
         '                apply_admission(scheduler, action, admission_mode, now)\n                action["threshold_before"] = scheduler.scheduler_config.long_prefill_token_threshold', 1),
        ('return capture_episode(engine, workload, config, regime="steady", arrival_scale=1,',
         'episode_raw = capture_episode(engine, workload, dict(config, admission_width_mode=admission_mode), regime="steady", arrival_scale=1,', 1),
        ('                                       inject=bool(chunk), expected_first_tokens=2 + chunk))',
         '                                       inject=True, expected_first_tokens=2 + chunk if admission_mode == "immediate32" else 2))\n                return episode_raw', 1),
        ('            finally:\n                if args.same_engine_admission_width_baseline:', '''            finally:
                restoration = dict(drained=not engine.has_unfinished_requests(),
                    cap_before=scheduler.max_num_running_reqs,
                    threshold_before=scheduler.scheduler_config.long_prefill_token_threshold)
                scheduler.max_num_running_reqs = 3
                scheduler.scheduler_config.long_prefill_token_threshold = 32
                restoration.update(cap_after=3, threshold_after=32)
                if episode_raw is not None:
                    episode_raw["admission_restore"] = restoration
                if args.same_engine_admission_width_baseline:''', 1),
        ('            cell_chunk = args.injection_chunk',
         '            cell_chunk = args.injection_chunk\n            cell_arm = "immediate32"', 1),
        ('''                cell_chunk = args.chunk_curve_order[repeat]
                variant = f"expert_chunk{cell_chunk}"''', '''                cell_arm = args.admission_width_order[repeat]
                cell_chunk = 32
                variant = cell_arm''', 1),
        ('            release_after_old = retention_run or variant == "release8" or args.same_engine_runtime_diagnostic',
         '            release_after_old = cell_arm == "admit2_release64" if args.same_engine_admission_width_baseline else retention_run or variant == "release8" or args.same_engine_runtime_diagnostic', 1),
        ('''                    warmups = [(chunk, False, label, "none") for chunk, label in
                               [(16, "token16_before"), (8, "expert8"), (16, "expert16"),
                                (32, "expert32"), (16, "token16_after")]]''', '''                    warmups = [(chunk, label == "admit2_release64", label, "none") for chunk, label in
                               [(32, "token32_before"), (32, "immediate32"), (32, "admit2_32"),
                                (32, "admit2_release64"), (32, "token32_after")]]''', 1),
        ('label in ("expert8", "expert16", "expert32")',
         'label in ("immediate32", "admit2_32", "admit2_release64")', 1),
        ('                measured_execution=cell_execution, measured_chunk=cell_chunk, group_retention=retention_mode,',
         '                measured_execution=cell_execution, measured_chunk=cell_chunk, arm=cell_arm, group_retention=retention_mode,', 1),
        ('''                episodes[-1].update(chunk=cell_chunk, block=repeat // 3, order=repeat,
                    order_in_block=repeat % 3, randomization_seed=args.chunk_curve_seed)''', '''                episodes[-1].update(arm=cell_arm, chunk=cell_chunk, block=repeat // 3, order=repeat,
                    order_in_block=repeat % 3, ordering=args.admission_width_ordering)''', 1),
    ])

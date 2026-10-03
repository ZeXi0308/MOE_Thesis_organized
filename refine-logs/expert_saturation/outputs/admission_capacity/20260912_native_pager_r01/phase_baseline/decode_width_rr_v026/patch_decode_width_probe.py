"""Prepare eight static/RR cells from the executed expert chunk curve source."""
import hashlib

RUNNER_SHA256 = "555fb35fe5fd97eb4e8080804f3da14db2099c8733d1a649b06c5df8cf0e8160"
CAPTURE_SHA256 = "7e981d3ea2d0d206908ceaab568471d7923e2db0fc1b4f38fe6b2a73ebc5d9da"
ARMS = ("static8", "static16", "static32", "rr32", "rr32", "static32", "static16", "static8")


def _check(source, digest):
    if hashlib.sha256(source.encode()).hexdigest() != digest:
        raise ValueError("decode width patch requires the executed frozen source")


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"decode width anchor count: {old[:70]!r}")
    return source.replace(old, new)


def patch_native_capture(source: str) -> str:
    _check(source, CAPTURE_SHA256)
    edits = [
        ('from admission_feedback import AdmissionFeedback, apply_nonpreemptive_limit',
         'from admission_feedback import AdmissionFeedback, apply_nonpreemptive_limit\nfrom decode_width_rr import schedule_with_width'),
        ('            result = original(*args, **kwargs)', '''            width_decision = {}
            engine_calls[-1]["decode_width_rr"] = width_decision
            result = schedule_with_width(scheduler, original, *args,
                mode=config.get("decode_width_mode", "none"), decision=width_decision, now=now, **kwargs)
            declared_held = set(width_decision["held_request_ids"])
            if declared_held and not width_decision.get("held_state_verified"):
                raise RuntimeError("unverified held-decode exemption")'''),
        ('                target_cap=target, running_before=running_before, waiting_before=waiting_before,',
         '                decode_width_rr=width_decision, target_cap=target, running_before=running_before, waiting_before=waiting_before,'),
        ('                step["existing_decode_all_scheduled"] = not missing', '''                step["existing_decode_all_scheduled"] = not missing
                step["intentional_decode_held_request_ids"] = [internal_to_source[rid] for rid in declared_held]
                step["existing_decode_required_all_scheduled"] = not (set(missing) - declared_held)'''),
        ('                        missing or step["preempted_request_ids"]',
         '                        (set(missing) - declared_held) or step["preempted_request_ids"]'),
        ('for rid in existing_decode_ids):', 'for rid in existing_decode_ids if rid not in declared_held):'),
    ]
    for old, new in edits:
        source = _replace(source, old, new)
    return source


def patch_runner(source: str) -> str:
    _check(source, RUNNER_SHA256)
    source = source.replace('--same-engine-expert-chunk-curve', '--same-engine-decode-width-comparison')
    source = source.replace('same_engine_expert_chunk_curve', 'same_engine_decode_width_comparison')
    edits = [
        ('help="Six expert chunk8/16/32 episodes; randomized first block and its reverse"',
         'help="Eight static8/static16/static32/rr32 episodes in declared order then reverse"'),
        ('''        args.chunk_curve_seed = 2026091303
        args.chunk_curve_ordering = "randomized_first_block_then_reverse"
        args.chunk_curve_order = (32, 16, 8, 8, 16, 32)''', f'''        args.decode_width_order = {ARMS!r}
        args.decode_width_ordering = "declared_first_block_then_reverse"'''),
        ('"expert curve requires cap16/requests3/tokens64/KV512MiB/injection-placeholder16/none/early"',
         '"decode width comparison requires cap16/requests3/tokens64/KV512MiB/injection-placeholder16/none/early"'),
        ('scope="same-source c10 expert chunk curve; natural 32/32/128 prefixes"',
         'scope="same-source c10 static chunk versus pure-decode width2 round robin"'),
        ('        def injection_episode(chunk, run_id, release_after_old=False):', '''        def injection_episode(chunk, run_id, release_after_old=False):
            width_mode = "rr2" if args.same_engine_decode_width_comparison and (
                (run_id.endswith("/measurement") and cell_arm == "rr32")
                or run_id.endswith("/warmup_injection_rr32")) else "none"'''),
        ('"/warmup_injection_expert32"))', '"/warmup_injection_expert32", "/warmup_injection_rr32"))'),
        ('return capture_episode(engine, workload, config, regime="steady", arrival_scale=1,',
         'return capture_episode(engine, workload, dict(config, decode_width_mode=width_mode), regime="steady", arrival_scale=1,'),
        ('        if args.same_engine_decode_width_comparison:\n            count = 6',
         '        if args.same_engine_decode_width_comparison:\n            count = 8'),
        ('            cell_chunk = args.injection_chunk', '            cell_chunk = args.injection_chunk\n            cell_arm = "static"'),
        ('''                cell_chunk = args.chunk_curve_order[repeat]
                variant = f"expert_chunk{cell_chunk}"''', '''                cell_arm = args.decode_width_order[repeat]
                cell_chunk = 32 if cell_arm == "rr32" else int(cell_arm.removeprefix("static"))
                variant = cell_arm'''),
        ('(32, "expert32"), (16, "token16_after")', '(32, "expert32"), (32, "rr32"), (16, "token16_after")'),
        ('label in ("expert8", "expert16", "expert32")', 'label in ("expert8", "expert16", "expert32", "rr32")'),
        ('                measured_execution=cell_execution, measured_chunk=cell_chunk, group_retention=retention_mode,',
         '                measured_execution=cell_execution, measured_chunk=cell_chunk, arm=cell_arm, group_retention=retention_mode,'),
        ('''                episodes[-1].update(chunk=cell_chunk, block=repeat // 3, order=repeat,
                    order_in_block=repeat % 3, randomization_seed=args.chunk_curve_seed)''', '''                episodes[-1].update(arm=cell_arm, chunk=cell_chunk,
                    decode_width_mode="rr2" if cell_arm == "rr32" else "none",
                    block=repeat // 4, order=repeat, order_in_block=repeat % 4,
                    ordering=args.decode_width_ordering)'''),
    ]
    for old, new in edits:
        source = _replace(source, old, new)
    return source

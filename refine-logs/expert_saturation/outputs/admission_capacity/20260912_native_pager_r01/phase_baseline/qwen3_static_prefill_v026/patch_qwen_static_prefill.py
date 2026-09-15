"""CPU source patch for fixed Qwen32/16/16/32; requires prior numerical qualification."""
import hashlib

RUNNER_SHA256 = "2471381a19def73fd489a3592982ddee91c33f1d6f2b2c9576afc890aa2ddd62"
CAPTURE_SHA256 = "df3527fcd7a829b5bca065de1e902b8b4f63a025dd596c6caaef444dd4c253d2"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Qwen static source anchor count: {old[:80]!r}")
    return source.replace(old, new)


def patch_native_capture(source):
    if hashlib.sha256(source.encode()).hexdigest() != CAPTURE_SHA256:
        raise ValueError("requires the executed continuous capture")
    return replace_once(source, "from continuous_admission import choose_phase",
                        "from qwen_static_prefill import choose_phase")


def patch_runner(source):
    if hashlib.sha256(source.encode()).hexdigest() != RUNNER_SHA256:
        raise ValueError("requires the frozen Qwen qualification runner")
    source = source.replace("qwen_qualification", "qwen_static_prefill").replace("--qwen-qualification", "--qwen-static-prefill")
    edits = [
        ('                or args.prefill_limit != 32 or args.arrival_interval != 0.05 or not args.verify_kernel',
         '                or args.prefill_limit != 32 or args.arrival_interval != 0.05 or args.verify_kernel'),
        ('Qwen qualification requires 4 requests/cap48/expert/KV512MiB/token64/8out/prefill32/50ms/verify/episode trace; no event or full warmup',
         'Qwen static comparison requires cap48/expert/KV512MiB/token64/maxseq4, fixed input, episode trace, no verification/event/other warmup; legacy CLI defaults are placeholders'),
        ('        prompts = frozen["actual_prompt_token_ids"][:args.requests]\n'
         '        if any(len(ids) < args.prompt_tokens for ids in prompts):\n'
         '            raise ValueError("requested prompt exceeds prepared input")\n'
         '        workload = dict(source_requests=frozen["source_requests"][:args.requests],\n'
         '                        actual_prompt_token_ids=[ids[:args.prompt_tokens] for ids in prompts],\n'
         '                        arrival_traces_s={"steady": [i * args.arrival_interval for i in range(args.requests)]})',
         '        from qwen_static_prefill import CHUNKS, prepare_workload\n'
         '        workload = prepare_workload(frozen)\n'
         '        prompts = workload["actual_prompt_token_ids"]'),
        ('                      "memory_observer.py", "qwen_serial_loader.py", "serial_shards.py"]',
         '                      "memory_observer.py", "qwen_serial_loader.py", "serial_shards.py", "qwen_static_prefill.py"]'),
        ('        if args.verify_kernel:\n            pager.enable_validation()',
         '        assert not args.verify_kernel  # No reference copies/compute in warmup or measurement.'),
        ('        config = dict(output_tokens=args.output_tokens, cap=args.requests,',
         '        config = dict(output_tokens=args.output_tokens, output_tokens_by_request=workload["output_tokens_by_request"], cap=args.requests,'),
        ('        count = 1\n', '        count = len(CHUNKS)\n'),
        ('            variant = "qwen_static_prefill"\n            cell_arm = variant\n            cell_chunk = 32\n'
         '            episode_name = f"repeat_{repeat}_qwen_static_prefill"',
         '            cell_chunk = CHUNKS[repeat]\n            variant = cell_arm = f"static{cell_chunk}"\n'
         '            episode_name = f"repeat_{repeat}_{cell_arm}"'),
        ('            phase, step = prefix + "warmup", 0\n'
         '            warmup = dict(workload, source_requests=workload["source_requests"][:1],\n'
         '                actual_prompt_token_ids=workload["actual_prompt_token_ids"][:1],\n'
         '                arrival_traces_s={"steady": [0.0]})\n'
         '            raw = capture_episode(engine, warmup, dict(config, output_tokens=2, output_tokens_by_request={}),\n'
         '                regime="steady", arrival_scale=1, run_id=phase, max_seconds=args.max_seconds,\n'
         '                cpu_diagnostics=same_engine, runtime_observer=observer)\n'
         '            dump(episode_out / "warmup.json", raw)\n'
         '            if raw["status"] != "COMPLETE":\n'
         '                raise RuntimeError("warmup incomplete: " + str(raw["error"]))',
         '''            warmup = dict(workload, arrival_traces_s={"steady": [0.0] * 4}, output_tokens_by_request={})
            for warmup_chunk in (16, 32):
                phase, step = prefix + f"warmup_static{warmup_chunk}", 0
                raw = capture_episode(engine, warmup,
                    dict(config, output_tokens=2, output_tokens_by_request={}, phase_prefill_mode=f"static{warmup_chunk}"),
                    regime="steady", arrival_scale=1, run_id=phase, max_seconds=args.max_seconds,
                    cpu_diagnostics=True, runtime_observer=observer)
                dump(episode_out / f"warmup_static{warmup_chunk}.json", raw)
                if raw["status"] != "COMPLETE":
                    raise RuntimeError("warmup incomplete: " + str(raw["error"]))
            post_warmup_reset = reset_drained_pager(engine, runner, runtime)
            post_warmup_reset.update(common_warmup_sequence=[16, 32], warmup_outputs_per_request=2,
                warmup_arrivals_s=[0.0] * 4, reference_validation_enabled=False)
            dump(episode_out / "post_warmup_reset.json", post_warmup_reset)'''),
        ('            scheduler.scheduler_config.long_prefill_token_threshold = 32 if injection else args.prefill_limit',
         '            scheduler.scheduler_config.long_prefill_token_threshold = cell_chunk\n'
         '            config["phase_prefill_mode"] = cell_arm'),
        ('                measured_prefill_limit=scheduler.scheduler_config.long_prefill_token_threshold, applied_on_drained_engine=True,',
         '                measured_prefill_limit=scheduler.scheduler_config.long_prefill_token_threshold, applied_on_drained_engine=True,\n'
         '                common_warmup_sequence=[16, 32], reference_validation_enabled=False,'),
        ('            if variant:\n                episodes[-1]["variant"] = variant',
         '            episodes[-1].update(arm=cell_arm, chunk=cell_chunk, block=repeat // 2, order=repeat,\n'
         '                order_in_block=repeat % 2, ordering="32/16/16/32 ABBA")\n'
         '            if variant:\n                episodes[-1]["variant"] = variant'),
        ('            memory["measurement_peak_scope"] = "since drained warmup; includes validation reference if enabled"',
         '            memory["measurement_peak_scope"] = "since post-warmup reset; no reference validation"'),
    ]
    for old, new in edits:
        source = replace_once(source, old, new)
    start = source.index('        validated = runtime.validation_results\n')
    end = source.index('        summary = pager.finalize()', start)
    source = source[:start] + '        assert not runtime.validation_results, "reference validation must remain disabled"\n' + source[end:]
    return source

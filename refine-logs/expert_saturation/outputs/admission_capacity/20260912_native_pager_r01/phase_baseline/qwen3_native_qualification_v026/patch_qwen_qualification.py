"""One Qwen BF16 loading/numerical qualification; not a performance experiment."""
import hashlib

RUNNER_SHA256 = "29d16549131480a5bd439f653ed002213b57a316ef32b70ac526497e1b140d7c"
QWEN_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def patch_runner(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != RUNNER_SHA256:
        raise ValueError("Qwen qualification requires the frozen admission runner")
    edits = [
        ('choices=[16, 24, 64]', 'choices=[48]'),
        ('    args = p.parse_args()', '''    p.add_argument("--qwen-qualification", action="store_true", required=True)
    for name in ("manifest", "index", "workspace", "receipt"):
        p.add_argument("--loader-" + name, type=Path, required=True)
    args = p.parse_args()'''),
        ('        matrices = (args.same_engine_admission_width_baseline,',
         '        matrices = (args.qwen_qualification, args.same_engine_admission_width_baseline,'),
        ('        observed_run = args.same_engine_admission_width_baseline or args.same_engine_runtime_diagnostic or args.same_engine_retention_lifecycle_comparison',
         '        observed_run = args.qwen_qualification or args.same_engine_admission_width_baseline or args.same_engine_runtime_diagnostic or args.same_engine_retention_lifecycle_comparison'),
        ('        before = check_gpu(out, "before_initialization")', '''        if (args.requests != 4 or args.expert_cap != 48 or args.execution != "expert"
                or args.kv_bytes != 536870912 or args.token_budget != 64 or args.output_tokens != 8
                or args.prefill_limit != 32 or args.arrival_interval != 0.05 or not args.verify_kernel
                or injection or args.warmup_full or args.warmup_executions or args.trace_retention != "episode"):
            raise ValueError("Qwen qualification requires 4 requests/cap48/expert/KV512MiB/token64/8out/prefill32/50ms/verify/episode trace; no event or full warmup")
        for path in (args.loader_manifest, args.loader_index):
            if not path.is_file():
                raise ValueError("serial loader metadata path is missing")
        before = check_gpu(out, "before_initialization")'''),
        ('        frozen = json.loads((args.prepared / "workload.json").read_text())',
         f'''        frozen = json.loads((args.prepared / "workload.json").read_text())
        if frozen.get("tokenizer_identity") != {{"repository": "Qwen/Qwen3-30B-A3B", "revision": "{QWEN_REVISION}"}}:
            raise ValueError("qualification requires prompts retokenized with the frozen Qwen tokenizer")
        if len(frozen["source_requests"]) != 4 or len(frozen["actual_prompt_token_ids"]) != 4:
            raise ValueError("qualification prepared input must contain exactly four requests")
        for key in ("request_id", "document_id", "document_sha256"):
            if len({{r[key] for r in frozen["source_requests"]}}) != 4:
                raise ValueError("qualification sources must be distinct documents")
        if any(type(t) is not int or not 0 <= t < 151936 for row in frozen["actual_prompt_token_ids"] for t in row):
            raise ValueError("Qwen input contains invalid token ids")'''),
        ('        dump(out / "workload.json", workload)',
         '        workload["tokenizer_identity"] = frozen["tokenizer_identity"]\n        dump(out / "workload.json", workload)'),
        ('        import wisp_v026_adapter as pager',
         '        import qwen_serial_loader  # registers qwen_bf16_serial_v026 before model construction\n        from memory_observer import verify_values\n        import wisp_v026_adapter as pager'),
        ('                      "native_capture.py", "admission_feedback.py"]',
         '                      "native_capture.py", "admission_feedback.py", "continuous_admission.py",\n                      "memory_observer.py", "qwen_serial_loader.py", "serial_shards.py"]'),
        ('        kwargs = dict(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=20260912,',
         '''        kwargs = dict(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=20260912,
            load_format="qwen_bf16_serial_v026", model_loader_extra_config={
                name: str(getattr(args, "loader_" + name).resolve())
                for name in ("manifest", "index", "workspace", "receipt")},'''),
        ('max_num_seqs=3 if args.same_engine_admission_width_baseline else 16', 'max_num_seqs=4'),
        ('                     intermediate=cfg.intermediate_size)', '                     intermediate=cfg.moe_intermediate_size)'),
        ('        assert shape["layers"] == 16 and shape["experts"] == 64 and shape["top_k"] == 8',
         '        assert shape == dict(layers=48, experts=128, top_k=8, hidden=2048, intermediate=768)\n        assert len(runtime.layers) == 48'),
        ('        scheduler = core.scheduler\n        original = scheduler.schedule',
         '''        scheduler = core.scheduler
        assert not engine.has_unfinished_requests() and not scheduler.requests
        device = next(iter(runtime.layers.values()))["state"].scratch_w13.device
        equality = verify_values(torch.cuda, device)
        equality.update(mode="nested", warmup_mode="nested", measurement_mode="nested")
        dump(out / "memory_observer.json", equality)
        if not equality["values_equal"]:
            raise RuntimeError("allocator values changed or observer fields differ")
        runtime.memory_observer_mode = "nested"
        original = scheduler.schedule'''),
        ('                      policy="static", allow_preemption=True)',
         '                      policy="static", allow_preemption=False, phase_prefill_mode="static32")'),
        ('        if observed_run:\n            cycle = dict(', '        count = 1\n        if observed_run:\n            cycle = dict('),
        ('            episode_name = f"repeat_{repeat}" + ("_" + variant if variant else "")',
         '            variant = "qwen_qualification"\n            cell_arm = variant\n            cell_chunk = 32\n            episode_name = f"repeat_{repeat}_qwen_qualification"'),
        ('                                       run_id="measurement", max_seconds=args.max_seconds, runtime_observer=observer))',
         '                                       run_id=phase, max_seconds=args.max_seconds, runtime_observer=observer, cpu_diagnostics=True))'),
        ('        summary = pager.finalize()', '''        validated = runtime.validation_results
        layer_bytes = {entry["layer_name"]: runtime.expert_bytes(entry["state"]) * entry["state"].num_experts
                       for entry in runtime.layers.values()}
        reference_bytes = sum(layer_bytes[row["layer_name"]] for row in validated)
        dump(out / "qualification_validation.json", dict(
            scope="Independent loading and same-precall full-weight numerical qualification; no performance claim. Raw capture includes reference copies/compute; pager bytes exclude those reference copies and model loading.",
            validation_calls=len(validated), expected_layers=48,
            unique_validated_layers=len({row["layer_name"] for row in validated}),
            allfinite=all(row["allfinite"] for row in validated),
            allclose_diagnostic=all(row["allclose"] for row in validated),
            reference_full_weight_copy_bytes=reference_bytes,
            reference_copy_accounting="Tensor payload derived from the two actual full-weight .to(device) calls per validation; not measured wire bytes",
            layer_validation=validated))
        if len(validated) != 48 or set(layer_bytes) != {row["layer_name"] for row in validated}:
            raise RuntimeError("qualification did not validate exactly one full-weight reference per layer")
        if not all(row["allfinite"] for row in validated):
            raise RuntimeError("non-finite numerical qualification output")
        summary = pager.finalize()'''),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"Qwen qualification anchor count: {old[:80]!r}")
        source = source.replace(old, new)
    return source

"""Add a native singleton-token cost diagnostic after patch_map_probe()."""


def patch_controlled_token_probe(source: str) -> str:
    def patch(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("controlled-token source anchor changed: " + old[:80])
        source = source.replace(old, new, 1)

    patch("def main():\n", '''def token_control_mask_snapshot(worker):
    batch = worker.model_runner.input_batch
    return dict(pid=os.getpid(), masks={name: dict(data_ptr=t.data_ptr(),
        storage_bytes=t.untyped_storage().nbytes(), shape=list(t.shape),
        device=str(t.device), pinned=t.is_pinned()) for name, t in (
        ("gpu", batch.allowed_token_ids_mask), ("cpu", batch.allowed_token_ids_mask_cpu_tensor))})


def main():
''')
    patch('        raw = args.workload.read_bytes()\n', '''        control_path = args.workload.with_name("token_control.json")
        control = json.loads(control_path.read_text())
        controlled_id = control["token_id"]
        require(type(controlled_id) is int and not args.no_new, "Need integer controlled token and new request")
        result["token_control"] = dict(control, config_path=str(control_path),
            diagnostic_only=True, scope="New-request repeated token; native sampler; includes mask costs")
        raw = args.workload.read_bytes()
''')
    patch("params = lambda count: SamplingParams(", "params = lambda count, allowed=None: SamplingParams(")
    patch("min_tokens=count, ignore_eos=True, detokenize=False, output_kind=RequestOutputKind.CUMULATIVE)",
          "min_tokens=count, ignore_eos=True, detokenize=False, output_kind=RequestOutputKind.CUMULATIVE, allowed_token_ids=allowed)")
    patch('        result["warmup"] = dict(name="warmup")\n', '''        tokenizer = llm.get_tokenizer()
        require(0 <= controlled_id < min(len(tokenizer), int(model_config["vocab_size"]))
                and controlled_id not in tokenizer.all_special_ids, "Controlled token must be ordinary and in vocabulary")
        require(not engine.has_unfinished_requests(), "Primer requires drained engine")
        primer_id = "token-control-primer"
        primer = result["token_control_primer"] = dict(status="RUNNING", output_token_ids=[], finished=False)
        primer_start = time.perf_counter()
        try:
            engine.add_request(primer_id, {"prompt_token_ids": [controlled_id]},
                               params(1, [controlled_id]), arrival_time=time.time())
            while engine.has_unfinished_requests():
                for output in engine.step():
                    require(output.request_id == primer_id and len(output.outputs) == 1, "Unexpected primer output")
                    primer.update(output_token_ids=list(output.outputs[0].token_ids), finished=bool(output.finished))
        finally:
            primer["wall_s"] = time.perf_counter() - primer_start
        primer.update(output_count=len(primer["output_token_ids"]),
                      output_sha256=hashlib.sha256(json.dumps(primer["output_token_ids"]).encode()).hexdigest())
        require(primer["finished"] and primer["output_token_ids"] == [controlled_id], "Primer singleton output mismatch")
        primer.update(status="COMPLETED", mask_storage=engine.engine_core.collective_rpc(token_control_mask_snapshot))
        result["warmup"] = dict(name="warmup")
''')
    patch('params(row["max_tokens"]), arrival_time=',
          'params(row["max_tokens"], [controlled_id] if rid == new_id else None), arrival_time=')
    patch('        result["wall_s"] = now()\n', '''        result["wall_s"] = now()
        require(result["requests"][new_id]["output_token_ids"] == [controlled_id] * 8,
                "Controlled new-request output must equal the frozen token eight times")
        result["token_control"]["new_output_validated"] = True
''')
    patch("Host receipt; includes inline observation and action snapshot costs",
          "CONTROLLED_OUTPUT_COST_DIAGNOSTIC: host receipt includes native mask, observation and action snapshot costs")
    compile(source, "controlled_token_phase_probe.py", "exec")
    return source

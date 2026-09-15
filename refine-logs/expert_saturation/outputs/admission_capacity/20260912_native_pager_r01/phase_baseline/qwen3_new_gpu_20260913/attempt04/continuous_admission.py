"""CPU-only input/state helpers for a bounded native clock-arrival episode."""
import math


def prepare_workload(frozen, args):
    indices = args.source_indices if args.source_indices is not None else list(range(8))
    lengths = args.prompt_lengths if args.prompt_lengths is not None else [args.prompt_tokens] * 8
    outputs = args.output_lengths if args.output_lengths is not None else [args.output_tokens] * 8
    arrivals = args.arrival_times if args.arrival_times is not None else [i * args.arrival_interval for i in range(8)]
    if (args.requests != 8 or any(len(v) != 8 for v in (indices, lengths, outputs, arrivals))
            or len(set(indices)) != 8 or any(type(i) is not int or i < 0 for i in indices)
            or any(type(n) is not int or n < 1 for n in lengths)
            or any(type(n) is not int or not 2 <= n <= 128 for n in outputs)
            or any(not math.isfinite(t) or t < 0 for t in arrivals)):
        raise ValueError("need eight unique sources, positive prefixes, output lengths2..128 and finite arrivals")
    sources = [frozen["source_requests"][i] for i in indices]
    prompts = [frozen["actual_prompt_token_ids"][i] for i in indices]
    if len({r["request_id"] for r in sources}) != 8 or any(len(ids) < n for ids, n in zip(prompts, lengths)):
        raise ValueError("duplicate source IDs or prefix exceeds prepared tokens")
    maximum = max(n + o for n, o in zip(lengths, outputs))
    if maximum > 32768:
        raise ValueError("request exceeds the model context limit")
    return dict(source_requests=sources, actual_prompt_token_ids=[ids[:n] for ids, n in zip(prompts, lengths)],
        source_indices=indices, arrival_traces_s={"steady": arrivals}, max_request_tokens=maximum,
        output_tokens_by_request={r["request_id"]: n for r, n in zip(sources, outputs)},
        scope="Finite eight-request clock-arrival episode; native enqueue at synchronous step boundaries")


def prepare_warmup(frozen, args):
    indices, length = args.warmup_source_indices, args.warmup_prompt_tokens
    if (indices is None or not 1 <= len(indices) <= 3 or len(set(indices)) != len(indices)
            or any(i < 0 for i in indices) or not 64 <= length <= 32766):
        raise ValueError("choose one to three calibration sources and a warmup prefix64..32766")
    if set(indices) & set(args.source_indices if args.source_indices is not None else range(8)):
        raise ValueError("common warmup sources must be distinct from evaluation sources")
    sources = [frozen["source_requests"][i] for i in indices]
    prompts = [frozen["actual_prompt_token_ids"][i] for i in indices]
    if any(len(ids) < length for ids in prompts):
        raise ValueError("warmup exceeds prepared tokens")
    return dict(source_requests=sources, actual_prompt_token_ids=[ids[:length] for ids in prompts],
                source_indices=indices, arrival_traces_s={"steady": [0.] * len(indices)}, max_request_tokens=length + 2)


def choose_phase(scheduler, mode, now):
    if mode not in ("static32", "phase32"):
        raise ValueError("phase mode must be static32 or phase32")
    if (scheduler.scheduler_config.async_scheduling or scheduler.num_spec_tokens
            or scheduler.num_lookahead_tokens or scheduler.max_num_scheduled_tokens != 64):
        raise RuntimeError("phase probe requires synchronous non-speculative budget64")
    for r in scheduler.requests.values():
        if r.num_preemptions or r.num_output_placeholders or r.spec_token_ids:
            raise RuntimeError("phase probe excludes preemption, asynchronous and speculative state")
        if not 0 <= r.num_computed_tokens <= r.num_tokens or r.num_tokens < r.num_prompt_tokens:
            raise RuntimeError("invalid request progress")
    ready = []
    for r in scheduler.running:
        if r.status.name != "RUNNING":
            raise RuntimeError("running list contains a non-running request")
        if r.num_tokens > r.num_prompt_tokens:
            if r.num_computed_tokens != r.num_tokens - 1:
                raise RuntimeError("output-bearing request is not ready for one decode")
            ready.append(r.request_id)
    return dict(mode=mode, ready_decode_ids=ready, threshold_before=scheduler.scheduler_config.long_prefill_token_threshold,
                chosen_threshold=32 if mode == "static32" or ready else 0,
                effective_token_budget=64, decision_s=now(), status="decided")

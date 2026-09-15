"""Fixed Qwen prefill thresholds; shared native ready-decode guards, no feedback."""
from continuous_admission import choose_phase as checked_ready_state

CHUNKS = (32, 16, 16, 32)


def choose_phase(scheduler, mode, now):
    if mode not in ("static16", "static32"):
        raise ValueError("Qwen comparison only accepts static16/static32")
    decision = checked_ready_state(scheduler, "static32", now)
    decision.update(mode=mode, chosen_threshold=int(mode[6:]))
    return decision


def prepare_workload(frozen):
    lengths, outputs, arrivals = [64, 64, 128, 32], [32, 32, 16, 24], [0, 0, 2, 4]
    if (frozen.get("source_indices") != [4, 5, 6, 7]
            or frozen.get("prompt_lengths") != lengths or frozen.get("output_lengths") != outputs
            or frozen.get("arrival_traces_s") != {"steady": arrivals}
            or list(map(len, frozen["actual_prompt_token_ids"])) != lengths):
        raise ValueError("Qwen comparison requires the frozen four-request lengths and absolute arrivals")
    return dict(frozen, output_tokens_by_request={r["request_id"]: n
        for r, n in zip(frozen["source_requests"], outputs)})

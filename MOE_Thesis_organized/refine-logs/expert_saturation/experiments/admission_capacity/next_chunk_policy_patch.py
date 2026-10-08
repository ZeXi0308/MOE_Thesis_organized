"""One next-chunk intervention on copies of the frozen static/map-count probe.

State hashes cover request tokens, logical KV allocation/progress and pager state;
the existing worker snapshot does not capture KV tensor contents.
"""
import hashlib
import json
import math


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def next_chunk_threshold(result, action):
    """Called by the phase wrapper before native schedule/allocate_slots."""
    _require(type(action) is int and action in (8, 16, 32), "Invalid next-chunk action")
    if "action" not in result:
        return 32
    first = result["action"]["engine_call"]
    index = result["steps"][-1]["index"]
    _require(index >= first, "Action precedes the common first mixed chunk")
    if index == first + 1:
        boundary = result.get("next_chunk_branch", {})
        _require(boundary.get("engine_call") == index and boundary.get("selected_chunk") == action and
                 boundary.get("visible_feedback", {}).get("engine_call") == first,
                 "Completed first-chunk feedback/state unavailable before action")
        boundary["chosen_threshold"] = action
        boundary["decision_boundary"] = "before_native_schedule_and_KV_allocation"
        return action
    return 16


def capture_next_chunk_state(result, scheduler, snapshot, scheduler_state, now, *, action, curve_s):
    """Run after first-chunk receipts are committed, before creating the next call."""
    first = result["action"]["engine_call"]
    _require("next_chunk_branch" not in result and len(result["steps"]) == first + 1,
             "Prebranch snapshot must follow exactly one mixed chunk")
    call, counters = result["steps"][-1], result["first_action_map_counters"]
    duration = call["return_s"] - call["start_s"]
    _require(call["index"] == first and call.get("returned") and
             math.isfinite(duration) and duration > 0, "First mixed chunk incomplete")
    _require(counters["engine_call"] == first and counters["layer_count"] == 16,
             "First-chunk counters do not match the completed call")
    _require(sum(r["prefill_tokens"] for r in call["scheduled"]) == 16 and
             sum(r["decode_tokens"] for r in call["scheduled"]) == 2,
             "Expected common first chunk16 plus two incumbent decode rows")
    started = now()
    workers = snapshot()
    state = dict(requests={rid: {key: row[key] for key in (
        "document_id", "document_sha256", "prompt_token_ids", "output_token_ids",
        "max_tokens", "status")} for rid, row in result["requests"].items()},
        scheduler=scheduler_state(scheduler),
        pager=[worker["pager_execution_state"] for worker in workers],
        worker_kv=[{key: worker[key] for key in (
            "num_gpu_blocks", "kv_allocated_tensor_bytes", "kv_groups")} for worker in workers])
    state = json.loads(json.dumps(state, sort_keys=True))  # Freeze mutable token lists.
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    feedback = json.loads(json.dumps(dict(engine_call=first, duration_s=duration,
        available_after_s=call["return_s"], map_counters=counters)))
    curve = {str(n): float(curve_s[str(n)]) for n in (8, 16, 32)}
    _require(type(action) is int and action in (8, 16, 32) and
             all(math.isfinite(v) and v > 0 for v in curve.values()), "Invalid frozen curve/action")
    predictions = dict(static=curve[str(action)], recent=duration * curve[str(action)] / curve["16"])
    ended = now()
    return dict(engine_call=first + 1, visible_feedback=feedback, before_state=state,
        prestate_sha256=digest, selected_chunk=action, first_mixed_step_s=duration,
        predictions_s=predictions, frozen_curve_s=curve, capture_start_s=started, capture_end_s=ended,
        capture_duration_s=ended-started,
        state_scope="Request tokens, logical KV blocks/progress and pager slots/LRU; no KV tensor bytes",
        timing_scope="Capture/synchronization remains in episode wall and request receipt gaps")


def _replace(source, old, new):
    _require(source.count(old) == 1, "Next-chunk source anchor changed: " + old[:80])
    return source.replace(old, new, 1)


def patch_next_chunk_policy(source: str) -> str:
    """Apply after patch_static_policy; retain its ready-decode/preemption checks."""
    source = _replace(source,
        "def install(scheduler, *, vllm_config, mode, is_enabled, current_call):",
        "def install(scheduler, *, vllm_config, mode, is_enabled, current_call, next_threshold):")
    source = _replace(source,
        "        decision = choose_threshold(scheduler.running, enabled=enabled, mode=mode)\n",
        "        decision = choose_threshold(scheduler.running, enabled=enabled, mode=mode)\n"
        "        decision[\"chosen_threshold\"] = next_threshold()\n")
    compile(source, "next_chunk_phase_policy.py", "exec")
    return source


def patch_next_chunk_probe(source: str) -> str:
    """Apply last, after static, map and first-step-map-counter probe patches."""
    _require(source.count("first_map_before = None") == 1, "First-chunk counter patch required")
    source = _replace(source, "from wisp_paging_trace import TRACE\n",
        "from wisp_paging_trace import TRACE\n"
        "from next_chunk_policy_patch import capture_next_chunk_state, next_chunk_threshold\n")
    source = _replace(source, "    args = p.parse_args()\n",
        "    p.add_argument(\"--next-chunk\", type=int, choices=(8, 16, 32), required=True)\n"
        "    args = p.parse_args()\n"
        "    require(args.chunk == 16 and args.policy == \"static16\", \"Common chunk/policy must be16\")\n"
        "    next_curve = json.loads((Path(__file__).parent / \"config.json\").read_text())[\"frozen_prediction\"][\"curve_s\"]\n")
    source = _replace(source, '                             current_call=lambda: result["steps"][-1])',
        '                             current_call=lambda: result["steps"][-1],\n'
        '                             next_threshold=lambda: next_chunk_threshold(result, args.next_chunk))')
    source = _replace(source,
        '                if result.get("action", {}).get("engine_call") == call["index"]:\n',
        '                if call["index"] in (result.get("action", {}).get("engine_call"),\n'
        '                        result.get("next_chunk_branch", {}).get("engine_call")):\n')
    source = _replace(source,
        '                    require("first_action_map_counters" not in result, "First-action map counters repeated")\n'
        '                    result["first_action_map_counters"] = dict(',
        '                    first_counter = call["index"] == result["action"]["engine_call"]\n'
        '                    counter_owner = result if first_counter else result["next_chunk_branch"]\n'
        '                    counter_key = "first_action_map_counters" if first_counter else "map_counters"\n'
        '                    require(counter_key not in counter_owner, "Step map counters repeated")\n'
        '                    counter_owner[counter_key] = dict(')
    source = _replace(source, '            call = dict(index=len(result["steps"]), output_request_ids=[])\n',
        '            if "action" in result and len(result["steps"]) == result["action"]["engine_call"] + 1:\n'
        '                result["next_chunk_branch"] = capture_next_chunk_state(\n'
        '                    result, scheduler, snapshot, scheduler_state, now,\n'
        '                    action=args.next_chunk, curve_s=next_curve)\n'
        '            call = dict(index=len(result["steps"]), output_request_ids=[])\n')
    source = _replace(source, '        require("action" in result, "Action boundary never reached")\n',
        '        require("action" in result, "Action boundary never reached")\n'
        '        require(result.get("next_chunk_branch", {}).get("chosen_threshold") == args.next_chunk,\n'
        '                "Next-chunk action was not applied")\n'
        '        branch = result["steps"][result["next_chunk_branch"]["engine_call"]]\n'
        '        require(branch["phase_prefill"]["actual_prefill_rows"] == args.next_chunk,\n'
        '                "Next-chunk actual prefill shape differs from chosen action")\n')
    compile(source, "next_chunk_probe.py", "exec")
    return source

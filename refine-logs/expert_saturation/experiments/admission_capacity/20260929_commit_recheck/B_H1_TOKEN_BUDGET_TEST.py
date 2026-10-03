"""CPU counterexample for H1's unqualified first-output prediction.

Run with the uncommitted H1 adapter path, for example::

    PYTHONDONTWRITEBYTECODE=1 python3 B_H1_TOKEN_BUDGET_TEST.py \
        /path/to/staged_store_rotation.py

This executes the adapter's actual read-only direct gate. The token accounting
models the pinned vLLM 0.26 running-before-waiting scheduler with no host KV
hit, no speculative tokens, and the candidate's 1024-token batch budget. It
does not execute native allocation, a connector, or a GPU worker.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N


def load_gate(path: Path):
    tree = ast.parse(path.read_text())
    funcs = [node for node in tree.body
             if isinstance(node, ast.FunctionDef)
             and node.name == "_direct_resume_reason"]
    assert len(funcs) == 1, "H1 direct gate missing or duplicated"
    namespace = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace["_direct_resume_reason"]


def counterexample(gate):
    # 31 pure-decode residents leave one of the 32 sequence slots free.
    running = [N(request_id=f"r{i}", status=N(name="RUNNING"),
                 num_prompt_tokens=16, num_output_tokens=1,
                 num_computed_tokens=16) for i in range(31)]
    victim = running[-1]
    target = N(request_id="target", status=N(name="PREEMPTED"),
               num_prompt_tokens=993, num_output_tokens=1,
               num_computed_tokens=0, is_finished=lambda: False)
    owned = {}
    blocks = {}
    for index, request in enumerate(running):
        ids = (2 * index, 2 * index + 1)
        owned[request.request_id] = [
            N(block_id=bid, is_null=False, ref_cnt=1) for bid in ids]
        blocks.update((block.block_id, block)
                      for block in owned[request.request_id])
    for bid in range(62, 162):
        blocks[bid] = N(block_id=bid, is_null=False, ref_cnt=0)
    owned[target.request_id] = []
    pool = N(blocks=blocks, get_num_free_blocks=lambda: 100)
    manager = N(get_blocks=lambda rid: N(get_block_ids=lambda: [
        [block.block_id for block in owned[rid]]]))
    scheduler = N(running=running, waiting=[target], skipped_waiting=[],
                  max_num_running_reqs=32, num_waiting_for_streaming_input=0,
                  max_num_scheduled_tokens=1024)
    connector = N(_req_status={"target": N(transfer_jobs=set())})

    disposition = gate(scheduler, manager, pool, owned, connector, target)
    assert disposition == "DIRECT_READY", disposition
    assert len(running) < scheduler.max_num_running_reqs
    assert pool.get_num_free_blocks() >= (target.num_prompt_tokens
                                          + target.num_output_tokens + 15) // 16
    # All residents have two full KV blocks for 17 historical tokens; their
    # next one-token decode needs no new block, so the KV hold rule permits all.
    assert all(len(owned[r.request_id]) == 2 for r in running)

    history_tokens = target.num_prompt_tokens + target.num_output_tokens
    direct_grant = min(history_tokens,
                       scheduler.max_num_scheduled_tokens - len(running))
    old_grant = min(history_tokens,
                    scheduler.max_num_scheduled_tokens - (len(running) - 1))
    assert (history_tokens, direct_grant, old_grant) == (994, 993, 994)
    # With no host KV hit, a recomputing target can produce a new output only
    # after its complete 994-token history has been scheduled. The old commit
    # reaches that boundary in this call; direct falls short by one token.
    assert direct_grant < history_tokens == old_grant
    return {
        "gate": disposition,
        "running_before_commit": len(running),
        "sequence_slots": scheduler.max_num_running_reqs,
        "free_blocks": pool.get_num_free_blocks(),
        "target_needed_blocks": (history_tokens + 15) // 16,
        "token_budget": scheduler.max_num_scheduled_tokens,
        "target_history_tokens": history_tokens,
        "direct_target_scheduled_tokens": direct_grant,
        "old_target_scheduled_tokens": old_grant,
        "direct_reaches_generation_boundary_this_call": False,
        "old_reaches_generation_boundary_this_call": True,
        "evidence": "CPU conditional counterexample; native full-request UNRUN",
    }


def historical_bounds(path: Path):
    """Bound no-host-hit backlog; old derived rows omit exact token counters."""
    rows = json.loads(path.read_text())["rows"]
    changed = [row for row in rows
               if row.get("decision") == "RESUME_WITHOUT_VICTIM"]
    assert len(changed) == 1, (path, len(changed))
    row = changed[0]
    blocks = row["target_remaining_blocks"]
    running = row["running_count"]
    assert row["all_running_pure_decode"] is True
    return {
        "step": row["step"],
        "target_needed_blocks": blocks,
        "running_count": running,
        "no_host_hit_backlog_if_preempted_owns_zero_blocks":
            [16 * (blocks - 1) + 1, 16 * blocks],
        "direct_one_call_token_capacity": 1024 - running,
        "old_one_call_token_capacity": 1024 - running + 1,
        "conservative_whole_history_direct_gate": "KEEP_PLANNED_SWAP",
        "exact_backlog_and_native_host_hit": "UNOBSERVED_IN_DERIVED_ROW",
    }


if __name__ == "__main__":
    if len(sys.argv) not in (2, 4):
        raise SystemExit("usage: B_H1_TOKEN_BUDGET_TEST.py PATH_TO_H1_ADAPTER "
                         "[SAVED_D_SELECTED_JSON SAVED_E_FULL_JSON]")
    result = {"synthetic": counterexample(load_gate(Path(sys.argv[1])))}
    if len(sys.argv) == 4:
        result["historical_cpu_snapshot_bounds"] = [
            historical_bounds(Path(path)) for path in sys.argv[2:]]
    print(json.dumps(result, indent=2))

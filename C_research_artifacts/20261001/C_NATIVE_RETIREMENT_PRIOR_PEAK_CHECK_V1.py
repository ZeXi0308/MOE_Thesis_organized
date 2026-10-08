#!/usr/bin/env python3
"""Compare the resident retirement peak with the prior sorted-prefix expression.

CPU algebra/examples only when the saved native trace lacks resident state.
This is not a LightLLM execution, complete port, or performance comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from C_NATIVE_RETIREMENT_ADMISSION_V1 import decode_envelope


def prior_peak(rows, block_size=None):
    """Prior descending-horizon prefixes, optionally paged per request."""
    ordered = sorted(rows, key=lambda row: -row[1])
    values = []
    for i, (_, horizon) in enumerate(ordered, 1):
        allocations = [tokens + horizon for tokens, _ in ordered[:i]]
        values.append(sum(allocations) if block_size is None else
                      sum((n + block_size - 1) // block_size for n in allocations))
    return max(values, default=0)


def endpoint_peak(rows, block_size=None):
    values = []
    for t in {0, *(h for _, h in rows)}:
        allocations = [tokens + t for tokens, h in rows if t <= h]
        values.append(sum(allocations) if block_size is None else
                      sum((n + block_size - 1) // block_size for n in allocations))
    return max(values, default=0)


def example(name, triples):
    requests = [SimpleNamespace(num_prompt_tokens=p, num_output_tokens=o, max_tokens=m)
                for p, o, m in triples]
    rows = [(p + o, m - o) for p, o, m in triples]
    actual_c = decode_envelope(requests)
    paged_prior = prior_peak(rows, 16)
    token_prior = prior_peak(rows)
    assert actual_c == paged_prior == endpoint_peak(rows, 16)
    assert token_prior == endpoint_peak(rows)
    aggregate_blocks = (token_prior + 15) // 16
    assert actual_c >= aggregate_blocks
    return dict(name=name, evidence="SYNTHETIC_INTEGER_STATE",
                resident_P_O_M=triples, same_endpoint_rows_x_h=rows,
                c_source_decode_envelope_blocks=actual_c,
                prior_same_endpoint_per_request_paged_blocks=paged_prior,
                prior_same_endpoint_token_peak=token_prior,
                ceil_of_aggregate_token_peak_blocks=aggregate_blocks,
                per_request_paging_extra_blocks=actual_c - aggregate_blocks,
                ae_normal_running_cap_h_minus_1_token_peak=prior_peak(
                    [(x, max(0, h - 1)) for x, h in rows]))


def trace_summary(root, label):
    path = root / label / "native_retirement" / "retirement-envelope.json"
    trace = json.loads(path.read_text())
    eligible = [s for s in trace["steps"] if s["eligible_all_decode"]]
    # This frozen schema has only aggregate counts and peaks, not P/O/M snapshots.
    state_keys = {"residents", "running_requests", "resident_rows", "decode_rows"}
    assert not any(state_keys.intersection(s) for s in trace["steps"])
    empty = [s for s in eligible if s["running_before"] == 0]
    assert all(s["conditional_decode_peak"] == decode_envelope([]) == prior_peak([], 16)
               for s in empty)
    return dict(label=label, path=str(path), step_count=len(trace["steps"]),
                eligible_calls=len(eligible),
                eligible_nonempty_states=sum(s["running_before"] > 0 for s in eligible),
                observed_nonempty_states_compared=0,
                trivial_observed_empty_states_compared=len(empty),
                eligible_step_fields=sorted(set().union(*(s.keys() for s in eligible))),
                recorded_conditional_peak_range_blocks=[
                    min(s["conditional_decode_peak"] for s in eligible),
                    max(s["conditional_decode_peak"] for s in eligible)],
                missing="Per-resident identity, P, O and M at eligible boundaries; aggregate peaks cannot reconstruct them.",
                replay_status="NONEMPTY_OBSERVED_STATE_COMPARISON_UNAVAILABLE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--prior-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads((args.prior_source_root / "source-receipt.json").read_text())
    current = args.prior_source_root / "current-queue.py"
    examples = [example(name, rows) for name, rows in [
        ("empty", []),
        ("one_resident", [(1500, 400, 1024)]),
        ("equal_horizons_paging", [(16, 1, 2), (16, 1, 2)]),
        ("three_tied_horizons", [(1, 1, 2), (1, 1, 2), (1, 1, 2)]),
        ("staggered_cap_retirement", [(2976, 988, 1024)] * 8 + [(2976, 64, 1024)] * 8),
        ("mixed_horizons", [(15, 1, 2), (17, 2, 17), (130, 23, 100)]),
        ("indexing_crosses_page", [(15, 1, 2)]),
    ]]
    old_rows, waiting_row = [(31, 1)], (16, 64)
    new_full = (sum(waiting_row) + 15) // 16
    separate = prior_peak(old_rows, 16) + new_full
    joint = prior_peak(old_rows + [waiting_row], 16)
    assert separate == 7 and joint == 5
    result = dict(
        schema="c-native-retirement-prior-peak-check-v1",
        status="ALGEBRAIC_IDENTITY_AND_SYNTHETIC_CHECK_ONLY",
        source_receipt=sources,
        current_queue=dict(path=str(current), sha256=hashlib.sha256(current.read_bytes()).hexdigest()),
        c_function="C_NATIVE_RETIREMENT_ADMISSION_V1.py:decode_envelope (imported unchanged)",
        trace_coverage=[trace_summary(args.fresh_root, label) for label in ["retirement_1", "retirement_2"]],
        algebra={
            "domain": "x_j=P_j+O_j>=0; h_j=M_j-O_j>=0; occupancy retained at t=h_j; block size B=16.",
            "c": "max_{t in {0,h_j}} sum_{j:h_j>=t} ceil((x_j+t)/B)",
            "prior_token": "Sort h_1>=...>=h_n; max_{i=1..n}(sum_{j<=i} x_j + i*h_i).",
            "prior_paged_same_endpoint": "max_{i=1..n} sum_{j<=i} ceil((x_j+h_i)/B)",
            "identity": "C equals the per-request-paged sorted-prefix expression at the same endpoints.",
            "proof": "At t=h_i, active residents are the prefix ending at the last index with that horizon. Within a tie, shorter prefixes cannot exceed its last prefix because allocations are nonnegative. t=0 is dominated by the smallest nonnegative horizon endpoint. Between releases each surviving allocation is nondecreasing. The same argument without ceil proves token equality.",
            "paging": "ceil(prior_token_peak/B) <= C; per-request rounding cannot be replaced by rounding the aggregate once. The gap is at most n-1 blocks for n>0.",
        },
        synthetic_examples=examples,
        admission_example=dict(evidence="SYNTHETIC_NORMALIZED_GROWTH_ONLY",
            old_x_h=old_rows, waiting_x_h=waiting_row,
            c_resident_peak_plus_new_full_blocks=separate,
            jointly_modeled_peak_blocks=joint,
            limit="Illustrates separate full charge versus joint growth; does not model real waiting prefill or prove a new native admission safe."),
        policy_differences={
            "new_requests": "C adds each waiting/partial-prefill request's full ceil((P+M)/16) to the old decode peak, and stops admitting while any resident prefills. Prior _can_add_new_req appends each new tuple before recomputing the joint sorted-prefix peak.",
            "length_estimate": "AE NormalReq uses max_output_len already when busy or ignore_eos; otherwise min(max_output_len,max(int(1.1*O),router_max_new_token_len)). C always uses the known cap. Cap use alone is not novel relative to the prior conservative branch.",
            "indexing": "AE NormalReq RUNNING (zero prompt cache) returns (P+O,max(0,L-O-1)); WAIT_IN_QUEUE returns (P+1,max(0,L-2)). C uses (P+O,M-O), one token beyond current computed KV. Equality above normalizes endpoints and is not literal equality of these raw tuples. AE SplitFuseReq additionally counts pending prefill chunks.",
            "capacity_test": "Official prior queue uses peak < available_token_capacity; C accepts paged peak plus full new quotas <=4096 blocks. Prior also charges paused/cache state and queue limits outside the formula.",
            "progress_contract": "C checks all-decode eligibility, a protected running prefix, one scheduled and produced output token per call until completion, and aborts on preemption or violation in a synchronous single-GPU/no-offload domain. The inspected prior queue/tuple functions do not implement these C-specific runtime checks; this is not an audit of the whole LightLLM engine.",
        },
        conclusion="The resident cap-retirement peak is the same prior peak accounting after endpoint normalization and per-request paging. This check does not support a new peak formula claim; any residual claim must be about a demonstrated policy/execution contract, not this identity.",
        limits="No nonempty observed-state replay, prior runtime execution, matched performance comparison, new controller, or GPU work. Synthetic states are mathematical examples, not measured opportunities or benefits.",
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(status=result["status"], synthetic_cases=len(examples),
                         trace_coverage=[{k: s[k] for k in ("label", "eligible_calls",
                         "observed_nonempty_states_compared", "trivial_observed_empty_states_compared")}
                         for s in result["trace_coverage"]]), ensure_ascii=False))


if __name__ == "__main__":
    main()

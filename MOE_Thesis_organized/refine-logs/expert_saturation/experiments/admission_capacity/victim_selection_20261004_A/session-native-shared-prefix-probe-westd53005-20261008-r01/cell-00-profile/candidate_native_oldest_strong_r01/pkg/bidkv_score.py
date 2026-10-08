# SPDX-License-Identifier: Apache-2.0
"""Default BidKV score adaptation, official selector commit 5ee80256d263d58b1e512d9d436d47e9bac564ba.

Source: vLLM-HUST/vllm-ascend-hust-bidkv, vllm_ascend_bidkv/selector.py.
Only the score and tie order are adapted. Candidate-set and continuation
semantics belong to this experiment, not to the full BidKV implementation.
"""

def bidkv_score(request):
    computed = max(int(getattr(request, 'num_computed_tokens', 0) or 0), 0)
    preemptions = max(int(getattr(request, 'num_preemptions', 0) or 0), 0)
    output_ids = getattr(request, 'output_token_ids', None)
    try:
        outputs = len(output_ids) if output_ids is not None else int(getattr(request, 'num_output_tokens', 0) or 0)
    except TypeError:
        outputs = int(getattr(request, 'num_output_tokens', 0) or 0)
    maximum = getattr(request, 'max_tokens', None)
    if not isinstance(maximum, (int, float)) or maximum <= 0:
        maximum = 1024
    completion = min(max(float(outputs) / float(maximum), 0.0), 1.0)
    utility = max(float(computed), 0.0) / max(1.0 + 0.5 * completion + 0.3 * max(float(preemptions), 0.0) + 1e-6, 1e-6)
    return dict(utility=utility, computed_tokens=computed, completion=completion,
                num_preemptions=preemptions)

def bidkv_order(row):
    return (-row['bidkv']['utility'], row['arrival_time'], str(row['request']))

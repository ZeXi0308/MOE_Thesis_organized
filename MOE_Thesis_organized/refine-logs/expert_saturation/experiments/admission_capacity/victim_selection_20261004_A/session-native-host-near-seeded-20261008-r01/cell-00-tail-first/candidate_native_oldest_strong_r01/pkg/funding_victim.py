"""Read-only victim choice for the existing single-funder recovery action.

No native lookup/touch, transfer, target, trigger, or service-quantum changes.
The signal is the number of materialized whole KV pages beyond the currently
ready contiguous host prefix, not historical work already spent on a request.
"""
import time
import math

from staged_save_contract import prepare


def host_prefix(state, cs, block_size):
    try:
        if (type(cs.manager).__name__ != 'CPUOffloadingManager'
                or type(cs.manager._policy).__name__ != 'LRUCachePolicy'
                or cs.config.blocks_per_chunk != 1
                or len(cs.config.kv_group_configs) != 1):
            raise ValueError('unsupported native host layout')
        config = cs.config.kv_group_configs[0]
        if (config.tokens_per_chunk != block_size
                or config.sliding_window_size_in_chunks is not None):
            raise ValueError('requires full-attention page chunks')
        status = cs._req_status[state.request_id]
        if len(status.group_states) != 1:
            raise ValueError('unexpected request groups')
        group = status.group_states[0]
        count = state.computed // block_size
        if (len(group.offload_keys) < count
                or tuple(group.block_ids[:count]) != state.blocks[:count]):
            raise ValueError('host key / GPU ownership mapping unknown')
        # Access the dictionary directly: policy.get/manager.lookup may have
        # tracking effects. ref_cnt == -1 is an unfinished store, never ready.
        cache = cs.manager._policy.blocks
        prefix = 0
        for key in group.offload_keys[:count]:
            block = cache.get(key)
            if block is None or block.ref_cnt < 0:
                break
            prefix += 1
        return dict(host_ready_prefix_blocks=prefix,
                    host_missing_suffix_blocks=count-prefix,
                    host_state_error=None)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        return dict(host_ready_prefix_blocks=None, host_missing_suffix_blocks=None,
                    host_state_error=f'{type(error).__name__}: {error}')


def select_funder(running, target, free, view, step, cs, block_size, rule, residence):
    if rule not in ('tail', 'host_missing', 'arrival'):
        raise ValueError('unknown funding victim rule')
    started = time.perf_counter()
    choices = []
    rows = []
    for index, request in enumerate(running):
        try:
            state = view(request)
            plan = prepare(step, state, target, free)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError):
            continue
        epoch = residence.get(request.request_id)
        rows.append(dict(request=request.request_id, running_index=index,
            held_blocks=len(state.blocks), computed_tokens=state.computed,
            original_arrival_time=getattr(request, 'arrival_time', None),
            output_tokens=state.output, num_preemptions=request.num_preemptions,
            residence_outputs=(state.output-epoch[1] if epoch is not None
                and epoch[0] == request.num_preemptions else None),
            **host_prefix(state, cs, block_size)))
        choices.append((request, plan))
    if not choices:
        return None, None, None
    baseline = len(choices)-1
    selected = baseline
    unknown = any(row['host_missing_suffix_blocks'] is None for row in rows)
    if rule == 'host_missing' and not unknown:
        selected = min(range(len(rows)), key=lambda i:
            (rows[i]['host_missing_suffix_blocks'], -rows[i]['running_index']))
    fallback = 'UNKNOWN_HOST_STATE' if unknown else None
    if rule == 'arrival':
        arrivals = [row['original_arrival_time'] for row in rows]
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in arrivals):
            selected = max(range(len(rows)), key=lambda i: (arrivals[i], rows[i]['running_index']))
            fallback = None
        else:
            fallback = 'UNKNOWN_ORIGINAL_ARRIVAL'
    decision = dict(step=step, rule=rule, target=target.request_id,
        free_blocks=free, deficit_blocks=target.remaining_blocks-free,
        baseline_victim=rows[baseline]['request'], selected_victim=rows[selected]['request'],
        changed=selected != baseline, candidates=rows,
        fallback=fallback,
        baseline_held_blocks=rows[baseline]['held_blocks'],
        selected_held_blocks=rows[selected]['held_blocks'],
        selector_wall_s=time.perf_counter()-started)
    return *choices[selected], decision

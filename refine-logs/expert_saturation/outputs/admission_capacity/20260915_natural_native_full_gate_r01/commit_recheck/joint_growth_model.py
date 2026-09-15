"""Exclusive-block KV feasibility envelope, not latency or policy simulation.

No arrivals, releases, sharing, transfers or EOS are predicted. All retained
requests advance the stated token count. In this conditional regime the block
cost is exact; otherwise it is only the explicitly restricted envelope.
"""

def blocks(tokens, block_size=16):
    return (tokens + block_size - 1) // block_size


def growth(computed, allocated, tokens, block_size=16):
    if min(computed, allocated, tokens) < 0:
        raise ValueError("negative resource state")
    return max(0, blocks(computed + tokens, block_size) - allocated)


def swap_envelope(snapshot, victim, target, peer_tokens=16, target_outputs=16):
    """Hypothetical immediate exclusive-block swap; prepare/load delay excluded.

    target prompt+output is current recovery work through its next new output.
    The final new output has no computed KV yet, hence target_outputs - 1.
    This is a capacity score, not a guarantee that the native action executes.
    """
    requests = snapshot['requests']
    if victim not in snapshot['running_ids'] or target in snapshot['running_ids']:
        raise ValueError("invalid swap population")
    def held(rid):
        value = requests[rid]['held_blocks']
        if not isinstance(value, int):
            raise ValueError("requires one exclusive KV pool")
        return value
    t = requests[target]
    target_need = max(0, blocks(t['prompt'] + t['output'] + target_outputs - 1) - held(target))
    peers = {rid: growth(requests[rid]['computed'], held(rid), peer_tokens)
             for rid in snapshot['running_ids'] if rid != victim}
    free_after_release = snapshot['free_blocks'] + held(victim)
    return dict(victim=victim, released_blocks=held(victim), target_blocks=target_need,
                peer_growth_blocks=sum(peers.values()),
                margin_blocks=free_after_release-target_need-sum(peers.values()))

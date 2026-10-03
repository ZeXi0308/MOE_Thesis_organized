"""One-step funding prediction from the pre-prepare state only."""
from joint_growth_model import blocks, growth


def predict(snapshot, victim, target):
    states = snapshot['requests']
    t = states[target]
    need = max(0, blocks(t['prompt']+t['output'])-t['held_blocks'])
    now = snapshot['free_blocks']+states[victim]['held_blocks']-need
    peers = {r:growth(states[r]['computed'], states[r]['held_blocks'], 1)
             for r in snapshot['running_ids'] if r != victim}
    # The victim's extra block is allocated and subsequently released, cancelling.
    return dict(immediate_margin=now, peer_growth=sum(peers.values()),
                next_margin=now-sum(peers.values()))

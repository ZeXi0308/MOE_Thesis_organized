"""Only the new selector boundary: legality, read-only signal and fallback."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'pkg'))
from funding_victim import select_funder
from staged_save_contract import RequestState
from staged_store_rotation import _last_sufficient_funder


class LRUCachePolicy:
    def __init__(self, blocks): self.blocks = blocks
    def get(self, *args): raise AssertionError('must not touch policy lookup')


class CPUOffloadingManager:
    def __init__(self, blocks): self._policy = LRUCachePolicy(blocks)
    def lookup(self, *args): raise AssertionError('must not touch native lookup')


class FundingVictimTests(unittest.TestCase):
    def test_same_legal_set_signal_pending_and_tail_tie(self):
        reqs = [NS(request_id=x, num_preemptions=0) for x in ('a','b','small')]
        states = {x: RequestState(x, 63, 48, 16, 100, 'RUNNING', ids)
                  for x, ids in [('a',(1,2,3,4)),('b',(5,6,7,8)),('small',(9,))]}
        target = RequestState('target', 0, 63, 1, 100, 'PREEMPTED', ())
        cache = {x:NS(ref_cnt=0) for x in ('a0','a1','a2','b0','b2')}
        cache['b1'] = NS(ref_cnt=-1)
        cs = NS(manager=CPUOffloadingManager(cache), config=NS(blocks_per_chunk=1,
            kv_group_configs=[NS(tokens_per_chunk=16,sliding_window_size_in_chunks=None)]),
            _req_status={x:NS(group_states=[NS(block_ids=list(s.blocks),
                offload_keys=[x+str(i) for i in range(3)])]) for x,s in states.items()})
        kwargs = dict(running=reqs, target=target, free=0, view=lambda r:states[r.request_id],
                      step=2, cs=cs, block_size=16, residence={})
        original, _ = _last_sufficient_funder(reqs,target,0,kwargs['view'],2)
        tail, _, baseline = select_funder(**kwargs, rule='tail')
        selected, _, decision = select_funder(**kwargs, rule='host_missing')
        self.assertIs(tail, original)
        self.assertEqual(tail.request_id, 'b')
        self.assertEqual(selected.request_id, 'a')
        self.assertEqual([r['host_missing_suffix_blocks'] for r in decision['candidates']], [0,2])
        self.assertEqual(list(cache), ['a0','a1','a2','b0','b2','b1'])
        cache['b1'].ref_cnt = 0
        self.assertIs(select_funder(**kwargs,rule='host_missing')[0], tail)
        cs._req_status.pop('a')
        row = select_funder(**kwargs,rule='host_missing')
        self.assertIs(row[0], tail)
        self.assertEqual(row[2]['fallback'], 'UNKNOWN_HOST_STATE')


if __name__ == '__main__': unittest.main()

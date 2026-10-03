"""Narrow CPU checks for the new one-shot selection boundary; GPU UNRUN."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / 'pkg'))
from staged_save_contract import RequestState
from staged_store_rotation import _last_sufficient_funder, _oldest_observed_waiter


def request(rid, status='PREEMPTED', output=1):
    return SimpleNamespace(request_id=rid, status=SimpleNamespace(name=status),
                           num_output_tokens=output)


class OldestAdmissionSelection(unittest.TestCase):
    def test_oldest_real_output_and_stable_tie(self):
        rows=[request('new', 'WAITING', 0), request('b'), request('a')]
        chosen,reason=_oldest_observed_waiter(rows, {'b':1.0,'a':1.0})
        self.assertIsNone(reason)
        self.assertEqual(chosen.request_id,'a')

    def test_unknown_last_output_falls_back(self):
        chosen,reason=_oldest_observed_waiter([request('a'),request('b')], {'a':1.0})
        self.assertIsNone(chosen)
        self.assertEqual(reason,'KEEP_UNKNOWN_LAST_OUTPUT_TIME')

    def test_last_qualified_running_funder_only(self):
        target=RequestState('target',0,1024,1,1024,'PREEMPTED',())
        # Target requires 65 blocks; free=10. Tail has too few blocks;
        # preceding request is sufficient and already has complete prefix.
        states={
            'front':RequestState('front',1024,1000,25,1024,'RUNNING',tuple(range(64))),
            'tail':RequestState('tail',320,300,21,1024,'RUNNING',tuple(range(20))),
        }
        front=request('front','RUNNING',25);tail=request('tail','RUNNING',21)
        victim,plan=_last_sufficient_funder([front,tail],target,10,
                                            lambda row:states[row.request_id],9)
        self.assertIs(victim,front)
        self.assertEqual(plan.victim.request_id,'front')
        self.assertEqual(plan.target.request_id,'target')

    def test_unfunded_and_invalid_running_state_fall_back(self):
        target=RequestState('target',0,1024,1,1024,'PREEMPTED',())
        bad=request('bad','RUNNING',1)
        state=RequestState('bad',1024,1000,25,1024,'RUNNING',tuple(range(20)))
        victim,plan=_last_sufficient_funder([bad],target,10,lambda _:state,9)
        self.assertIsNone(victim)
        self.assertIsNone(plan)


if __name__=='__main__':
    unittest.main()

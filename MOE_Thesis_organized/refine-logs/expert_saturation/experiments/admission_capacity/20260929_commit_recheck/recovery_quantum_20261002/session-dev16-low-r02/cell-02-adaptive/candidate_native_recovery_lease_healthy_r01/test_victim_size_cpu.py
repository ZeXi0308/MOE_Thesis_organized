"""Small boundary and native FCFS closure checks for sufficient-victim selection."""
import unittest
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'pkg'))
from staged_store_rotation import _size_deficit_blocks, _size_sufficient_choice  # noqa: E402
from test_bidkv_full_running_cpu import Request, Scheduler, fcfs_branch  # noqa: E402


def rows():
    return [dict(index=index,request=str(index),held_blocks=held,
                 qualified=True,already_scheduled=False)
            for index,held in ((1,100),(2,5),(3,10),(4,5))]


class SizeVictimClosure(unittest.TestCase):
    def test_deficit_and_unknown_boundaries(self):
        self.assertEqual(_size_deficit_blocks(32,2,0,16),1)
        self.assertEqual(_size_deficit_blocks(31,2,0,16),0)
        self.assertEqual(_size_deficit_blocks(32,2,1,16),0)
        self.assertIsNone(_size_deficit_blocks(None,2,0,16))
        self.assertIsNone(_size_deficit_blocks(32,None,0,16))

    def test_min_max_tie_and_fallback(self):
        self.assertEqual(_size_sufficient_choice(rows(),1,1,'min_held_other',4),(4,None))
        self.assertEqual(_size_sufficient_choice(rows(),1,1,'max_held_other',4),(3,None))
        self.assertEqual(_size_sufficient_choice(rows(),1,11,'min_held_other',4),
                         (4,'NO_SUFFICIENT_OTHER'))
        unknown=rows();unknown[2]['qualified']=False
        self.assertEqual(_size_sufficient_choice(unknown,1,1,'min_held_other',4),
                         (4,'UNKNOWN_SUFFIX_STATE'))
        self.assertEqual(_size_sufficient_choice(rows(),1,0,'min_held_other',4),
                         (4,'UNKNOWN_OR_NO_DEFICIT'))

    def test_native_branch_preempts_chosen_unscheduled_suffix(self):
        running=[Request(str(i)) for i in range(5)]
        choice,_=_size_sufficient_choice(rows(),1,1,'max_held_other',4)
        scheduler=Scheduler(running,choice=choice,full_running=False)
        planned=[running[0]];tokens={running[0].request_id:1}
        result=fcfs_branch()(scheduler,1,3,planned,tokens,
                             {running[0].request_id:object()},{},{},0)
        self.assertIs(result[0],running[choice])
        self.assertEqual([req.request_id for req in scheduler.running],['0','1','2','4'])
        self.assertEqual(tokens,{'0':1})


if __name__=='__main__':
    unittest.main()

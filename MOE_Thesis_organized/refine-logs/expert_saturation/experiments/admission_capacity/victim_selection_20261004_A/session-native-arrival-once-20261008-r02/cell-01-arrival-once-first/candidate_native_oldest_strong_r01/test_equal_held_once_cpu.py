"""Bounded checks for the capacity-neutral one-proposal boundary."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'pkg'))
from staged_store_rotation import _equal_held_once_choice


def row(index, held, computed, qualified=True):
    # Deliberately omit host, output and recovery-history fields.
    return dict(index=index, held_blocks=held, computed_tokens=computed,
                qualified=qualified)


class EqualHeldOnceTests(unittest.TestCase):
    def choose(self, rows, consumed=False, unknown=False, active=False, block_size=16):
        return _equal_held_once_choice(rows, unknown, active, consumed, block_size)

    def test_nearest_qualified_row_in_given_suffix_and_matching_pages(self):
        rows=[row(4,184,2931), row(5,184,2930), row(6,184,2944), row(7,184,2932)]
        self.assertEqual(self.choose(rows), (5,None,True,2))
        # A held match across a whole-page boundary is not an opportunity.
        self.assertEqual(self.choose(rows[2:]),
            (7,'NO_EQUAL_HELD_AND_COMPUTED_PAGES',False,0))
        self.assertEqual(self.choose([row(6,183,2931),rows[-1]]),
            (7,'NO_EQUAL_HELD_AND_COMPUTED_PAGES',False,0))

    def test_single_consumption_even_when_later_candidates_change(self):
        first=[row(4,184,2931),row(5,184,2932)]
        _,_,consumed,_=self.choose(first)
        later=[row(7,90,1430),row(8,90,1431)]
        self.assertEqual(self.choose(later, consumed=consumed),
            (8,'ALREADY_CONSUMED',True,0))

    def test_unknown_and_protected_opportunities_do_not_consume(self):
        rows=[row(4,184,2931),row(5,184,2932)]
        self.assertEqual(self.choose(rows,unknown=True),
            (5,'UNKNOWN_SUFFIX_STATE',False,0))
        self.assertEqual(self.choose(rows,active=True),
            (5,'ACTIVE_PROTECTION_OR_PHASE',False,0))
        invalid=[row(4,184,2931,False),rows[-1]]
        self.assertEqual(self.choose(invalid),
            (5,'UNKNOWN_SUFFIX_STATE',False,0))
        self.assertEqual(self.choose(rows), (4,None,True,1))

    def test_invalid_capacity_and_singleton_do_not_consume(self):
        tail=row(5,184,2932)
        self.assertEqual(self.choose([tail]),
            (5,'NO_EQUAL_HELD_AND_COMPUTED_PAGES',False,0))
        for bad in (row(4,None,2931),row(4,184,None),row(4,184,-1)):
            self.assertEqual(self.choose([bad,tail]),
                (5,'UNKNOWN_CAPACITY_OR_COMPUTED',False,0))
        self.assertEqual(self.choose([row(4,184,2931),tail],block_size=0),
            (5,'UNKNOWN_CAPACITY_OR_COMPUTED',False,0))


if __name__ == '__main__':
    unittest.main()

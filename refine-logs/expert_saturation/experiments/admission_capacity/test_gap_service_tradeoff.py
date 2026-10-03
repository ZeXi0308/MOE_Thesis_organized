import unittest

from analyze_gap_service_tradeoff import baseline_envelope, compare, request_metrics


class GapServiceAccounting(unittest.TestCase):
    def test_static_baseline_envelope_keeps_whole_policy_denominators(self):
        def c(label,wall,gaps):
            return dict(label=label,wall_s=wall,requests=[dict(request_id=str(i),max_gap_s=g) for i,g in enumerate(gaps)])
        r=baseline_envelope([c('fast',2,[1,3]),c('smooth',3,[2,2])],c('middle',4,[1.5,2.5]))
        self.assertFalse(r['action_above_envelope_anywhere'])
        rows={x['lower_inclusive_s']:x for x in r['intervals']}
        self.assertEqual(rows[2]['rates'],{'fast':.5,'smooth':2/3,'middle':.25})
        self.assertEqual(rows[2]['best_baselines'],['smooth'])

    def test_inclusive_threshold_and_full_episode_denominator(self):
        def cell(label, wall, gaps):
            return dict(label=label, wall_s=wall, requests=[dict(request_id=str(i),
                max_gap_s=g, completion_s=wall) for i,g in enumerate(gaps)])
        result=compare(cell('a',10,[1,2]),cell('b',20,[1,1]))
        rows={r['lower_inclusive_s']:r for r in result['intervals']}
        self.assertEqual(rows[0]['baseline_qualified'],0)
        self.assertEqual((rows[1]['baseline_qualified'],rows[1]['action_qualified']),(1,2))
        self.assertEqual(rows[1]['relation'],'equal')
        self.assertEqual(rows[2]['action_rate'],.1)
        self.assertEqual(rows[2]['baseline_rate'],.2)
        self.assertEqual(rows[2]['relation'],'lower')
        self.assertTrue(result['gap_distribution_action_first_order_dominates'])

    def test_ttft_separate_from_inter_output_gap(self):
        r=dict(request_id='r',arrival_s=0,completion_s=10.5,
            token_times_s=[10,10.2,10.5],output_token_ids=[1,2,3],status='completed')
        raw=dict(status='COMPLETE',error=None,requests=[r])
        result=request_metrics(raw)[0]
        self.assertEqual(result['ttft_s'],10)
        self.assertAlmostEqual(result['max_gap_s'],.3)
        r['token_times_s']=[10.2,10,10.5]
        with self.assertRaises(AssertionError):request_metrics(raw)


if __name__=='__main__':unittest.main()

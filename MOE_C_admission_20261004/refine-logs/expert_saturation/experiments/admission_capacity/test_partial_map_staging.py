"""CPU lifetime/precedence checks; these do not qualify real CUDA execution."""
import itertools
import unittest
import numpy as np
from partial_map_staging import PartialMapBank, submission_schedule


class Stream:
    cuda_stream = 1

    def __init__(self):
        self.queue, self.cursor = [], 0

    def drain(self, stop=None):
        stop = len(self.queue) if stop is None else stop
        while self.cursor < stop:
            self.queue[self.cursor]()
            self.cursor += 1


class Event:
    def record(self, stream):
        self.stream, self.stop = stream, len(stream.queue)

    def synchronize(self):
        self.stream.drain(self.stop)


class Tensor:
    def __init__(self, n, stream):
        self.data, self.stream = np.zeros(n, dtype=np.int32), stream

    def numpy(self):
        return self.data

    def copy_(self, src, *, non_blocking=False):
        # Intentionally read src at DMA execution, not submission.
        self.stream.queue.append(lambda: self.data.__setitem__(slice(None), src.data))


class Torch:
    int32 = 'int32'

    def __init__(self):
        self.stream = Stream()
        self.cuda = self

    def empty(self, n, **kwargs):
        return Tensor(n, self.stream)

    def Event(self):
        return Event()

    def current_stream(self, device):
        return self.stream

    def tensor(self, values, **kwargs):
        result = Tensor(len(values), self.stream)
        self.stream.queue.append(lambda: result.data.__setitem__(slice(None), values))
        self.stream.drain()  # Model blocking factory's queue ordering only.
        self.blocking_factories = getattr(self, 'blocking_factories', 0)+1
        return result


class Checks(unittest.TestCase):
    def test_staging_reuse_preserves_dma_and_consumer_lifetime(self):
        t = Torch()
        bank = PartialMapBank(t, 4, 2, 2, 'cuda')
        seen = []
        for index, mapping in [(0, [0,-1,1,-1]), (1, [-1,1,-1,0]),
                               (0, [1,0,-1,-1]), (1, [-1,-1,0,1])]:
            gpu = bank.materialize(mapping, index)
            t.stream.queue.append(lambda gpu=gpu: seen.append(gpu.data.tolist()))
        t.stream.drain()
        self.assertEqual(seen, [[0,-1,1,-1],[-1,1,-1,0],[1,0,-1,-1],[-1,-1,0,1]])
        self.assertEqual(bank.extra_bytes, dict(host_pinned=32, gpu=32))

    def test_invalid_map_and_stream_fail_before_reuse(self):
        t = Torch()
        bank = PartialMapBank(t, 4, 1, 2, 'cuda')
        for mapping in ([0,1], [0,1,2,-1], [0,1,False,-1]):
            with self.assertRaises(ValueError): bank.materialize(mapping, 0)
        bank.materialize([0,1,-1,-1], 0)
        t.stream.cuda_stream = 2
        with self.assertRaises(RuntimeError): bank.materialize([1,0,-1,-1], 0)

    def test_model_bound_and_both_decisions(self):
        for w,m,h,c in itertools.product((0., .01, .2, 2.), repeat=4):
            r = submission_schedule(pending_ms=w,map_dma_ms=m,host_launch_ms=h,extra_stage_ms=c)
            self.assertAlmostEqual(r['saving_ms'], min(h, w+min(m,h)-c))
            self.assertLessEqual(r['saving_ms'], h+1e-12)
        self.assertEqual(submission_schedule(pending_ms=5,map_dma_ms=.01,
            host_launch_ms=.2,extra_stage_ms=.08)['choice'], 'async')
        self.assertEqual(submission_schedule(pending_ms=0,map_dma_ms=.01,
            host_launch_ms=.2,extra_stage_ms=.08)['choice'], 'blocking')

    def test_shared_two_publications_preserve_values_and_barrier_control(self):
        from types import SimpleNamespace
        from shared_map_staging import SharedMapPair
        local_maps = [[0,-1,1,-1],[-1,1,-1,0]]
        execution_maps = [[0,-1,5,-1,-1,-1],[-1,1,-1,5,-1,-1]]
        for mode, factories in [('blocking',4),('execution_only',2),('all_async',0)]:
            t=Torch(); state=SimpleNamespace(expert_map_device=Tensor(4,t.stream))
            pair=SharedMapPair(t,num_experts=4,private_cap=2,pool_cap=6,device='cuda')
            seen=[]
            for local,execution in zip(local_maps,execution_maps):
                gpu=pair.publish(state,local,execution,mode=mode)
                t.stream.queue.append(lambda gpu=gpu: seen.append(
                    (state.expert_map_device.data.tolist(),gpu.data.tolist())))
            t.stream.drain()
            self.assertEqual(seen,list(zip(local_maps,execution_maps)))
            self.assertEqual(getattr(t,'blocking_factories',0),factories)
            with self.assertRaises(ValueError):pair.publish(state,local_maps[0],None,mode='all_async')

    def test_preparation_cost_cannot_be_hidden_in_launch_only_bound(self):
        from shared_map_staging import publication_timeline
        # The earlier H ceiling applies only after common preparation cancels.
        old = publication_timeline(0,[dict(host_prepare_ms=.3,device_ms=.01,blocking=True)],.1)
        new = publication_timeline(0,[dict(host_prepare_ms=.02,device_ms=.01,blocking=False)],.1)
        self.assertAlmostEqual(old['kernel_start_ms']-new['kernel_start_ms'],.29)
        self.assertGreater(.29,.1)
        first = dict(host_prepare_ms=0.,device_ms=.01,blocking=True)
        d2d = dict(host_prepare_ms=0.,device_ms=.01,blocking=False)
        final = dict(host_prepare_ms=0.,device_ms=.01,blocking=False)
        control = publication_timeline(5,[first,d2d,final],.2)
        both = publication_timeline(5,[dict(first,blocking=False),d2d,final],.2)
        self.assertAlmostEqual(control['kernel_start_ms'],5.21)
        self.assertAlmostEqual(both['kernel_start_ms'],5.03)


if __name__ == '__main__':
    unittest.main()

"""Only EOS termination, output progress and late-arrival boundaries."""
import sys
import os
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from prepare_streaming_recovery import patch_capture
frozen=Path(os.environ.get("STREAM_FROZEN_CAPTURE", str(Path(__file__).resolve().parents[2]/"outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/preparation/pkg/native_capture.py")))
capture_module=ModuleType("stream_capture_test")
exec(compile(patch_capture(frozen.read_text()),str(frozen),"exec"),capture_module.__dict__)
capture_episode=capture_module.capture_episode
from test_native_capture import Clock, Engine, workload


class EosEngine(Engine):
    def __init__(self, clock, frames, stop_sources):
        super().__init__(clock, frames)
        self.stop_sources = stop_sources

    def add_request(self, *args, **kwargs):
        rid = super().add_request(*args, **kwargs)
        self.scheduler.requests[rid].num_output_tokens = 0
        return rid

    def step(self):
        source = self.frames[self.index][0]
        outputs = super().step()
        rid = self.internal[source]
        if rid in self.scheduler.requests:
            self.scheduler.requests[rid].num_output_tokens = len(outputs[0].outputs[0].token_ids)
        if outputs[0].finished and source in self.stop_sources:
            outputs[0].outputs[0].finish_reason = 'stop'
            outputs[0].outputs[0].stop_reason = 50279
        return outputs


def eos_capture(engine, arrivals, mode='eos', count=4):
    fake={'vllm':NS(SamplingParams=lambda **kw:NS(**kw)),
          'vllm.sampling_params':NS(RequestOutputKind=NS(CUMULATIVE='cumulative'))}
    with patch.dict(sys.modules,fake),patch.object(capture_module,'time',engine.clock):
        return capture_episode(engine,workload(arrivals),dict(cap=1,output_tokens=count,output_mode=mode),
                               regime='steady',arrival_scale=1,run_id='eos-test')


class EosCaptureTest(unittest.TestCase):
    def test_early_stop_keeps_later_arrivals_and_actual_output_counts(self):
        e=EosEngine(Clock(),[('r0',2,[8],False),('r0',1,[8,9],True),
                            ('r1',2,[7],True)],{'r0','r1'})
        raw=eos_capture(e,[0,1.0])
        self.assertEqual(raw['status'],'COMPLETE',raw['error'])
        self.assertEqual([len(r['output_token_ids']) for r in raw['requests']],[2,1])
        self.assertEqual([r['stop_reason'] for r in raw['requests']],['stop','stop'])
        self.assertEqual([r['native_stop_reason'] for r in raw['requests']],[50279,50279])
        self.assertGreaterEqual(raw['requests'][1]['admission_s'],1.0)
        self.assertTrue(all(p.max_tokens==4 and p.min_tokens==0 and p.ignore_eos is False for _,p,_ in e.adds))

    def test_terminal_without_new_output_does_not_reset_output_time(self):
        e=EosEngine(Clock(),[('r0',2,[8],False),('r0',1,[8],True)],{'r0'})
        raw=eos_capture(e,[0]);r=raw['requests'][0]
        self.assertEqual(raw['status'],'COMPLETE',raw['error'])
        self.assertEqual(len(r['token_times_s']),1)
        self.assertGreater(r['completion_s'],r['token_times_s'][-1])
        self.assertEqual(raw['output_events'][-1]['chunk_size'],0)

    def test_fixed_mode_still_rejects_early_stop_and_eos_rejects_short_length(self):
        for stops,mode in [({'r0'},'fixed'),(set(),'eos')]:
            e=EosEngine(Clock(),[('r0',2,[8],True)],stops)
            raw=eos_capture(e,[0],mode)
            self.assertEqual(raw['status'],'INCOMPLETE')
            self.assertIn('completion violated',raw['error'])
            self.assertNotIn('schedule',vars(e.scheduler))


if __name__=='__main__':unittest.main()

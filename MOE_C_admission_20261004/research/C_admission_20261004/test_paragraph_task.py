"""Checks for the new task's stopping contract, not performance claims."""
import sys
import types
import unittest
from unittest.mock import patch

from request_measurement import measure_episode
from run import paragraph_population


class ParagraphTaskTest(unittest.TestCase):
    def test_prefixes_preserve_every_identity_and_external_arrival(self):
        work = dict(source_requests=[dict(request_id='a', prompt_sha256='old')],
            actual_prompt_token_ids=[[1, 2, 3, 4, 5, 6, 7]],
            arrival_traces_s={'steady': [3.]})
        selected = paragraph_population(work)
        self.assertEqual(selected['actual_prompt_token_ids'], [[1, 2, 3, 4, 5]])
        self.assertEqual(selected['arrival_traces_s'], work['arrival_traces_s'])
        self.assertEqual(selected['source_requests'][0]['request_id'], 'a')
        self.assertNotIn('prompt_sha256', selected['source_requests'][0])
        self.assertEqual(work['actual_prompt_token_ids'][0], [1, 2, 3, 4, 5, 6, 7])
        line = paragraph_population(work, '\n')
        self.assertEqual(line['actual_prompt_token_ids'], selected['actual_prompt_token_ids'])
        self.assertEqual(line['output_contract']['stop_strings'], ['\n'])
        self.assertEqual(selected['output_contract']['stop_strings'], ['\n\n'])

    def test_string_stop_is_completed_and_visible_text_can_be_absent(self):
        for stop_string, final_text in (('\n\n', 'continuation'), ('\n\n', ''),
                                        ('\n', 'continuation'), ('\n', '')):
            with self.subTest(stop_string=stop_string, final_text=final_text):
                class Engine:
                    active = False
                    vllm_config = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(
                        async_scheduling=False, stream_interval=1))
                    def add_request(self, rid, prompt, params, arrival_time):
                        self.rid, self.params, self.active = rid, params, True
                        return 'internal/'+rid
                    def has_unfinished_requests(self):
                        return self.active
                    def step(self):
                        self.active = False
                        return [types.SimpleNamespace(request_id=self.rid, finished=True,
                            outputs=[types.SimpleNamespace(token_ids=[31, 32], text=final_text,
                                finish_reason='stop', stop_reason=stop_string)])]
                fake = types.ModuleType('vllm')
                fake.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
                sampling = types.ModuleType('vllm.sampling_params')
                sampling.RequestOutputKind = types.SimpleNamespace(CUMULATIVE='cumulative')
                engine = Engine()
                with patch.dict(sys.modules, {'vllm': fake, 'vllm.sampling_params': sampling}):
                    raw = measure_episode(engine,
                        dict(source_requests=[dict(request_id='a', document_id='a')],
                            actual_prompt_token_ids=[[11]], arrival_traces_s={'steady': [0.]}),
                        dict(output_tokens=1024, min_tokens=0, ignore_eos=False, stop_strings=[stop_string]),
                        'steady', 1., 'test')
                self.assertEqual(raw['status'], 'COMPLETE')
                self.assertTrue(engine.params.detokenize)
                self.assertFalse(engine.params.include_stop_str_in_output)
                self.assertEqual(engine.params.stop, [stop_string])
                row = raw['requests'][0]
                self.assertEqual(row['output_token_ids'], [31, 32])
                self.assertEqual(row['native_stop_reason'], stop_string)
                self.assertEqual(row['output_text'], final_text)
                self.assertEqual(row['first_visible_text_s'] is None, not final_text)


if __name__ == '__main__':
    unittest.main()

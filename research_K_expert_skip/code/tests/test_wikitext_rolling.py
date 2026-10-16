"""CPU checks of rolling coverage, identity, fixed cohorts and pooled metrics."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_wikitext_rolling as rolling
import quality_decode as quality
from teacher_force import quality_summary


class Tokenizer:
    bos_token_id, eos_token_id, pad_token_id, unk_token_id = 0, 1, None, None
    all_special_ids = [0, 1]
    name_or_path = 'a-local-path'

    def __init__(self, changed=False, padding=None):
        self.changed, self.padding = changed, padding
        self.backend_tokenizer = self

    def to_str(self):
        return json.dumps({'version': '1.0', 'model': {'vocab': self.get_vocab()},
                           'padding': self.padding, 'truncation': None, 'normalizer': None})

    def get_vocab(self):
        return {'a': 3 if self.changed else 2, '<bos>': 0, '<eos>': 1}

    def encode(self, *args, **kwargs):
        raise AssertionError('Pretokenized input must never re-encode')


def fixture(path, length=30, prefix=4, scored=6):
    tokens = list(range(length))
    rows = list(rolling.rolling_windows(tokens, prefix, scored))
    output = path / 'windows.jsonl'
    output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = {'status': 'READY', 'schema': rolling.SCHEMA, 'split': 'validation',
                'context_prefix_tokens': prefix, 'engine_prompt_tokens': prefix-1,
                'max_scored_tokens': scored, 'token_stream': {'tokens': length, 'sha256': rolling.json_hash(tokens)},
                'tokenizer': rolling.tokenizer_identity(Tokenizer()),
                'windows_file': {'path': output.name, 'sha256': rolling.file_hash(output), 'rows': len(rows)},
                'coverage': {'scored_tokens': length-prefix}}
    source = path / 'manifest.json'
    source.write_text(json.dumps(manifest))
    return source, rows, manifest


class RollingTest(unittest.TestCase):
    def test_complete_coverage_and_true_decode_bridge_for_exact_and_short_tails(self):
        for length in (5, 10, 11, 28, 29, 30):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as directory:
                source, rows, _ = fixture(Path(directory), length)
                docs, selection, _ = rolling.load_rolling_documents(source)
                scored_offsets, scored_values = [], []
                for doc in docs:
                    self.assertEqual(len(doc['prompt_token_ids']), 3)
                    self.assertEqual(doc['forced_tokens'][0], doc['prefill_bridge_token_offset'])
                    self.assertEqual(doc['prompt_token_ids'] + doc['forced_tokens'],
                                     list(range(doc['context_start_token'], doc['score_stop_token'])))
                    # Position zero is emitted from prefill; only positions >=1 count.
                    scored_offsets.extend(range(doc['score_start_token'], doc['score_stop_token']))
                    scored_values.extend(doc['forced_tokens'][1:])
                self.assertEqual(scored_offsets, list(range(4, length)))
                self.assertEqual(scored_values, scored_offsets)
                self.assertEqual(selection['selected_scored_tokens'], length-4)
                self.assertTrue(selection['complete_split_after_initial_context'])
                # No incomplete cohort or short tail is lost, and all rows of a
                # cohort end together under teacher_force's strict batch contract.
                groups = list(quality.document_groups(docs, 3))
                self.assertEqual(sum(map(len, groups)), len(docs))
                for group in groups:
                    self.assertLessEqual(len(group), 3)
                    self.assertEqual(len({len(doc['forced_tokens']) for doc in group}), 1)

    def test_limit_is_explicit_and_identity_checks_never_encode(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, manifest = fixture(Path(directory))
            docs, selection, _ = rolling.load_rolling_documents(source, 2)
            self.assertEqual(len(docs), 2)
            self.assertEqual(selection['selected_scored_span'], [4, 16])
            self.assertFalse(selection['complete_split_after_initial_context'])
            self.assertEqual(selection['unselected_suffix_tokens'], 14)
            runtime = Tokenizer(padding={'pad_id': 2})
            runtime.name_or_path = 'another-runtime-path'
            rolling.validate_tokenizer_identity(manifest['tokenizer'], runtime)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                rolling.validate_tokenizer_identity(manifest['tokenizer'], Tokenizer(changed=True))

    def test_gap_bad_context_and_corruption_are_rejected(self):
        for fault in ('gap', 'context', 'sha'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                source, rows, manifest = fixture(path)
                if fault == 'gap':
                    rows[1]['score_start_token'] += 1
                else:
                    rows[1]['token_ids'][0] += 1
                windows = path / 'windows.jsonl'
                windows.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                if fault != 'sha':
                    manifest['windows_file']['sha256'] = rolling.file_hash(windows)
                    source.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    rolling.load_rolling_documents(source)
        with self.assertRaises(ValueError):
            list(rolling.rolling_windows([1, 2, 3], prefix=4))

    def test_pooled_metrics_and_runtime_phase_guard(self):
        rows = []
        for group, count in [('a', 7), ('b', 2)]:
            rows += [dict(group=group, doc_key='window', nll=float(i+1),
                          baseline_nll=float(i)+.75, kl_base_to_policy=.01*i) for i in range(count)]
        combined = quality.merge_quality_summaries([quality_summary(rows[:7]), quality_summary(rows[7:])])
        expected = quality_summary(rows)
        for key, value in expected.items():
            self.assertAlmostEqual(combined[key], value)
        with tempfile.TemporaryDirectory() as directory:
            source, _, _ = fixture(Path(directory), 5)
            docs, _, _ = rolling.load_rolling_documents(source)
            metrics = Path(directory) / 'metrics.jsonl'
            records = [dict(doc_key=docs[0]['doc_key'], group='g', arm='native', batch_rows=1,
                            output_position=position, target_token=target,
                            prediction_phase='prefill_next_token' if position == 0 else 'decode',
                            context_tokens=3+position) for position, target in enumerate(docs[0]['forced_tokens'])]
            metrics.write_text(''.join(json.dumps(row)+'\n' for row in records))
            quality.read_and_validate_metrics(metrics, 'g', docs, 'native', 1, 1)
            records[1]['prediction_phase'] = 'prefill_next_token'
            metrics.write_text(''.join(json.dumps(row)+'\n' for row in records))
            with self.assertRaisesRegex(RuntimeError, 'true-decode contract'):
                quality.read_and_validate_metrics(metrics, 'g', docs, 'native', 1, 1)

    def test_old_text_cli_defaults_remain(self):
        args = quality.build_parser().parse_args(['--model', 'local', '--articles', 'input.jsonl', '--output', 'out.json'])
        self.assertEqual((args.tokens, args.prefix_tokens, args.limit_docs), (32, 128, 32))
        self.assertIsNone(args.tokenized_windows)
        self.assertIsNone(args.limit_windows)


if __name__ == '__main__':
    unittest.main()

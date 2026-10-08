"""Reproduce the already frozen GSM8K numeric metric, without GPU imports.

Use an installed tokenizers package, or explicitly pass an existing native
extension with --tokenizers-extension. No package search or installation occurs.
The rule is loaded from the preserved initial scoring record, not retuned here.
"""
import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tokenizers(extension):
    if extension is None:
        import tokenizers
        return tokenizers
    spec = importlib.util.spec_from_file_location('tokenizers', extension)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load the explicitly provided tokenizers extension')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(args):
    definition = read(args.definition)
    pattern = re.compile(definition['number_regex'])
    binding = load_tokenizers(args.tokenizers_extension)
    tokenizer = binding.Tokenizer.from_file(str(args.tokenizer))
    assert digest(args.tokenizer) == definition['provenance']['tokenizer_sha256']
    assert digest(args.sources) == definition['provenance']['source_gsm8k_sha256']
    sources = {row['source_index']: row['source'] for row in read(args.sources)}
    frozen = {row['external_id']: row for row in definition['rows']}
    audits, rows, failures = {}, {}, []
    paths = sorted(args.group.rglob('raw.json'))
    if not paths:
        raise ValueError('No completed raw episodes in local group')
    if read(args.group / 'status.json').get('status') != 'COMPLETE':
        raise ValueError('Group is not COMPLETE')
    raw_hashes, input_hashes = {}, {}
    for path in paths:
        cell = path.parent
        name = str(cell.relative_to(args.group))
        if read(cell / 'status.json').get('status') != 'COMPLETE':
            raise ValueError(f'Episode is not COMPLETE: {name}')
        raw, inputs = read(path), read(cell / 'inputs.json')
        assert len(raw['requests']) == len(inputs)
        scored = []
        identities = set()
        for index, (request, inp) in enumerate(zip(raw['requests'], inputs)):
            if inp.get('task') != 'gsm8k':
                continue
            external = request['external_id']
            assert external == f'measured/E{index:03d}'
            expected = frozen[external]
            prompt_hash = hashlib.sha256(json.dumps(inp['prompt_token_ids'],
                separators=(',', ':')).encode()).hexdigest()
            assert inp['example_index'] == expected['gsm8k_test_index']
            assert Decimal(inp['gold'].replace(',', '')) == Decimal(expected['gold'].replace(',', ''))
            assert prompt_hash == expected['prompt_token_ids_sha256']
            identities.add(external)
            source = sources[inp['example_index']]
            assert source['answer'] == inp['reference_answer']
            gold = Decimal(source['answer'].rsplit('####', 1)[1].strip().replace(',', ''))
            assert gold == Decimal(inp['gold'].replace(',', ''))
            text = tokenizer.decode(request['output_token_ids'], skip_special_tokens=True)
            matches = list(pattern.finditer(text))
            last = matches[-1] if matches else None
            value = Decimal(last.group().replace(',', '')) if last else None
            match = value == gold
            score = dict(text=text, output_token_ids=request['output_token_ids'],
                finish_reason=request['finish_reason'], stop_reason=request.get('stop_reason'),
                extracted_numeric=last.group() if last else None,
                normalized_prediction=str(value) if value is not None else None,
                numeric_span=list(last.span()) if last else None,
                numeric_exact_match=match, truncated=request['finish_reason'] == 'length',
                unparseable=last is None,
                stop_numeric_match=match and request['finish_reason'] == 'stop')
            record = rows.setdefault(external, dict(external_id=external,
                gsm8k_test_index=inp['example_index'], gold=inp['gold'],
                prompt_token_ids_sha256=prompt_hash,
                reference_answer=inp['reference_answer'], arms={}))
            assert record['gold'] == inp['gold'] and record['gsm8k_test_index'] == inp['example_index']
            if cell.name in record['arms']:
                raise ValueError('Duplicate arm basename; use a single experiment group')
            record['arms'][cell.name] = score
            scored.append(score)
        assert identities == set(frozen), 'Frozen short-task population changed'
        audits[name] = dict(requests=len(scored),
            **{field: sum(bool(row[field]) for row in scored) for field in
               ('stop_numeric_match', 'numeric_exact_match', 'truncated', 'unparseable')})
        raw_hashes[name], input_hashes[name] = digest(path), digest(cell / 'inputs.json')
    verified = None
    if args.verify_against:
        reference = read(args.verify_against)
        expected = {row['external_id']: row for row in reference['rows']}
        assert set(expected) == set(rows)
        count = 0
        for key, record in rows.items():
            assert set(record['arms']) == set(expected[key]['arms'])
            for arm, actual in record['arms'].items():
                previous = expected[key]['arms'][arm]
                for field in ('text', 'output_token_ids', 'finish_reason', 'stop_reason',
                              'extracted_numeric', 'numeric_span', 'numeric_exact_match',
                              'truncated', 'unparseable', 'stop_numeric_match'):
                    if actual[field] != previous[field]:
                        failures.append(dict(request=key, arm=arm, field=field))
                count += 1
        verified = dict(path=str(args.verify_against), sha256=digest(args.verify_against),
                        request_arm_scores=count, failures=failures)
    return dict(schema='E.short_quality_reproduction.v1', group=str(args.group),
        metric_definition={key: definition[key] for key in
            ('extraction_rule', 'number_regex', 'primary_metric', 'diagnostic_metric')},
        provenance=dict(definition_sha256=digest(args.definition),
            tokenizer_sha256=digest(args.tokenizer), source_sha256=digest(args.sources),
            decode_skip_special_tokens=True,
            binding_version=binding.__version__, binding_path=str(binding.__file__),
            binding_file_sha256=digest(Path(binding.__file__)),
            raw_sha256=raw_hashes, inputs_sha256=input_hashes),
        summary=audits, rows=list(rows.values()), verification=verified,
        interpretation=definition['interpretation_limit'])


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--definition', type=Path, default=root / 'summary256_short_quality.json')
    parser.add_argument('--tokenizer', type=Path, default=root / 'inputs_pro6000/sources/tokenizer.json')
    parser.add_argument('--sources', type=Path, default=root / 'inputs_pro6000/sources/gsm8k_test_selected.json')
    parser.add_argument('--tokenizers-extension', type=Path)
    parser.add_argument('--verify-against', type=Path)
    args = parser.parse_args()
    args.group = args.group.resolve()
    result = evaluate(args)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    verification = result['verification']
    if verification:
        verification = {**verification, 'failure_count': len(verification['failures']),
                        'failures': verification['failures'][:5]}
    print(json.dumps(dict(output=str(args.out), summary=result['summary'],
                         verification=verification), ensure_ascii=False, indent=2))
    return int(bool(result['verification'] and result['verification']['failures']))


if __name__ == '__main__':
    raise SystemExit(main())

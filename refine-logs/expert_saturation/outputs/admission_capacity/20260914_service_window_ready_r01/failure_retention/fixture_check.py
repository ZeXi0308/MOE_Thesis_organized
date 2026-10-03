"""Two CPU failure fixtures for the extracted frozen runner fragment only."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
expected_hash, source_name = (ROOT / 'source.sha256').read_text().strip().split('  ', 1)
assert sha256(Path(source_name).read_bytes()).hexdigest() == expected_hash


def exercise(case, version):
    out = ROOT / 'fixture_artifacts' / case / version
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'raw.json').exists(), 'Use a fresh fixture output directory'
    partial = case == 'drain_raises'
    captured = {
        'status': 'INCOMPLETE' if partial else 'COMPLETE',
        'error': 'fixture capture stopped with partial output' if partial else None,
        'requests': [
            {'request_id': 'r0', 'status': 'completed', 'output_token_ids': [11, 12]},
            {'request_id': 'r1', 'status': 'unfinished' if partial else 'completed',
             'output_token_ids': [21]},
        ],
        'output_events': [{'request_id': 'r1', 'new_token_ids': [21], 'received_s': 1.25}],
        'observation_end_s': 2.5,
    }
    before = {'fixture_gpu_state': 'CPU_ONLY'}
    events = []
    raw_writes = []

    def dump(path, value):
        events.append('dump:' + path.name)
        if path.name == 'raw.json':
            raw_writes.append(deepcopy(value))
        path.write_text(json.dumps(value, indent=2) + '\n')

    def capture(*args, **kwargs):
        events.append('capture_return')
        return deepcopy(captured)

    def drain(engine):
        events.append('drain_enter')
        assert not raw_writes, 'Large raw JSON was written before drain'
        if partial:
            events.append('drain_raise')
            raise RuntimeError('fixture drain failure')
        events.append('drain_return')
        return {'calls': 0, 'seconds': 0.0}

    def uninstall_selective():
        events.append('cleanup_selective')
        return {'applied_rotations': 2 if partial else 1}

    def uninstall():
        events.append('cleanup_observer')

    names = dict(capture_with_memory=capture, capture_episode=None, engine=None,
                 workload={}, config={}, drain_offload=drain, dump=dump, out=out,
                 uninstall_selective=uninstall_selective, uninstall=uninstall,
                 offload_data={}, decisions=[], before=before,
                 args=SimpleNamespace(completion_policy='native'),
                 selective_data={'applied_rotations': 2 if partial else 1})
    expected_error = ('fixture drain failure' if partial else
                      'Repeated intervention did not execute at least twice')
    try:
        code = (ROOT / f'runner_fragment_{version}.py').read_text()
        exec(compile(code, f'runner_fragment_{version}.py', 'exec'), names)
    except RuntimeError as error:
        assert str(error) == expected_error
    else:
        raise AssertionError('Fixture did not reach the requested failure')
    expected = dict(captured, gpu_before=before)
    if version == 'before':
        assert raw_writes == [] and not (out / 'raw.json').exists()
    else:
        assert len(raw_writes) == 1
        assert json.loads((out / 'raw.json').read_text()) == expected
        assert events.index('drain_enter') < events.index('dump:raw.json')
        assert events.index('cleanup_observer') < events.index('dump:raw.json')
    return dict(version=version, exception=expected_error,
                raw_write_count=len(raw_writes), raw_exists=(out / 'raw.json').exists(),
                retained_request_count=len(raw_writes[0]['requests']) if raw_writes else 0,
                retained_output_ids=[r['output_token_ids'] for r in raw_writes[0]['requests']]
                    if raw_writes else [],
                observation_end_s=raw_writes[0]['observation_end_s'] if raw_writes else None,
                events=events)


results = {case: [exercise(case, version) for version in ('before', 'after')]
           for case in ('drain_raises', 'insufficient_applied_rotations')}
report = dict(status='PASS', source=source_name, source_sha256=expected_hash,
              fixture_count=2, cases=results,
              scope='CPU execution of actual runner fragments; no engine, GPU, or capture-internal changes. '
                    'Failure of the raw write itself and process termination are outside this patch.')
(ROOT / 'fixture_output.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

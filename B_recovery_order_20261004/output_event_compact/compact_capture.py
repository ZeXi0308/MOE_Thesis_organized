"""Frozen capture with optional primitive-tuple output-event storage only."""
import hashlib
from pathlib import Path

PARENT = Path(__file__).resolve().parents[1]/'pkg/request_measurement.py'
PARENT_SHA = '1b322be02505381dedb0aaf47f58513dfef61d60bf98e8d9590e18e3012534e4'
SOURCE = PARENT.read_text()
if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
    raise RuntimeError('Frozen request measurement source changed')


def _once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Capture transformation boundary changed: '+old)
    return source.replace(old, new)


def compact_source():
    source = _once(SOURCE, '''                event = dict(request_id=rid, external_request_id=output.request_id,
                    engine_call_index=call_count - 1, received_s=received, cumulative_tokens=len(tokens),
                    new_token_ids=[], chunk_size=0, finished=bool(output.finished),
                    finish_reason=completion.finish_reason, prefix_valid=False)
                events.append(event)''', '''                event_index = len(events)
                events.append((rid, output.request_id, call_count - 1, received, len(tokens),
                               0, bool(output.finished), completion.finish_reason, False))''')
    source = _once(source,
        '                event.update(new_token_ids=added, chunk_size=len(added), prefix_valid=True)',
        '''                event = events[event_index]
                events[event_index] = (event[0], event[1], event[2], event[3], event[4],
                                       len(added), event[6], event[7], True)''')
    source = _once(source, '    end = now()\n',
        '    end = now()\n    events, compact_stats = _compact_materialize(events, rows, time.perf_counter)\n')
    return _once(source, '    return result\n',
        "    result['output_event_storage'] = compact_stats\n    return result\n")


def _materialize(events, rows, clock):
    started = clock()
    records, valid = [], 0
    for event in events:
        if type(event) is not tuple or len(event) != 9:
            raise RuntimeError('Compact capture did not retain tuple9 records')
        rid, external, call, received, cumulative, chunk, finished, reason, prefix = event
        # The unchanged cumulative-prefix check guarantees each valid historical
        # delta is a slice of the final retained request sequence, even on failure.
        added = rows[rid]['output_token_ids'][cumulative-chunk:cumulative] if prefix else []
        records.append(dict(request_id=rid, external_request_id=external,
            engine_call_index=call, received_s=received, cumulative_tokens=cumulative,
            new_token_ids=added, chunk_size=chunk, finished=finished,
            finish_reason=reason, prefix_valid=prefix))
        valid += bool(prefix)
    stopped = clock()
    elapsed = stopped-started
    return records, dict(mode='compact', measurement_representation='primitive_tuple9',
        event_count=len(events), compact_tuple_records=len(events),
        tuple_append_count=len(events), valid_tuple_replace_count=valid,
        legacy_dict_records=0, materialized_event_count=len(records),
        materialization_s=elapsed, materialization_applied=True,
        materialization_start_host_perf_s=started, materialization_end_host_perf_s=stopped)


def _compile(source, mode):
    namespace = dict(__name__='output_capture_'+mode, __file__=str(PARENT),
                     _compact_materialize=_materialize)
    exec(compile(source, str(PARENT) if mode == 'legacy' else str(PARENT)+'[compact]', 'exec'), namespace)
    return namespace['measure_episode']


COMPACT_SOURCE = compact_source()
_CAPTURES = dict(legacy=_compile(SOURCE, 'legacy'), compact=_compile(COMPACT_SOURCE, 'compact'))


def get_capture(mode):
    """Legacy returns the unmodified original function; signatures are identical."""
    if mode not in _CAPTURES:
        raise ValueError(mode)
    return _CAPTURES[mode]


def annotate(raw, mode):
    """Common post-return metadata only; no clock calls or measurement changes."""
    get_capture(mode)
    if mode == 'legacy':
        count = len(raw['output_events'])
        stats = dict(mode='legacy', measurement_representation='original_dict_with_delta_list',
            event_count=count, compact_tuple_records=0, tuple_append_count=0,
            valid_tuple_replace_count=0, legacy_dict_records=count,
            materialized_event_count=0, materialization_s=0.0, materialization_applied=False,
            materialization_start_host_perf_s=None, materialization_end_host_perf_s=None)
        raw['output_event_storage'] = stats
    else:
        stats = raw['output_event_storage']
        if stats['mode'] != mode or stats['materialized_event_count'] != len(raw['output_events']):
            raise RuntimeError('Compact capture execution evidence mismatch')
    stats.update(parent_sha256=PARENT_SHA,
        compiled_source_sha256=hashlib.sha256((SOURCE if mode == 'legacy' else COMPACT_SOURCE).encode()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        counts_semantics='Retained records count actual appends; valid tuple records count successful replacements',
        timing_semantics='Materialization occurs strictly after captured observation_end_s; '
                         'legacy performs no separate materialization. Capture return includes this artifact cost.')
    return raw

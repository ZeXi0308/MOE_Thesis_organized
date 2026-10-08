"""Frozen/compact capture equivalence with deterministic CPU engines and clock."""
import inspect
import sys
from types import ModuleType, SimpleNamespace as NS

import compact_capture as compact


class Clock:
    def __init__(self):
        self.ticks = 0
        self.calls = []
    def perf_counter(self):
        self.ticks += 1
        result = self.ticks/1000
        self.calls.append(result)
        return result
    def time(self):
        return 1000000.0
    def sleep(self, delay):
        self.ticks += round(delay*1000)


class Engine:
    def __init__(self, script, clock):
        self.script, self.clock = list(script), clock
        self.active, self.external = set(), []
        self.step_clock_counts, self.parameters = [], []
        self.vllm_config = NS(scheduler_config=NS(async_scheduling=False, stream_interval=1))
        self.scheduler = NS(_preempt_request=lambda request, *args, **kwargs: None)
        self.engine_core = NS(engine_core=NS(scheduler=self.scheduler))
    def add_request(self, external, prompt, params, arrival_time):
        self.external.append(external); self.active.add(external)
        self.parameters.append((external, prompt, vars(params), arrival_time))
        return 'internal:'+external
    def has_unfinished_requests(self):
        return bool(self.active)
    def step(self):
        self.step_clock_counts.append(len(self.clock.calls))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        outputs = []
        for index, tokens, finished, reason in item:
            external = self.external[index]
            outputs.append(NS(request_id=external, finished=finished,
                outputs=[NS(token_ids=list(tokens), finish_reason=reason)]))
            if finished:
                self.active.remove(external)
        if not self.script:
            self.active.clear()  # Includes a deliberate premature-drain fixture.
        return outputs


def run(mode, script, count=1, max_seconds=10):
    clock = Clock(); engine = Engine(script, clock)
    workload = dict(source_requests=[dict(request_id='r'+str(i), document_id='d'+str(i)) for i in range(count)],
        actual_prompt_token_ids=[[1, 2] for _ in range(count)], arrival_traces_s={'steady':[0.0]*count})
    config = dict(output_tokens=3, ignore_eos=True, min_tokens=0)
    fn = compact.get_capture(mode)
    original_clock = fn.__globals__['time']
    original_materialize = fn.__globals__['_compact_materialize']
    observed = []
    def materialize(events, rows, now):
        assert all(type(e) is tuple and len(e) == 9 for e in events)
        assert all(all(type(v) in (str, int, float, bool, type(None)) for v in e) for e in events)
        observed.append(len(events))
        return original_materialize(events, rows, now)
    fn.__globals__['time'] = clock
    fn.__globals__['_compact_materialize'] = materialize
    before_preempt = engine.scheduler._preempt_request
    try:
        raw = fn(engine, workload, config, 'steady', 1.0, 'cpu', max_seconds,
                 record_preemptions=True)
        compact.annotate(raw, mode)
    finally:
        fn.__globals__['time'] = original_clock
        fn.__globals__['_compact_materialize'] = original_materialize
    assert engine.scheduler._preempt_request is before_preempt
    return raw, clock, engine, observed


fake_vllm = ModuleType('vllm'); fake_sampling = ModuleType('vllm.sampling_params')
fake_vllm.SamplingParams = lambda **kwargs: NS(**kwargs)
fake_sampling.RequestOutputKind = NS(CUMULATIVE='cumulative')
saved = {name:sys.modules.get(name) for name in ('vllm', 'vllm.sampling_params')}
sys.modules.update({'vllm':fake_vllm, 'vllm.sampling_params':fake_sampling})
cases = [
    ('normal', [[(0,[10],False,None)],[(0,[10,11],False,None)],[(0,[10,11,12],True,'length')]],1,10),
    ('multi_token', [[(0,[10],False,None)],[(0,[10,11,12],True,'length')]],1,10),
    ('invalid_prefix', [[(0,[10],False,None)],[(0,[99,11],False,None)]],1,10),
    ('overlength', [[(0,[10],False,None)],[(0,[10,11,12,13],True,'length')]],1,10),
    ('engine_error', [[(0,[10],False,None)],RuntimeError('native engine failure')],1,10),
    ('partial', [[(0,[10,11,12],True,'length'),(1,[20],False,None)],RuntimeError('native engine failure')],2,10),
    ('unfinished', [[(0,[10],False,None)]],1,10),
    ('invalid_finish', [[(0,[10],True,'length')]],1,10),
    ('timeout', [[(0,[10],False,None)],[(0,[10,11],False,None)],[(0,[10,11,12],True,'length')]],1,.007),
]
try:
    assert compact.get_capture('legacy') is compact.get_capture('legacy')
    assert str(inspect.signature(compact.get_capture('legacy'))) == str(inspect.signature(compact.get_capture('compact')))
    for name, script, count, deadline in cases:
        a, ca, ea, oa = run('legacy', script, count, deadline)
        b, cb, eb, ob = run('compact', script, count, deadline)
        sa, sb = a.pop('output_event_storage'), b.pop('output_event_storage')
        assert a == b, name  # Every original raw field, timestamp, event and failure.
        assert ea.parameters == eb.parameters and ea.step_clock_counts == eb.step_clock_counts
        assert cb.calls[:-2] == ca.calls and len(cb.calls) == len(ca.calls)+2
        assert oa == [] and ob == [len(a['output_events'])]
        assert sb['tuple_append_count'] == sb['materialized_event_count'] == len(a['output_events'])
        assert sb['valid_tuple_replace_count'] == sum(e['prefix_valid'] for e in a['output_events'])
        assert sb['materialization_applied'] and not sa['materialization_applied']
        assert sb['materialization_start_host_perf_s'] > b['measurement_origin_perf_counter_s']+b['observation_end_s']
        assert sb['materialization_end_host_perf_s']-sb['materialization_start_host_perf_s'] == sb['materialization_s']
        if name in ('invalid_prefix', 'overlength'):
            assert not b['output_events'][-1]['prefix_valid'] and b['output_events'][-1]['new_token_ids'] == []
        if name in ('normal', 'multi_token'):
            assert a['status'] == 'COMPLETE'
        else:
            assert a['status'] == 'INCOMPLETE', name
        if name == 'partial':
            assert [r['status'] for r in a['requests']] == ['completed', 'failed']
    print('PASS: original/compact complete raw equality on 9 cases; actual primitive tuples; exact delta reconstruction; unchanged measurement clock/arrivals/SamplingParams/horizon; failures and hook cleanup; post-end materialization only')
finally:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

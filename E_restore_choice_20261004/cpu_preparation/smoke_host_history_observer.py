"""CPU protocol smoke for observer transparency, not native engine correctness."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from host_history_observer import HostHistoryObserver


class Fixture:
    def __init__(self):
        self.req = NS(request_id='r', num_preemptions=1, num_tokens=128,
                      num_output_tokens=16, num_computed_tokens=32)
        self.group = NS(next_stored_chunk_idx=7, num_hit_chunks=2,
                        offload_keys=list(range(8)))
        self.state = NS(req=self.req, config=NS(kv_group_configs=[NS(tokens_per_chunk=16)]),
                        group_states=[self.group])
        self._req_status = {'r': self.state}
        self._jobs = {}
        self.alloc_result, self.store_result, self.update_result = object(), {}, object()
        self.calls = []
        self.failure = None

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.calls.append('alloc')
        if num_external_tokens:
            self.group.next_stored_chunk_idx = num_external_tokens//16
        return self.alloc_result

    def _build_store_jobs(self, scheduler_output):
        self.calls.append('store')
        if self.failure is not None:
            raise self.failure
        self.group.next_stored_chunk_idx = 4
        self._jobs[9] = NS(is_store=True, keys={2, 3}, pending_count=2)
        self.store_result[9] = NS(req_id='r', src_spec=NS(block_ids=[3, 4]))
        return self.store_result

    def update_connector_output(self, connector_output):
        self.calls.append('update')
        for jid, count in connector_output.kv_connector_worker_meta.completed_jobs.items():
            self._jobs[jid].pending_count -= count
            if self._jobs[jid].pending_count == 0:
                del self._jobs[jid]
        return self.update_result


def main():
    clocks = []
    def clock():
        clocks.append(1)
        return len(clocks)*0.1
    output = NS(num_scheduled_tokens={'r': 32}, finished_req_ids=set())
    native = Fixture()
    original_alloc = native.update_state_after_alloc
    original_store = native._build_store_jobs
    original_update = native.update_connector_output
    pending = {'r': dict(event=12, preemptions=1, known_tokens=128,
                         generated_tokens=16, host_hit_tokens=32, action='host')}
    observer = HostHistoryObserver(native, clock, NS(pending=pending), enabled=True)
    with observer:
        observer.service_step(416)
        assert native.update_state_after_alloc(native.req, object(), 32) is native.alloc_result
        assert native._build_store_jobs(output) is native.store_result
        assert native.calls == ['alloc', 'store'] and len(clocks) == 2
        a, s = observer.records
        assert a['before']['groups'][0]['next_stored_chunk_idx'] == 7
        assert a['after']['groups'][0]['next_stored_chunk_idx'] == 2
        assert a['lookup']['event'] == 12
        assert s['before'][0]['last_allocation']['record_index'] == a['record_index']
        assert s['before'][0]['last_allocation']['lookup']['host_hit_tokens'] == 32
        assert s['after'][0]['groups'][0]['next_stored_chunk_idx'] == 4
        assert s['jobs'][0]['groups'][0]['chunk_intervals'] == [[2, 4]]
        assert s['jobs'][0]['unmatched_key_count'] == 0
        completion = NS(kv_connector_worker_meta=NS(completed_jobs={9: 1}))
        assert native.update_connector_output(completion) is native.update_result
        assert native.update_connector_output(completion) is native.update_result
        u1, u2 = observer.records[2:]
        assert u1['before'][0]['creation_record_index'] == s['record_index']
        assert u1['before'][0]['pending_count_before'] == 2
        assert u1['after'][0]['pending_count_after'] == 1
        assert not u1['after'][0]['removed_by_native']
        assert u2['before'][0]['pending_count_before'] == 1
        assert u2['after'][0]['pending_count_after'] == 0
        assert u2['after'][0]['removed_by_native']
        native._jobs[8] = NS(pending_count=1)
        unknown = NS(kv_connector_worker_meta=NS(completed_jobs={8: 1}))
        assert native.update_connector_output(unknown) is native.update_result
        assert len(clocks) == 4
        observer.enabled = False
        assert native.update_state_after_alloc(native.req, object(), 0) is native.alloc_result
        assert len(clocks) == 4
        observer.enabled = True
        native.req.num_preemptions = 0
        assert native._build_store_jobs(output) is native.store_result
        assert len(clocks) == 4
    assert native.update_state_after_alloc == original_alloc
    assert native._build_store_jobs == original_store
    assert native.update_connector_output == original_update
    assert 'update_state_after_alloc' not in vars(native)
    assert '_build_store_jobs' not in vars(native)
    assert not observer.errors

    disabled = HostHistoryObserver(native, clock, enabled=False)
    with disabled:
        assert 'update_state_after_alloc' not in vars(native)
        native.update_state_after_alloc(native.req, object(), 0)
    assert len(clocks) == 4 and not disabled.records

    native.req.num_preemptions = 1
    marker = RuntimeError('native sentinel')
    native.failure = marker
    exception_observer = HostHistoryObserver(native, clock, enabled=True)
    try:
        with exception_observer:
            native._build_store_jobs(output)
    except RuntimeError as caught:
        assert caught is marker
    else:
        raise AssertionError('Native exception was swallowed')
    assert exception_observer.records[0]['status'] == 'native_raised'
    assert native._build_store_jobs == original_store

    def broken_clock():
        raise ValueError('observer-only failure')
    broken = HostHistoryObserver(native, broken_clock, enabled=True)
    with broken:
        assert native.update_state_after_alloc(native.req, object(), 0) is native.alloc_result
    assert len(broken.errors) == 1 and not broken.records
    print('PASS: cursor/job/completion attribution; original return identity and exception; '
          'restore; disabled/unpreempted zero clocks; observer-error isolation')


if __name__ == '__main__':
    main()

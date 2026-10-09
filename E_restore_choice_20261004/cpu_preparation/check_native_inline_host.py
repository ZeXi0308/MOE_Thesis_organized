"""CPU-only adapter smoke: hook order, narrow action domain, native ack plumbing.

This deliberately does not claim GPU transfer correctness or service benefit.
Run from the E research directory: python cpu_preparation/check_native_inline_host.py
"""
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_inline_host import NativeInlineHostRecovery


class OffloadPolicy(Enum):
    BLOCK_LEVEL = 1


class CPUOffloadingManager:
    pass


class CPUOffloadingWorker:
    def __init__(self, log):
        self.log = log

    def wait(self, job_ids):
        assert job_ids == {10}
        self.log.append('wait')


class SchedulerSide:
    def __init__(self, env):
        self.env = env
        self.manager = CPUOffloadingManager()
        self.config = NS(num_workers=1, kv_group_configs=[NS(tokens_per_chunk=16,
            tokens_per_block=16, sliding_window_size_in_chunks=None, is_eagle_group=False)])
        self._req_status = {'r': NS(req=env.req, transfer_jobs=set(),
            offloading_context=NS(policy=OffloadPolicy.BLOCK_LEVEL),
            group_states=[NS(next_stored_chunk_idx=2)])}
        self._current_batch_load_jobs = {}
        self._jobs = {}

    def get_num_new_matched_tokens(self, request, local):
        self.env.log.append('lookup')
        self.env.sel.pending['r'] = dict(event=1, eligible=self.env.eligible,
            joint_capacity=self.env.eligible, action='host', fallback='none')
        return self.env.hit, bool(self.env.hit)

    def update_state_after_alloc(self, request, blocks, external):
        self.env.log.append('alloc')
        self._req_status['r'].group_states[0].next_stored_chunk_idx = external//16
        self._current_batch_load_jobs[10] = NS(req_id='r')
        self._jobs[10] = NS(is_store=False)

    def build_connector_meta(self, output):
        self.env.log.append('build')
        meta = NS(load_jobs=self._current_batch_load_jobs,
                  store_jobs={20: NS(req_id='r')} if self.env.add_store else {})
        self._current_batch_load_jobs = {}
        return meta


class WorkerSide:
    def __init__(self, env):
        self.env = env
        self.connector_worker = NS(worker=CPUOffloadingWorker(env.log), _load_jobs={},
            _connector_worker_meta=NS(completed_jobs={}, transfer_stats=object()))
        self._connector_metadata = None

    def start_load_kv(self, *args, **kwargs):
        if self.env.native_error is not None:
            raise self.env.native_error
        self.env.log.append('submit_native')
        self.connector_worker._load_jobs.update(
            {jid: job.req_id for jid, job in self._connector_metadata.load_jobs.items()})
        return 'native_start_result'

    def get_finished(self, *args, **kwargs):
        if self.env.native_error is not None:
            raise self.env.native_error
        self.env.log.append('prepare_store_native')
        self.connector_worker._connector_worker_meta.completed_jobs[10] = 1
        self.connector_worker._load_jobs.pop(10)
        self.env.log.append('completion_native')
        return {'unrelated_send'}, {'r', 'unrelated_receive'}


def make(known=35, hit=32, eligible=True):
    env = NS(log=[], hit=hit, eligible=eligible, add_store=False, native_error=None)
    env.req = NS(request_id='r', all_token_ids=list(range(known)), num_tokens=known,
        num_output_tokens=3, num_preemptions=1, num_computed_tokens=0,
        sampling_params=object(), status=NS(name='PREEMPTED'), stop_reason=None)
    cs = SchedulerSide(env)
    tick = iter(range(1000))
    env.sel = NS(cs=cs, policy='host', target_spec=None, enabled=True, pending={},
        scheduler=NS(scheduler_reserve_full_isl=True, needs_kv_cache_zeroing=False),
        _validate_headroom_scope=lambda: None, now=lambda: next(tick))
    env.worker = WorkerSide(env)
    env.probe = NativeInlineHostRecovery(env.sel, env.worker, enabled=True)
    return env


def commit(env):
    cs = env.sel.cs
    assert cs.get_num_new_matched_tokens(env.req, 0) == (32, False)
    cs.update_state_after_alloc(env.req, None, 32)
    env.req.num_computed_tokens = 32
    env.req.status = NS(name='RUNNING')
    meta = cs.build_connector_meta(NS(num_scheduled_tokens={'r': 3}))
    env.worker._connector_metadata = meta
    return meta


def expect_error(fn, error_type):
    try:
        fn()
    except error_type:
        return
    raise AssertionError('expected '+error_type.__name__)


def main():
    env = make()
    before = env.probe._state(env.req)
    original = env.sel.cs.get_num_new_matched_tokens
    stats = env.worker.connector_worker._connector_worker_meta.transfer_stats
    with env.probe:
        meta = commit(env)
        assert env.worker.start_load_kv(None) == 'native_start_result'
        env.log.append('forward_return')
        assert env.log.index('submit_native') < env.log.index('wait') < env.log.index('forward_return')
        assert env.worker.get_finished(set()) == ({'unrelated_send'}, {'unrelated_receive'})
        assert env.worker.connector_worker._connector_worker_meta.completed_jobs == {10: 1}
        assert env.worker.connector_worker._connector_worker_meta.transfer_stats is stats
        assert list(meta.load_jobs) == [10] and not meta.store_jobs
        env.sel.cs._jobs.clear()  # Models unchanged native manager consuming completed_jobs.
    assert env.sel.cs.get_num_new_matched_tokens == original
    row, = env.probe.records
    assert all(row[k] for k in ('requested', 'committed', 'waited', 'filtered', 'request_state_preserved'))
    after = env.probe._state(env.req)
    assert before[:6] == after[:6] and before[7] == after[7] and before[9] == after[9]

    for known, hit, eligible in [(32,32,True), (49,32,True), (35,0,True), (35,32,False)]:
        env = make(known, hit, eligible)
        with env.probe:
            assert env.sel.cs.get_num_new_matched_tokens(env.req,0) == (hit,bool(hit))
        assert not env.probe.records[0]['requested']

    env = make()
    env.probe.enabled = False
    original = env.sel.cs.get_num_new_matched_tokens
    with env.probe:
        assert env.sel.cs.get_num_new_matched_tokens == original and not env.probe._saved

    env = make()
    env.sel.scheduler.needs_kv_cache_zeroing = True
    expect_error(env.probe.__enter__, AssertionError)
    assert not env.probe._saved

    env = make()
    env.add_store = True
    def bad_store():
        with env.probe:
            commit(env)
    expect_error(bad_store, AssertionError)
    assert not env.probe._saved

    for stage in ('start', 'finished'):
        env = make()
        error = RuntimeError('native sentinel')
        try:
            with env.probe:
                commit(env)
                if stage == 'finished':
                    env.worker.start_load_kv(None)
                env.native_error = error
                (env.worker.start_load_kv if stage == 'start' else env.worker.get_finished)(None)
        except RuntimeError as exc:
            assert exc is error
        else:
            raise AssertionError('native exception was swallowed')
        assert not env.probe._saved

    env = make()
    def missing_ack():
        with env.probe:
            commit(env)
            env.worker.start_load_kv(None)
    expect_error(missing_ack, AssertionError)
    assert not env.probe._saved
    print('PASS: CPU adapter ordering, fallback, native metadata, STORE guard, restoration and exception checks; GPU untested')


if __name__ == '__main__':
    main()

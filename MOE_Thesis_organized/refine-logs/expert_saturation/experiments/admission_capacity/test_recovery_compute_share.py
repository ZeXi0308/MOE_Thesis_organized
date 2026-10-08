"""Targeted callback contract checks; no transfer or GPU performance claims."""
from types import SimpleNamespace as NS
import unittest

from recovery_compute_share import install


def fixture():
    def request(rid, history, computed, output):
        return NS(request_id=rid, num_tokens=history, num_computed_tokens=computed,
                  num_output_tokens=output, max_tokens=2048, status=NS(name='RUNNING'))
    peer, target = request('peer', 17, 16, 1), request('target', 1024, 0, 20)
    group = type('FullAttentionManager', (), {})()
    group.req_to_blocks = {'peer': [None], 'target': [None]*64}
    s = NS(requests={'target': target, 'peer': peer}, running=[peer, target],
           max_num_scheduled_tokens=1024, active=True, deny_peer=False,
           scheduler_config=NS(async_scheduling=False, long_prefill_token_threshold=0),
           kv_cache_manager=NS(enable_caching=False,
               coordinator=NS(single_type_managers=[group]),
               block_pool=NS(get_num_free_blocks=lambda: 10)))
    def begin(preempted, timestamp):
        s._rotation_target = 'target' if s.active else None
    s._rotation_begin, s._rotation_hold = begin, lambda r: s.deny_peer and r is peer
    s._rotation_target = 'target'
    return s, peer, target


class HookContract(unittest.TestCase):
    def test_disabled_keeps_callbacks(self):
        s, _, _ = fixture()
        original = s._rotation_begin, s._rotation_hold
        data, remove = install(s)
        self.assertEqual((s._rotation_begin, s._rotation_hold), original)
        self.assertEqual(remove(), data)

    def test_base_release_clears_extra_hold(self):
        s, peer, _ = fixture()
        original = s._rotation_begin, s._rotation_hold
        data, remove = install(s, enabled=True)
        s._rotation_begin([], 0)
        self.assertTrue(s._rotation_hold(peer))
        self.assertEqual(data['planned_changes'], 1)
        # The base's observed-output/terminal transition owns this flag.
        # This checks propagation, not the base's actual output recognition.
        s.active = False
        s._rotation_begin([], 1)
        self.assertFalse(s._rotation_hold(peer))
        self.assertEqual(data['planned_changes'], 1)
        remove()
        self.assertEqual((s._rotation_begin, s._rotation_hold), original)

    def test_native_load_no_reorder_and_base_kv_denial(self):
        s, peer, target = fixture()
        data, remove = install(s, enabled=True)
        target.status.name = 'WAITING_FOR_REMOTE_KVS'
        s._rotation_begin([], 0)
        self.assertFalse(s._rotation_hold(peer))
        target.status.name = 'RUNNING'
        s.running.reverse()
        s._rotation_begin([], 1)
        self.assertFalse(s._rotation_hold(peer))
        self.assertIs(s.running[0], target)
        s.deny_peer = True
        self.assertTrue(s._rotation_hold(peer))
        self.assertEqual(data['planned_changes'], 0)
        remove()


if __name__ == '__main__':
    unittest.main()

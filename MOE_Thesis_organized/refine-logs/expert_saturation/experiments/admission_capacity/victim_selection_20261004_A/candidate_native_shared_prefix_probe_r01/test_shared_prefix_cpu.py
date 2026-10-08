"""Focused CPU fixtures for qualification and read-only native-release shadows."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


PKG = Path(__file__).resolve().parent / 'pkg'
TREE = ast.parse((PKG / 'staged_store_rotation.py').read_text())


def extracted(names):
    nodes = [node for node in TREE.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<release-fixture>', 'exec'), namespace)
    return namespace


HELPERS = extracted({'_native_release_state', '_max_release_shadow_choice',
                     '_equal_held_once_choice', '_native_victim_current_guard',
                     '_rank_native_victim'})


def fake_engine(prefix=False, coordinator_name=None):
    FullAttentionSpec = type('FullAttentionSpec', (), {})
    spec = FullAttentionSpec()
    spec.block_size = 16
    spec.sliding_window = spec.attention_chunk_size = None
    pool = NS(num_gpu_blocks=101, get_num_free_blocks=lambda: 100,
              null_block=NS(block_id=0, is_null=True))
    single = type('FullAttentionManager', (), {})()
    single.block_pool = pool
    single.block_size = 16
    name = coordinator_name or ('UnitaryKVCacheCoordinator' if prefix
                                else 'KVCacheCoordinatorNoPrefixCache')
    coordinator = type(name, (), {})()
    coordinator.single_type_managers = [single]
    coordinator.scheduler_block_size = 16
    layout = NS(kv_cache_groups=[NS(kv_cache_spec=spec, is_eagle_group=False,
                                    layer_names=['layer'])], num_blocks=101)
    manager = NS(enable_caching=prefix, block_pool=pool, coordinator=coordinator,
                 num_kv_cache_groups=1, watermark_blocks=0, use_eagle=False,
                 kv_cache_config=layout)
    scheduler = NS(kv_cache_manager=manager, kv_cache_config=layout,
                   num_lookahead_tokens=0, num_spec_tokens=0, dcp_world_size=1,
                   pcp_world_size=1, requests={}, get_request_counts=lambda: (0, 0))
    engine = NS(engine_core=NS(engine_core=NS(scheduler=scheduler)),
                vllm_config=NS(cache_config=NS(enable_prefix_caching=prefix),
                               speculative_config=None,
                               scheduler_config=NS(max_num_seqs=384),
                               model_config=NS(max_model_len=4096)),
                has_unfinished_requests=lambda: False)
    config = dict(prompt_tokens=128, output_tokens=1024, requests=320,
                  engine_max_num_seqs=384, cap=384)
    module = ModuleType('vllm.v1.kv_cache_interface')
    module.FullAttentionSpec = FullAttentionSpec
    return engine, config, module


class SharedPrefixFixtures(unittest.TestCase):
    def test_exact_coordinator_and_explicit_boolean(self):
        spec = importlib.util.spec_from_file_location('safe_static_fixture', PKG / 'safe_static.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def qualify(engine, config, interface):
            with patch.dict(sys.modules, {'vllm.v1.kv_cache_interface': interface}):
                return module.qualify_safe_cap(engine, config)

        engine, config, interface = fake_engine()
        self.assertEqual(qualify(engine, config, interface)['status'], 'QUALIFIED')
        engine, config, interface = fake_engine(True)
        self.assertEqual(qualify(engine, config, interface)['status'], 'QUALIFICATION_FAILED')
        config['enable_prefix_caching'] = True
        self.assertEqual(qualify(engine, config, interface)['status'], 'QUALIFIED')
        config['enable_prefix_caching'] = 1
        self.assertEqual(qualify(engine, config, interface)['status'], 'QUALIFICATION_FAILED')
        config['enable_prefix_caching'] = True
        engine.engine_core.engine_core.scheduler.kv_cache_manager.enable_caching = False
        self.assertEqual(qualify(engine, config, interface)['status'], 'QUALIFICATION_FAILED')
        for name in ('KVCacheCoordinatorNoPrefixCache', 'UnverifiedCoordinator'):
            engine, config, interface = fake_engine(True, name)
            config['enable_prefix_caching'] = True
            self.assertEqual(qualify(engine, config, interface)['status'], 'QUALIFICATION_FAILED')

    def test_physical_reference_counts_and_unknown(self):
        read = HELPERS['_native_release_state']
        blocks = [NS(block_id=i, ref_cnt=ref, is_null=False)
                  for i, ref in enumerate((1, 2, 3))]
        pool = NS(blocks=blocks)
        before = copy.deepcopy([vars(block) for block in blocks])
        self.assertEqual(read(blocks, pool), dict(immediate_releasable_blocks=1,
                         shared_blocks=2, release_state_error=None))
        self.assertEqual([vars(block) for block in blocks], before)
        for bad in ([blocks[0], blocks[0]], [NS(**vars(blocks[0]))],
                    [NS(block_id=-1, ref_cnt=1, is_null=False)]):
            result = read(bad, pool)
            self.assertIsNone(result['immediate_releasable_blocks'])
            self.assertIsNone(result['shared_blocks'])
            self.assertIsNotNone(result['release_state_error'])
        for ref in (0, -1, True, None):
            block = NS(block_id=0, ref_cnt=ref, is_null=False)
            self.assertIsNone(read([block], NS(blocks=[block]))['immediate_releasable_blocks'])
        null = NS(block_id=0, ref_cnt=1, is_null=True)
        self.assertEqual(read([null], NS(blocks=[null], null_block=null))['immediate_releasable_blocks'], 0)

    def test_shadow_uses_native_suffix_not_pure_decode_and_tail_still_executes(self):
        choice = HELPERS['_max_release_shadow_choice']
        rows = [dict(index=4, qualified=False, immediate_releasable_blocks=3),
                dict(index=5, qualified=True, immediate_releasable_blocks=1)]
        self.assertEqual(choice(rows, False), (4, None))
        rows[1]['immediate_releasable_blocks'] = 3
        self.assertEqual(choice(rows, False), (5, None))
        self.assertEqual(choice(rows, True), (5, 'ACTIVE_PROTECTION_OR_PHASE'))
        rows[0]['immediate_releasable_blocks'] = None
        self.assertEqual(choice(rows, False), (5, 'UNKNOWN_PHYSICAL_RELEASE'))

        install = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'install')
        pick = next(node for node in install.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'pick_victim')
        blocks = [NS(block_id=i, ref_cnt=ref, is_null=False)
                  for i, ref in enumerate((1, 1, 1, 1, 2, 2))]
        running = [NS(request_id=f'r{i}', arrival_time=float(i), num_preemptions=0,
                      num_output_tokens=0 if i == 1 else 1) for i in range(3)]
        owned = {r.request_id: blocks[2*i:2*i+2] for i, r in enumerate(running)}
        states = {r.request_id: NS(pure_decode=i != 1, status='RUNNING',
                  blocks=tuple(b.block_id for b in owned[r.request_id]),
                  computed=31, output=r.num_output_tokens, max_output=1024)
                  for i, r in enumerate(running)}
        data = dict(victim_decisions=[])
        namespace = dict(HELPERS, scheduler=NS(running=running),
            pool=NS(blocks=blocks, get_num_free_blocks=lambda: 0),
            owned=owned, victim_rule='tail', full_running_mode='off', current_guard='off',
            protected=None, phase=None, step=1, block_size=16, cs=None, time=time,
            view=lambda request: states[request.request_id],
            residence={r.request_id: (0, 0) for r in running}, data=data,
            bidkv_score=lambda request: None,
            host_prefix=lambda *args: {},
            _pending_native_store_dependencies=lambda *args: 0,
            equal_once_state=dict(mode='SHADOW', consumed=False, proposal_count=0,
                                  action_request_count=0))
        exec(compile(ast.Module(body=[pick], type_ignores=[]), '<installed-pick-victim>', 'exec'), namespace)
        self.assertEqual(namespace['pick_victim'](1, [running[0]]), 2)
        event = data['victim_decisions'][0]
        self.assertEqual([row['index'] for row in event['candidates']], [1, 2])
        self.assertTrue(event['fallback_unknown'])
        self.assertEqual(event['selected'], 'r2')
        self.assertEqual(event['max_release_shadow']['proposed_request'], 'r1')
        self.assertFalse(event['max_release_shadow']['action_requested'])
        self.assertTrue(event['max_release_shadow']['changed_from_tail'])
        self.assertEqual([block.ref_cnt for block in blocks], [1, 1, 1, 1, 2, 2])


if __name__ == '__main__':
    unittest.main()

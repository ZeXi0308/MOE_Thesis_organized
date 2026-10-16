"""CPU-only safetensors fixtures for the three requested tuner model families.

Run with: python -m unittest discover -s code/tests -p test_tune_moe.py
Needs torch, numpy and safetensors; never initializes CUDA or imports vLLM.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tune_moe
import torch
from safetensors.torch import save_file


def config_for(kind):
    config = dict(model_type=kind, hidden_size=4, intermediate_size=99,
                  num_hidden_layers=3, num_experts_per_tok=1, hidden_act='silu')
    if kind == 'olmoe':
        config.update(num_experts=2, intermediate_size=3)
    elif kind == 'qwen3_moe':
        config.update(num_experts=2, moe_intermediate_size=3,
                      decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True)
    else:
        config.update(n_routed_experts=2, moe_intermediate_size=3,
                      first_k_dense_replace=1, moe_layer_freq=1, n_shared_experts=2,
                      routed_scaling_factor=1.0, norm_topk_prob=False)
    return config


def checkpoint(path, kind):
    config = config_for(kind)
    layer = 1 if kind == 'deepseek_v2' else 0
    (path / 'config.json').write_text(json.dumps(config))
    tensors, mapping = [{}, {}], {}
    expected = {}
    for expert in range(2):
        for index, projection in enumerate(('gate_proj', 'up_proj', 'down_proj')):
            shape = (4, 3) if projection == 'down_proj' else (3, 4)
            # Distinct values expose expert ordering, gate/up reversal and transpose.
            value = (torch.arange(12).reshape(shape) + expert * 40 + index * 12).bfloat16()
            key = f'model.layers.{layer}.mlp.experts.{expert}.{projection}.weight'
            shard = (expert + index) % 2
            tensors[shard][key] = value
            mapping[key] = f'shard-{shard}.safetensors'
            expected[(expert, projection)] = value
    # This nonexistent shard must not be opened by routed-only loading.
    mapping[f'model.layers.{layer}.mlp.shared_experts.gate_proj.weight'] = 'shared-not-needed.safetensors'
    mapping['model.layers.0.mlp.gate_proj.weight'] = 'dense-not-needed.safetensors'
    for index, data in enumerate(tensors):
        save_file(data, str(path / f'shard-{index}.safetensors'))
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    return config, layer, expected


class TunerWeightsTest(unittest.TestCase):
    def test_all_three_layouts_use_routed_intermediate_and_shards_only(self):
        for kind in ('olmoe', 'qwen3_moe', 'deepseek_v2'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                config, layer, expected = checkpoint(path, kind)
                self.assertEqual(tune_moe.resolve_layer(config), layer)
                w1, w2, meta = tune_moe.load_layer_weights(path, config, None)
                self.assertEqual(tuple(w1.shape), (2, 6, 4))
                self.assertEqual(tuple(w2.shape), (2, 4, 3))
                self.assertEqual(w1.dtype, torch.bfloat16)
                for expert in range(2):
                    self.assertTrue(torch.equal(w1[expert, :3], expected[expert, 'gate_proj']))
                    self.assertTrue(torch.equal(w1[expert, 3:], expected[expert, 'up_proj']))
                    self.assertTrue(torch.equal(w2[expert], expected[expert, 'down_proj']))
                self.assertEqual(meta['source_files'], ['shard-0.safetensors', 'shard-1.safetensors'])
                self.assertEqual(meta['expert_weight_bytes'], 72)
                self.assertEqual(len(meta['index_sha256']), 64)
                self.assertEqual(meta['geometry']['shared_experts_excluded'], 2 if kind == 'deepseek_v2' else 0)

    def test_dense_layer_quantized_checkpoint_and_wrong_shape_are_rejected(self):
        config = config_for('deepseek_v2')
        with self.assertRaisesRegex(ValueError, 'not a routed MoE'):
            tune_moe.resolve_layer(config, 0)
        with self.assertRaises(ValueError):
            tune_moe.model_geometry({**config, 'quantization_config': {'quant_method': 'fp8'}})
        with self.assertRaisesRegex(ValueError, 'unit routed_scaling_factor'):
            tune_moe.model_geometry({**config, 'routed_scaling_factor': 2.5})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config, layer, _ = checkpoint(path, 'qwen3_moe')
            with self.assertRaisesRegex(ValueError, '!='):
                tune_moe.load_layer_weights(path, {**config, 'moe_intermediate_size': 4}, layer)
            (path / 'shard-0.safetensors').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'not complete'):
                tune_moe.load_layer_weights(path, config, layer)

    def test_plan_selects_deepseek_layer1_without_torch_and_preserves_native_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config, layer, _ = checkpoint(path, 'deepseek_v2')
            row = dict(layer=f'model.layers.{layer}.mlp.experts', phase='decode',
                       source='vllm_actual_select_experts', forward_completed=True,
                       topk_ids=[[1]], topk_weights=[[0.4]], num_experts=2, step=2)
            trace = path / 'trace.json'
            trace.write_text(json.dumps([{'spec': None, 'trace': [row]},
                                         {'spec': {'mode': 'global_native'}, 'trace': [row]}]))
            for quantization in ('bf16', 'fp8'):
                with self.subTest(quantization=quantization):
                    output = path / f'plan-{quantization}'
                    # -S excludes site packages, including torch/vLLM. The default
                    # branch also checks that old BF16 commands stay unchanged.
                    arguments = [] if quantization == 'bf16' else ['--quantization', 'fp8']
                    result = subprocess.run([sys.executable, '-S', str(Path(tune_moe.__file__)),
                                             '--model', str(path), '--traces', str(trace), '--output-dir', str(output),
                                             '--batches', '1', '--plan-only', *arguments], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads((output / 'plan.json').read_text())
                    self.assertEqual(plan['layer'], 1)
                    self.assertEqual(plan['shape']['intermediate'], 3)
                    self.assertEqual(plan['selected_routes']['1'][0]['topk_weights'], [[0.4]])
                    self.assertEqual(plan['route_provenance']['rejected']['not_explicit_native_or_dense'], 1)
                    self.assertEqual(len(plan['grid']), 8)
                    self.assertEqual(plan['quantization'], quantization)
                    self.assertEqual(plan['config_dtype'], 'fp8_w8a8' if quantization == 'fp8' else None)
                    self.assertEqual(plan['status'], 'CPU_PLAN')


if __name__ == '__main__':
    unittest.main()

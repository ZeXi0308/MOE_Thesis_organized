#!/usr/bin/env python3
"""Calibrate expert-only NVFP4 from the existing AWQ-smoothed BF16 master.

Default is a CPU-only plan. --execute requires the live run_locked ancestor,
the completed AWQ/common-master pair, and the pinned calibration environment.
No AWQ modifier is run and no second BF16 master is written. Official NVFP4
W4A4 calibration observes activation global scales on the fixed text dataset;
this is not a data-free quantization path. Quality remains to be measured.

The complete saved checkpoint retains byte-identical nonexpert tensors. The
same NVFP4 artifact exports are used for quality and performance while the
worker loads the original common-BF16 host. Existing CT exporter lineage fields
retain the original base; explicit immediate_source fields identify the actual
derived BF16 input without pretending it is the original checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import time

from calibration_common_basis import (
    check_compressed_expert_layout, model_basis, raw_tensor_hash, require_disk_space,
    saved_headers, saved_tensor_reader, source_layout, verify_saved_basis,
)
from export_bf16_fp8_artifact import canonical_json, resolve_child
from export_common_basis_sidecars import load_pair
from prepare_quantization import calibration_environment, read_json, register, sha256, validate_config, write_json


EXPERT_MODULE = r'^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$'
EXPERT_TARGET = 're:' + EXPERT_MODULE
EXPERT_WEIGHT = re.compile(EXPERT_MODULE[:-1] + r'\.weight$')


def config_inputs(config, root):
    required = {'source_master_manifest', 'source_awq_manifest', 'calibration_jsonl',
                'calibration_jsonl_sha256', 'output', 'base_checkpoint', 'base_revision',
                'samples', 'max_seq_length', 'seed', 'reserve_gib', 'run_name', 'timeout_s',
                'calibration_python', 'moe_calibrate_all_experts'}
    if not required <= config.keys():
        raise ValueError('Missing fixed calibration configuration fields')
    if (config['samples'] != 512 or config['max_seq_length'] != 512 or config['seed'] != 42 or
            config['moe_calibrate_all_experts'] is not False or config['reserve_gib'] < 0):
        raise ValueError('This prepared run requires 512x512, seed 42 and natural MoE routing')
    paths = {key: (root / config[key]).resolve() for key in
             ('source_master_manifest', 'source_awq_manifest', 'calibration_jsonl', 'output')}
    if sha256(paths['calibration_jsonl']) != config['calibration_jsonl_sha256']:
        raise ValueError('Calibration JSONL differs from the fixed dataset')
    return paths


def source_pair(paths, config):
    master, awq = load_pair(paths['source_master_manifest'], paths['source_awq_manifest'])
    if (master['model_key'] != 'olmoe' or master['base_checkpoint'] != config['base_checkpoint'] or
            master['base_checkpoint_revision'] != config['base_revision']):
        raise ValueError('Common-master pair differs from the pinned OLMoE source')
    source = Path(master['checkpoint_path']).resolve()
    output = paths['output']
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a, b in
           ((source, output), (Path(awq['checkpoint_path']).resolve(), output))):
        raise ValueError('NVFP4 output must be separate from both existing checkpoints')
    source_config = read_json(source / 'config.json')
    validate_config(source_config, 'olmoe', 'bf16')
    if sha256(source / 'config.json') != master['identity']['config_sha256']:
        raise ValueError('Common-master config SHA256 differs from its manifest')
    return master, awq, source, source_config


def source_tensor_specs(source, master):
    headers, mapping = source_layout(source, master)
    specs = {key: headers[filename][key] for key, filename in mapping.items()}
    experts = {key: spec for key, spec in specs.items() if EXPERT_WEIGHT.fullmatch(key)}
    expected = master['shape']['experts'] * master['shape']['layers'] * 3
    if len(experts) != expected or any(spec['dtype'] != 'BF16' for spec in experts.values()):
        raise ValueError('Master does not contain every expected BF16 expert projection')
    nonexperts = {key: spec for key, spec in specs.items() if key not in experts}
    # Payloads: packed FP4 plus one FP8 block scale per 16 weights and two globals.
    nvfp4_bytes = sum((spec['data_offsets'][1] - spec['data_offsets'][0]) // 2 * 9 // 16 + 8
                     for spec in experts.values())
    nonexpert_bytes = sum(spec['data_offsets'][1] - spec['data_offsets'][0] for spec in nonexperts.values())
    return specs, mapping, nonexperts, nvfp4_bytes + nonexpert_bytes + (64 << 20)


def file_nonexpert_hashes(directory, specs):
    headers = saved_headers(directory)
    observed = {key for key in headers if not re.match(EXPERT_MODULE[:-1] + r'\.', key)}
    if observed != set(specs):
        raise ValueError('Saved nonexpert tensor names changed or acquired quantization parameters')
    result = {}
    with saved_tensor_reader(directory, headers) as read:
        for key, spec in specs.items():
            actual = headers[key][1]
            if actual['shape'] != spec['shape'] or actual['dtype'] != spec['dtype']:
                raise ValueError(f'Nonexpert shape/dtype changed: {key}')
            result[key] = {'dtype': spec['dtype'], 'shape': spec['shape'], 'sha256': raw_tensor_hash(read(key))}
    return result


def verify_model_nonexperts(model, expected):
    import torch
    dtypes = {torch.bfloat16: 'BF16', torch.float16: 'F16', torch.float32: 'F32',
              torch.int64: 'I64', torch.int32: 'I32', torch.bool: 'BOOL'}
    for key, record in expected.items():
        parent, attr = key.rsplit('.', 1)
        tensor = getattr(model.get_submodule(parent), attr)
        if (dtypes.get(tensor.dtype) != record['dtype'] or list(tensor.shape) != record['shape'] or
                raw_tensor_hash(tensor) != record['sha256']):
            raise ValueError(f'Loaded/calibrated nonexpert tensor differs from common master: {key}')


def verify_frozen_scales(model, expected_names):
    import torch
    observed = set()
    for name, module in model.named_modules():
        scheme = getattr(module, 'quantization_scheme', None)
        if scheme is None:
            continue
        if name not in expected_names:
            raise ValueError(f'Unexpected nonexpert quantization: {name}')
        observed.add(name)
        if 'frozen' not in str(getattr(module, 'quantization_status', '')).lower():
            raise ValueError(f'Expert quantization was not frozen: {name}')
        for key in ('weight_global_scale', 'input_global_scale'):
            value = getattr(module, key).detach().cpu().float()
            if value.numel() != 1 or not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError(f'Missing/nonfinite calibrated global scale: {name}.{key}')
    if observed != expected_names:
        raise ValueError('Official modifier did not quantize exactly the expected experts')
    return {'quantized_expert_projections': len(observed), 'nonexpert_modules_quantized': 0,
            'positive_finite_weight_and_input_global_scales': True}


def execute(config, paths, root, lock_status):
    from bench_serving import verify_lock
    lock = verify_lock(lock_status)
    environment = calibration_environment()
    master, awq, source, source_config = source_pair(paths, config)
    for record in master['identity']['weight_files']:
        path = resolve_child(source, record['path'])
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Common-master payload differs from its exact content identity')
    _, _, nonexpert_specs, estimated_bytes = source_tensor_specs(source, master)
    output = paths['output']
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve previous output/partial data; choose a new empty NVFP4 directory')
    output.mkdir(parents=True, exist_ok=True)
    disk = require_disk_space([(output, estimated_bytes)], config['reserve_gib'])
    common = {'checkpoint_path': str(source), 'checkpoint': master['identity']['repo_id'],
              'checkpoint_fingerprint': master['checkpoint_fingerprint'],
              'checkpoint_revision': master['identity']['revision'],
              'manifest_sha256': sha256(paths['source_master_manifest'])}
    plan = {'configuration': config, 'common_bf16_master': common,
            'awq_manifest_sha256': sha256(paths['source_awq_manifest']),
            'disk_plan': disk, 'lock': lock, 'environment': environment,
            'expert_target': EXPERT_TARGET, 'scheme': 'NVFP4', 'awq_modifier_in_this_stage': False,
            'new_bf16_master_written': False, 'estimated_output_bytes': estimated_bytes}
    write_json(output / 'calibration_plan.json', plan)
    shutil.copyfile(paths['source_master_manifest'], output / 'source_common_bf16_manifest.json')
    shutil.copyfile(paths['source_awq_manifest'], output / 'source_awq_manifest.json')
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.utils import load_context
    from llmcompressor.modifiers.quantization import QuantizationModifier
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    rows = [json.loads(line) for line in paths['calibration_jsonl'].read_text().splitlines() if line.strip()]
    if len(rows) < config['samples'] or any(not isinstance(row.get('text'), str) for row in rows):
        raise ValueError('Fixed calibration JSONL lacks the required text records')
    data = Dataset.from_list(rows).shuffle(seed=config['seed']).select(range(config['samples']))
    data = data.map(lambda row: tokenizer(row['text'], truncation=True,
        max_length=config['max_seq_length'], add_special_tokens=False), remove_columns=data.column_names)
    expected_nonexperts = file_nonexpert_hashes(source, nonexpert_specs)
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
                    dtype=torch.bfloat16, device_map='auto')
    verify_model_nonexperts(model, expected_nonexperts)
    basis = model_basis(model, source_config)
    if basis != master['activation_bases']:
        raise ValueError('Actual loaded norm/router basis differs from the common master')
    modules = {name: module for name, module in model.named_modules() if re.fullmatch(EXPERT_MODULE, name)}
    expected_count = master['shape']['experts'] * master['shape']['layers'] * 3
    if len(modules) != expected_count or any(not isinstance(module, torch.nn.Linear) for module in modules.values()):
        raise ValueError('Expected every linearized OLMoE expert projection; no fallback')
    recipe = QuantizationModifier(targets=[EXPERT_TARGET], scheme='NVFP4')
    resolved = recipe.resolved_config.model_dump(mode='json')
    write_json(output / 'resolved_quantization_recipe.json', resolved)
    counts, hooks = {name: 0 for name in modules}, []
    for name, module in modules.items():
        def record(module, inputs, name=name):
            if inputs and inputs[0].numel():
                counts[name] += inputs[0].numel() // inputs[0].shape[-1]
        hooks.append(module.register_forward_pre_hook(record))
    started = time.monotonic()
    try:
        oneshot(model=model, dataset=data, recipe=recipe,
                num_calibration_samples=config['samples'], max_seq_length=config['max_seq_length'],
                moe_calibrate_all_experts=False)
    finally:
        for hook in hooks:
            hook.remove()
        write_json(output / 'calibration_coverage.json', {'projection_token_exposures': counts,
            'meaning': 'natural-route forward exposures including official sequential replay; not a route trace',
            'moe_calibrate_all_experts': False, 'elapsed_s': time.monotonic() - started})
    if any(count <= 0 for count in counts.values()):
        raise RuntimeError('A target expert projection received no calibration input; output is not accepted')
    scale_check = verify_frozen_scales(model, set(modules))
    verify_model_nonexperts(model, expected_nonexperts)
    if model_basis(model, source_config) != basis:
        raise ValueError('Norm/router basis changed during NVFP4 calibration')
    write_json(output / 'activation_bases.json', basis)
    nonexpert_check = {'tensors': expected_nonexperts,
        'identity_sha256': hashlib.sha256(canonical_json(expected_nonexperts).encode()).hexdigest(),
        'loaded_before_calibration_match': True, 'after_calibration_match': True,
        'saved_checkpoint_match': False}
    write_json(output / 'nonexpert_identity.json', nonexpert_check)
    require_disk_space([(output, estimated_bytes)], config['reserve_gib'])
    model.save_pretrained(output, save_compressed=True, max_shard_size='5GB')
    tokenizer.save_pretrained(output)
    if file_nonexpert_hashes(output, nonexpert_specs) != expected_nonexperts:
        raise ValueError('Compressed save changed nonexpert parameter bytes')
    nonexpert_check['saved_checkpoint_match'] = True
    write_json(output / 'nonexpert_identity.json', nonexpert_check)
    saved_basis = verify_saved_basis(output, basis)
    saved_layout = check_compressed_expert_layout(output, master['shape'], 'nvfp4')
    from export_ct_nvfp4_artifact import _check_quantization
    _check_quantization(read_json(output / 'config.json'))
    provenance = {'method': 'NVFP4 W4A4 activation calibration from a previously AWQ-smoothed BF16 master',
        'evidence': {'common_bf16_master': common, 'source_checkpoint': master['identity']['repo_id'],
            'source_revision': master['identity']['revision'], 'source_checkpoint_fingerprint': master['checkpoint_fingerprint'],
            'original_awq_manifest_sha256': sha256(paths['source_awq_manifest']),
            'original_awq_calibration': awq['calibration'],
            'awq_modifier_in_this_stage': False, 'calibration_jsonl_sha256': config['calibration_jsonl_sha256'],
            'samples': config['samples'], 'max_seq_length': config['max_seq_length'], 'seed': config['seed'],
            'moe_calibrate_all_experts': False, 'projection_token_exposures': counts,
            'environment': environment, 'resolved_quantization_recipe': resolved,
            'quantization_scope': {'targets': [EXPERT_TARGET], 'includes_attention': False,
                'nonexpert_parameters': 'all original names, shapes, dtypes and raw bytes verified unchanged'},
            'nonexpert_identity_sha256': nonexpert_check['identity_sha256'],
            'frozen_scale_check': scale_check, 'saved_basis_check': saved_basis,
            'saved_expert_layout_check': saved_layout, 'quality_validation': 'not_run'}}
    # Keep the existing exporter's original-base lineage convention. Immediate
    # source fields and common_bf16_master explicitly identify this stage's input.
    derivation = {'method': 'awq_nvfp4_calibrated_compression',
        'source_checkpoint': master['base_checkpoint'], 'source_revision': master['base_checkpoint_revision'],
        'source_checkpoint_role': 'original base lineage; not the immediate input to this stage',
        'source_checkpoint_fingerprint': master['derivation']['source_checkpoint_fingerprint'],
        'immediate_source_checkpoint': master['identity']['repo_id'],
        'immediate_source_revision': master['identity']['revision'],
        'immediate_source_checkpoint_fingerprint': master['checkpoint_fingerprint'],
        'common_bf16_master': common,
        'stages': ['AWQ smoothing inherited from the verified existing master',
                   'expert-only official NVFP4 activation calibration; no additional smoothing'],
        'new_bf16_master_written': False}
    return register(output, 'olmoe', 'nvfp4', master['base_checkpoint'], output / 'checkpoint_manifest.json',
        calibration=provenance, local_derivation=derivation,
        base_identity={'repo_id': master['base_checkpoint'], 'revision': master['base_checkpoint_revision']},
        activation_bases=basis)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=root / 'configs/olmoe_nvfp4_common_r01.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--lock-status', type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    paths = config_inputs(config, root)
    if not args.execute:
        ready = paths['source_master_manifest'].is_file() and paths['source_awq_manifest'].is_file()
        inputs = {'common_pair_manifests_present': ready}
        if ready:
            master, _, source, _ = source_pair(paths, config)
            _, _, _, estimate = source_tensor_specs(source, master)
            inputs.update(common_master_content_revision=master['identity']['revision'],
                          source_manifest_sha256=sha256(paths['source_master_manifest']), estimated_output_bytes=estimate)
        command = [str(root / config['calibration_python']), str(Path(__file__).resolve()),
                   '--config', str(args.config.resolve()), '--execute', '--lock-status',
                   str(root / 'runs' / config['run_name'] / 'status.json')]
        print(json.dumps({'state': 'CPU_PLAN_ONLY', 'inputs': inputs, 'configuration': config,
            'expert_target': EXPERT_TARGET, 'gpu_work_started': False,
            'command_inside_run_locked': shlex.join(command),
            'required_run_locked_name': config['run_name'], 'required_timeout_s': config['timeout_s']}, indent=2))
        return
    if not args.lock_status:
        parser.error('--execute requires --lock-status from the live run_locked ancestor')
    result = execute(config, paths, root, args.lock_status)
    print(json.dumps({'state': 'SAVED_CALIBRATED_CHECKPOINT', 'checkpoint_path': result['checkpoint_path'],
        'checkpoint_fingerprint': result['checkpoint_fingerprint'], 'format': result['format'],
        'quality_validated': False, 'runtime_export_and_worker_validation': 'not_run'}, indent=2))


if __name__ == '__main__':
    main()

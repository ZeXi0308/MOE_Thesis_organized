#!/usr/bin/env python3
"""Export all 16 real FP8/AWQ layers and construct the three integration arms.

Default: print the plan without importing CUDA or requiring completed inputs.
--execute-export: run the existing native exporters inside run_locked, then
write configs. --write: construct configs from an already completed export.
The same common BF16 host is used by every arm. Small-sample AWQ and threshold
32 are integration choices, not a quality result or a tuned scheduling policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from export_common_basis_sidecars import build_jobs, check_artifact, load_pair, read_json
from export_bf16_fp8_artifact import canonical_json, sha256


def export_command(args):
    command = [args.python, str(ROOT / 'code/export_common_basis_sidecars.py'),
        '--bf16-manifest', str(args.bf16_manifest), '--awq-manifest', str(args.awq_manifest),
        '--out-dir', str(args.out_dir), '--python', args.python, '--device', args.device,
        '--validate-tokens', '0', '--reserve-gib', str(args.reserve_gib)]
    if args.resume:
        command.append('--resume')
    return command


def completed_configs(args):
    bf16, awq = load_pair(args.bf16_manifest, args.awq_manifest)
    evidence = awq['calibration']['evidence']
    if (bf16['model_key'] != 'olmoe' or bf16['shape']['layers'] != 16 or
            evidence.get('samples') != 64 or evidence.get('max_seq_length') != 128 or
            evidence.get('seed') != 42):
        raise ValueError('Expected the complete OLMoE 16-layer, 64x128 seed42 integration pair')
    manifest_path = args.out_dir / 'sidecar_manifest.json'
    exported = read_json(manifest_path)
    if exported.get('state') != 'complete':
        raise ValueError('All 32 real layer artifacts must finish before worker configs are written')
    for key, path in (('bf16', args.bf16_manifest), ('awq', args.awq_manifest)):
        if exported['source_manifests'][key]['sha256'] != sha256(path):
            raise ValueError('Completed export came from a different common-basis pair')
    records = {(row['layer'], row['format']): row for row in exported['artifacts']}
    if len(records) != 32 or len(exported['artifacts']) != 32:
        raise ValueError('Completed export must contain exactly 16 FP8 and 16 AWQ artifacts')
    args.validate_tokens = 0
    jobs = build_jobs(args, bf16, awq)
    artifacts = {}
    for job in jobs:
        record = records[(job['layer'], job['format'])]
        if Path(record['artifact']).resolve() != Path(job['artifact']).resolve():
            raise ValueError('Completed artifact path differs from the requested export directory')
        checked = check_artifact(job, bf16 if job['format'] == 'fp8' else awq, hash_payload=False)
        if checked is None or any(checked[key] != record[key] for key in
                ('artifact_sha256', 'metadata_sha256', 'tensor_bytes')):
            raise ValueError('Artifact metadata changed after the native export completed')
        artifacts[(job['layer'], job['format'])] = job['artifact']
    common = {'schema_version': 1, 'mechanism_only': False, 'allow_unverified_source': False,
              'release_original_weights': True, 'record_limit': 1024,
              'record_cuda_events': False, 'validate_route_values': False}
    configs = {}
    for arm in ('single_fp8', 'single_awq', 'dual_phase'):
        config = {**common, 'report_path': str(args.report_dir / arm / 'worker_installation.json')}
        if arm == 'dual_phase':
            config.update(policy='phase', threshold=32, lowbit_format='awq')
        else:
            config.update(policy='single', single_format=arm.removeprefix('single_'))
        layers = []
        for layer in range(16):
            row = {'layer': layer, 'module': f'model.layers.{layer}.mlp.experts',
                'worker_activation_basis_id': bf16['activation_bases'][str(layer)]['activation_basis_id']}
            if arm == 'dual_phase':
                row.update(fp8_weights=artifacts[(layer, 'fp8')], lowbit_weights=artifacts[(layer, 'awq')])
            else:
                row['weights'] = artifacts[(layer, config['single_format'])]
            layers.append(row)
        config['layers'] = layers
        configs[arm] = config
    metadata = {'schema_version': 1, 'state': 'prepared_for_full_model_integration',
        'host_manifest': str(args.bf16_manifest), 'host_checkpoint': bf16['checkpoint_path'],
        'host_manifest_sha256': sha256(args.bf16_manifest),
        'host_checkpoint_fingerprint': bf16['checkpoint_fingerprint'],
        'host_runtime_quantization': bf16.get('runtime_quantization'),
        'awq_manifest': str(args.awq_manifest), 'awq_manifest_sha256': sha256(args.awq_manifest),
        'sidecar_manifest': str(manifest_path), 'sidecar_manifest_sha256': sha256(manifest_path),
        'calibration': {'method': 'real AWQ', 'samples': 64, 'max_seq_length': 128, 'seed': 42},
        'threshold': {'value': 32, 'scope': 'mixed scheduler steps only; integration value, not tuned'},
        'phase_rule': 'pure prefill uses FP8; pure decode uses AWQ; mixed uses threshold32',
        'required_runtime_evidence': 'complete forward, natural short generation and minimum serving; dual must execute both formats in all16 layers',
        'comparison': 'all arms use the identical BF16 master, including nonexpert parameters',
        'quality_validated': False, 'gpu_integration_validated': False,
        'artifact_hash_verification': 'payloads verified by completed exporter; consumers hash again before actual model launch',
        'configs': {arm: {'path': str(args.config_dir / (arm + '.json'))} for arm in configs}}
    return configs, metadata


def save_configs(args):
    configs, metadata = completed_configs(args)
    outputs = {args.config_dir / (arm + '.json'): value for arm, value in configs.items()}
    for path, value in outputs.items():
        if path.exists() and canonical_json(read_json(path)) != canonical_json(value):
            raise FileExistsError(f'Preserve existing configuration; choose a new config directory: {path}')
    args.config_dir.mkdir(parents=True, exist_ok=True)
    for path, value in outputs.items():
        path.write_text(json.dumps(value, indent=2) + '\n')
        metadata['configs'][path.stem]['sha256'] = sha256(path)
    path = args.config_dir / 'integration_manifest.json'
    if path.exists() and canonical_json(read_json(path)) != canonical_json(metadata):
        raise FileExistsError('Existing integration manifest differs; preserve it and use a new config directory')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bf16-manifest', type=Path, default=ROOT / 'checkpoints/olmoe_awq_integration_bf16_r01/checkpoint_manifest.json')
    p.add_argument('--awq-manifest', type=Path, default=ROOT / 'checkpoints/olmoe_awq_integration_r01/checkpoint_manifest.json')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'artifacts/olmoe_awq_integration_r01')
    p.add_argument('--config-dir', type=Path, default=ROOT / 'configs/olmoe_integration_r01')
    p.add_argument('--report-dir', type=Path, default=ROOT / 'runs/olmoe_full_model_integration_r01')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--reserve-gib', type=float, default=20)
    p.add_argument('--resume', action='store_true')
    action = p.add_mutually_exclusive_group()
    action.add_argument('--execute-export', action='store_true')
    action.add_argument('--write', action='store_true', help='CPU-only config construction after all native exports completed')
    p.add_argument('--lock-status', type=Path)
    args = p.parse_args()
    if args.reserve_gib < 0:
        p.error('reserve-gib must be nonnegative')
    for name in ('bf16_manifest', 'awq_manifest', 'out_dir', 'config_dir', 'report_dir'):
        setattr(args, name, getattr(args, name).resolve())
    command = export_command(args)
    if not args.execute_export and not args.write:
        print(json.dumps({'state': 'CPU_PLAN_ONLY', 'gpu_work_started': False,
            'inputs_ready': args.bf16_manifest.is_file() and args.awq_manifest.is_file(),
            'export_command_inside_run_locked': shlex.join(command + ['--execute', '--lock-status', 'runs/olmoe_integration_export_r01/status.json']),
            'configuration_builder_after_export': shlex.join([args.python, str(Path(__file__).resolve()), '--write']),
            'worker_configs': [str(args.config_dir / (arm + '.json')) for arm in ('single_fp8', 'single_awq', 'dual_phase')],
            'host_manifest': str(args.bf16_manifest), 'layers': 16, 'artifacts': 32,
            'additional_kernel_diagnostics': False, 'quality_validated': False}, indent=2))
        return
    if args.execute_export:
        if args.lock_status is None:
            p.error('--execute-export requires --lock-status from run_locked')
        subprocess.run(command + ['--execute', '--lock-status', str(args.lock_status.resolve())], check=True)
    print(json.dumps(save_configs(args), indent=2))


if __name__ == '__main__':
    main()

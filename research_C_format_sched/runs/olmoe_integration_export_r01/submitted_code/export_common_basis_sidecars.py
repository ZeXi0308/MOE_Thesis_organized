#!/usr/bin/env python3
"""Export every layer of one calibrated AWQ/common-BF16 pair for the worker.

Default is a CPU-only plan; manifests must already exist. --execute requires
a live run_locked ancestor and the base vLLM 0.26 / CT 0.17 environment. Each
child exporter hashes its actual source shards and performs native GPU layout
conversion. CPU hash I/O and conversion both count toward the wrapper budget.

  python code/export_common_basis_sidecars.py \
    --bf16-manifest checkpoints/olmoe_awq_common_bf16_r01/checkpoint_manifest.json \
    --awq-manifest checkpoints/olmoe_awq_common_r01/checkpoint_manifest.json \
    --out-dir artifacts/olmoe_awq_common_r01

To execute, wrap that command in run_locked and add --execute --lock-status
runs/<job>/status.json. --resume accepts only complete, hash-verified artifacts;
partial outputs are preserved and rejected. Optional --worker-config also
requires --policy, --threshold and --worker-report-path; it does not invent a
tuned threshold. Export success is not numerical/quality validation of FP8 or
the complete model. The AWQ child optionally runs its small numerical check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from calibration_common_basis import content_revision
from export_bf16_fp8_artifact import canonical_json, sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_pair(bf16_path, awq_path):
    bf16, awq = read_json(bf16_path), read_json(awq_path)
    for manifest, fmt, method in ((bf16, 'bf16', 'awq_smoothed_bf16_master'),
                                  (awq, 'awq', 'awq_calibrated_compression')):
        if (manifest.get('checkpoint_kind') != 'local_derived' or manifest.get('format') != fmt or
                manifest.get('derivation', {}).get('method') != method or
                manifest['identity'].get('revision') != content_revision(manifest['identity']) or
                manifest.get('exact_base_revision_known') is not True):
            raise ValueError(f'Expected an explicit, content-identified {fmt} common-basis manifest')
    for key in ('model_key', 'shape', 'base_checkpoint', 'base_checkpoint_revision'):
        if bf16[key] != awq[key]:
            raise ValueError(f'Common-basis pair disagrees on {key}')
    master_link = awq['calibration']['evidence'].get('common_bf16_master', {})
    if (master_link.get('checkpoint_fingerprint') != bf16['checkpoint_fingerprint'] or
            master_link.get('checkpoint_revision') != bf16['identity']['revision'] or
            master_link.get('manifest_sha256') != sha256(bf16_path)):
        raise ValueError('AWQ calibration provenance does not identify this exact BF16 master manifest')
    layers = bf16['shape']['layers']
    expected_layers = {str(layer) for layer in range(layers)}
    if set(bf16.get('activation_bases', {})) != expected_layers or set(awq.get('activation_bases', {})) != expected_layers:
        raise ValueError('Both manifests must describe every actual model layer')
    for layer in expected_layers:
        for key in ('activation_basis_id', 'activation_basis_descriptor', 'router_raw_bytes_sha256'):
            if bf16['activation_bases'][layer][key] != awq['activation_bases'][layer][key]:
                raise ValueError(f'Layer {layer} common norm/router contract differs')
    return bf16, awq


def artifact_bytes(shape, fmt):
    params = 3 * shape['experts'] * shape['d'] * shape['intermediate']
    return (params + 8 * shape['experts']) if fmt == 'fp8' else (params // 2 + params // 128 * 2)


def build_jobs(args, bf16, awq):
    code = Path(__file__).resolve().parent
    jobs = []
    for layer in range(bf16['shape']['layers']):
        directory = args.out_dir / f'layer{layer:02d}'
        for fmt, manifest, source in (('fp8', bf16, args.bf16_manifest), ('awq', awq, args.awq_manifest)):
            artifact = directory / f'layer{layer}_{fmt}_{"triton" if fmt == "fp8" else "marlin"}.pt'
            if fmt == 'fp8':
                command = [args.python, str(code / 'export_bf16_fp8_artifact.py'),
                    '--manifest', str(source), '--layer', str(layer), '--formats', 'fp8',
                    '--device', args.device, '--out-dir', str(directory)]
            else:
                command = [args.python, str(code / 'export_awq_artifact.py'),
                    '--manifest', str(source), '--layer', str(layer), '--device', args.device,
                    '--validate-tokens', str(args.validate_tokens), '--out', str(artifact)]
            jobs.append({'layer': layer, 'format': fmt, 'artifact': str(artifact), 'command': command,
                'source_checkpoint_fingerprint': manifest['checkpoint_fingerprint'],
                'checkpoint': manifest['identity']['repo_id'], 'checkpoint_revision': manifest['identity']['revision'],
                'activation_basis_id': manifest['activation_bases'][str(layer)]['activation_basis_id'],
                'estimated_tensor_bytes': artifact_bytes(manifest['shape'], fmt)})
    return jobs


def check_artifact(job, manifest, *, hash_payload):
    artifact = Path(job['artifact'])
    sidecar = artifact.with_suffix(artifact.suffix + '.json')
    if not artifact.exists() and not sidecar.exists():
        return None
    if not artifact.is_file() or not sidecar.is_file():
        raise ValueError(f'Partial artifact preserved; use a fresh output directory: {artifact}')
    metadata = read_json(sidecar)
    expected = {key: job[key] for key in ('format', 'layer', 'checkpoint', 'checkpoint_revision',
                                        'source_checkpoint_fingerprint', 'activation_basis_id')}
    expected.update(shape={k: manifest['shape'][k] for k in ('d', 'intermediate', 'experts', 'top_k')},
        base_checkpoint=manifest['base_checkpoint'], base_checkpoint_revision=manifest['base_checkpoint_revision'],
        activation_basis_verified=True,
        activation_basis_descriptor=manifest['activation_bases'][str(job['layer'])]['activation_basis_descriptor'],
        layout='vllm_triton' if job['format'] == 'fp8' else 'vllm_marlin_processed')
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Existing artifact {key} differs from this source pair: {artifact}')
    if hash_payload and sha256(artifact) != metadata.get('artifact_sha256'):
        raise ValueError(f'Existing artifact payload SHA256 mismatch: {artifact}')
    return {'artifact': str(artifact), 'artifact_sha256': metadata['artifact_sha256'],
            'metadata_sha256': sha256(sidecar), 'payload_hash_verified': hash_payload,
            'numeric_validation': metadata.get('numeric_validation', 'not_run'),
            'tensor_bytes': sum(row['bytes'] for row in metadata['tensor_manifest'].values())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bf16-manifest', type=Path, required=True)
    p.add_argument('--awq-manifest', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--python', default=sys.executable, help='Base vLLM environment interpreter used for both native exporters')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--validate-tokens', type=int, default=2)
    p.add_argument('--reserve-gib', type=float, default=20)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--lock-status', type=Path)
    p.add_argument('--worker-config', type=Path)
    p.add_argument('--worker-report-path', type=Path)
    p.add_argument('--policy', choices=('threshold', 'expert_split', 'phase'))
    p.add_argument('--threshold', type=int)
    args = p.parse_args()
    if not 0 <= args.validate_tokens <= 32 or args.reserve_gib < 0:
        p.error('validate-tokens must be 0..32 and reserve-gib must be nonnegative')
    if args.worker_config and (args.policy is None or args.threshold is None or args.threshold < 1 or not args.worker_report_path):
        p.error('--worker-config requires explicit --policy, positive --threshold and --worker-report-path')
    for field in ('bf16_manifest', 'awq_manifest', 'out_dir'):
        setattr(args, field, getattr(args, field).resolve())
    bf16, awq = load_pair(args.bf16_manifest, args.awq_manifest)
    jobs = build_jobs(args, bf16, awq)
    existing = {str(i): check_artifact(job, bf16 if job['format'] == 'fp8' else awq, hash_payload=False)
                for i, job in enumerate(jobs)}
    plan = {'schema_version': 1, 'state': 'planned', 'model': bf16['model_key'], 'shape': bf16['shape'],
        'source_manifests': {'bf16': {'path': str(args.bf16_manifest), 'sha256': sha256(args.bf16_manifest)},
                             'awq': {'path': str(args.awq_manifest), 'sha256': sha256(args.awq_manifest)}},
        'base_checkpoint': bf16['base_checkpoint'], 'base_checkpoint_revision': bf16['base_checkpoint_revision'],
        'jobs': jobs, 'existing_artifacts_metadata_only': existing,
        'estimated_total_tensor_bytes': sum(job['estimated_tensor_bytes'] for job in jobs),
        'remaining_estimated_write_bytes': sum(job['estimated_tensor_bytes'] for i, job in enumerate(jobs) if existing[str(i)] is None) + (64 << 20),
        'reserve_gib': args.reserve_gib, 'execution_environment': 'vLLM 0.26.0, compressed-tensors 0.17.0',
        'responsibility': {'cpu': 'manifest/basis checks, source/artifact hashes, checkpoint reads and output writes',
                           'gpu': 'native online FP8 quantization, Marlin repacking, optional AWQ numeric diagnostic'},
        'quality_claim': False, 'nonexpert_precision_alignment': 'must be fixed independently for each serving/quality arm'}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not args.lock_status:
        p.error('--execute requires --lock-status from the live run_locked ancestor')
    from bench_serving import verify_lock
    lock = verify_lock(args.lock_status)
    # Check the actual child environment without importing Torch/CUDA here.
    versions = json.loads(subprocess.check_output([args.python, '-c',
        'import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) for p in ("vllm","compressed-tensors")}))'], text=True))
    if versions['vllm'].split('+')[0] != '0.26.0' or versions['compressed-tensors'] != '0.17.0':
        raise RuntimeError(f'Native exporters require the base serving environment, got {versions}')
    if any(existing.values()) and not args.resume:
        raise FileExistsError('Outputs already exist; --resume requires every existing artifact to match this exact pair')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.out_dir).free
    if free < plan['remaining_estimated_write_bytes'] + int(args.reserve_gib * 2**30):
        raise RuntimeError('Insufficient space for remaining sidecars and reserve')
    plan.update(state='running', lock=lock, versions=versions, started_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.out_dir / 'export_plan.json', plan)
    records, layers = [], []
    for job in jobs:
        manifest = bf16 if job['format'] == 'fp8' else awq
        started = time.monotonic()
        checked = check_artifact(job, manifest, hash_payload=True)
        if checked is None:
            directory = Path(job['artifact']).parent
            directory.mkdir(parents=True, exist_ok=True)
            attempt = 1
            while (directory / f'{job["format"]}_export_attempt{attempt}.log').exists():
                attempt += 1
            log = directory / f'{job["format"]}_export_attempt{attempt}.log'
            print(json.dumps({'event': 'export_start', 'layer': job['layer'], 'format': job['format'], 'log': str(log)}), flush=True)
            with log.open('w') as stream:
                result = subprocess.run(job['command'], stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                plan.update(state='failed', failed_job=job, failed_log=str(log), exit_code=result.returncode)
                write_json(args.out_dir / 'export_plan.json', plan)
                raise RuntimeError(f'Exporter failed with code {result.returncode}; retained log: {log}')
            checked = check_artifact(job, manifest, hash_payload=True)
        records.append({**job, **checked, 'elapsed_s_including_cpu_io': time.monotonic() - started})
        write_json(args.out_dir / 'completed_artifacts.json', {'complete': False, 'artifacts': records})
    for layer in range(bf16['shape']['layers']):
        pair = {row['format']: row for row in records if row['layer'] == layer}
        layers.append({'layer': layer, 'module': f'model.layers.{layer}.mlp.experts',
            'fp8_weights': pair['fp8']['artifact'], 'lowbit_weights': pair['awq']['artifact'],
            'worker_activation_basis_id': bf16['activation_bases'][str(layer)]['activation_basis_id']})
    complete = {**plan, 'state': 'complete', 'artifacts': records, 'worker_layers': layers,
                'completed_utc': datetime.now(timezone.utc).isoformat(),
                'actual_tensor_bytes': sum(row['tensor_bytes'] for row in records)}
    write_json(args.out_dir / 'sidecar_manifest.json', complete)
    write_json(args.out_dir / 'completed_artifacts.json', {'complete': True, 'artifacts': records})
    if args.worker_config:
        worker = {'schema_version': 1, 'policy': args.policy, 'threshold': args.threshold,
            'lowbit_format': 'awq', 'mechanism_only': False, 'allow_unverified_source': False,
            'release_original_weights': True, 'report_path': str(args.worker_report_path.resolve()), 'layers': layers}
        args.worker_config.parent.mkdir(parents=True, exist_ok=True)
        if args.worker_config.exists() and canonical_json(read_json(args.worker_config)) != canonical_json(worker):
            raise FileExistsError('Existing worker config differs; choose a new path')
        write_json(args.worker_config, worker)
    print(json.dumps({'state': 'complete', 'layers': len(layers), 'artifacts': len(records),
                      'manifest': str(args.out_dir / 'sidecar_manifest.json'),
                      'actual_tensor_bytes': complete['actual_tensor_bytes'], 'quality_claim': False}))


if __name__ == '__main__':
    main()

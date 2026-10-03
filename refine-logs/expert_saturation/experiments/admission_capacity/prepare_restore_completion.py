#!/usr/bin/env python3
"""Derive one first-output obligation ablation from the executed prefix package."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/admission_capacity/20260914_ltr_packing_r01'
OUT = ROOT/'outputs/admission_capacity/20260914_restore_completion_r01/preparation'


def change(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    meta = json.loads((BASE/'preparation/preparation.json').read_text())
    assert meta['archive_sha256'] == '1fea4a612af77a0062163cec33566765c2df25ade4021407756ec43b1734ef73'
    assert not OUT.exists(), 'Never overwrite a preparation attempt'
    base = BASE/'execution/readback/pkg'
    assert all(sha(base/n) == h for n,h in meta['files_sha256'].items())
    OUT.mkdir(); pkg = OUT/'pkg'; shutil.copytree(base, pkg)
    shutil.copy2(Path(__file__).with_name('restore_obligation.py'), pkg/'restore_obligation.py')
    p = pkg/'ltr_recompute_native.py'; s = p.read_text()
    s = s[s.index('import ast'):]
    s = '"""Executed rank-prefix backend plus optional recovery-to-first-output obligation.\nOnly actual PREEMPTED restores with past outputs create an obligation.\nOriginal LTR200/10 counters are unchanged; this is a component ablation.\n"""\n'+s
    s = change(s, 'from recovery_service_components import LTRCounters',
               'from recovery_service_components import LTRCounters\nfrom restore_obligation import RequestView, RestoreObligations')
    s = change(s, 'quantum=10, packing="fit_scan"):', 'quantum=10, packing="rank_prefix", complete_restores=False):')
    s = change(s, '    current = None', '    current = None\n    obligations, previous_live = RestoreObligations(), {}')
    s = change(s, '        nonlocal current', '        nonlocal current, previous_live')
    s = change(s, '        priorities = counters.begin_schedule(live)', '''        step = len(decisions)
        before_views = {rid: RequestView(r.status.name, int(r.num_output_tokens),
                        int(r.num_tokens-r.num_computed_tokens)) for rid,r in live.items()}
        completed = {rid for rid,r in previous_live.items() if rid not in live and r.is_finished()}
        obligation_begin = obligations.begin_schedule(step, before_views, completed)
        previous_live = dict(live)
        priorities = counters.begin_schedule(live)
        effective = obligations.effective_priorities(
            {rid: p if boost else 0 for rid,p in priorities.items()}, enabled=complete_restores)''')
    s = change(s, 'Candidate(rid, priorities[rid] if boost else 0, r.arrival_time,',
               'Candidate(rid, effective[rid], r.arrival_time,')
    s = change(s, '                      free_before=pool.get_num_free_blocks(),', '''                      complete_restores=complete_restores,
                      obligation_begin=obligation_begin,
                      obligations_before=list(obligation_begin['outstanding']),
                      free_before=pool.get_num_free_blocks(),''')
    s = change(s, '            counters.after_schedule(result.num_scheduled_tokens)', '''            resumed = [rid for rid in result.num_scheduled_tokens
                       if before_views[rid].status == "PREEMPTED"]
            history_remaining = {rid: max(0, (r.num_tokens+block_size-1)//block_size
                                 - len(owned.get(rid, ()))) for rid,r in live.items()}
            record['obligation_after'] = obligations.after_schedule(
                step, before_views, resumed, result.num_scheduled_tokens,
                current.victims, history_remaining, pool.get_num_free_blocks(),
                enabled=complete_restores, effective_priorities=effective)
            counters.after_schedule(result.num_scheduled_tokens)''')
    s = change(s, '    return decisions, counters, uninstall', '    return decisions, counters, obligations, uninstall')
    p.write_text(s)
    p = pkg/'run_probe.py'; s = p.read_text()
    s = change(s, "    args = parser.parse_args()", "    parser.add_argument('--complete-restores', choices=['off', 'on'], required=True)\n    args = parser.parse_args()\n    if args.packing != 'rank_prefix':\n        raise ValueError('restore comparison keeps rank-prefix fixed')")
    s = change(s, "    config['component']['packing'] = args.packing", "    config['component']['packing'] = args.packing\n    config['component']['complete_restores'] = args.complete_restores == 'on'\n    config['completion_policy'] = 'restore_completion_' + args.complete_restores")
    s = change(s, "'ltr_recompute_native.py', 'recovery_service_components.py']", "'ltr_recompute_native.py', 'recovery_service_components.py', 'restore_obligation.py']")
    s = change(s, "        decisions, counters, uninstall = install_component(scheduler,", "        dump(out/'resolved-scheduler-config.json', dict(long_prefill_token_threshold=scheduler.scheduler_config.long_prefill_token_threshold))\n        decisions, counters, obligations, uninstall = install_component(scheduler,")
    s = change(s, '            boost=True, threshold=200, quantum=10, packing=args.packing)', '            boost=True, threshold=200, quantum=10, packing=args.packing,\n            complete_restores=args.complete_restores == \'on\')')
    s = change(s, "                regime='steady', arrival_scale=1.0, run_id='measured', max_seconds=120)\n        finally:", "                regime='steady', arrival_scale=1.0, run_id='measured', max_seconds=120)\n            if raw['status'] == 'COMPLETE':\n                obligations.finalize(len(decisions), {}, [r['internal_request_id'] for r in raw['requests'] if r['status']=='completed'])\n        finally:")
    s = change(s, "            dump(out/'component-decisions.json', decisions)", "            dump(out/'component-decisions.json', decisions)\n            dump(out/'restore-obligations.json', obligations.snapshot())")
    # Existing warmup and engine settings remain exact. Phase markers only clarify logs.
    s = change(s, "        # Identical warmups in every process; do not run the pressure case as warmup.", "        print('PHASE APPLICATION_WARMUP_BEGIN', flush=True)\n        # Identical warmups in every process; do not run the pressure case as warmup.")
    s = change(s, '        set_empty_admission_cap(engine, args.cap)', "        print('PHASE APPLICATION_WARMUP_END', flush=True)\n        set_empty_admission_cap(engine, args.cap)")
    s = change(s, '        try:\n            raw = capture_with_memory', "        try:\n            print('PHASE MEASUREMENT_BEGIN', flush=True)\n            raw = capture_with_memory")
    s = change(s, "            if raw['status'] == 'COMPLETE':\n                obligations.finalize", "            print('PHASE MEASUREMENT_END', flush=True)\n            if raw['status'] == 'COMPLETE':\n                obligations.finalize")
    p.write_text(s)
    campaign = json.loads((pkg/'campaign.json').read_text()); campaign['comparison'] = 'restore_completion'
    for i,(cell,enabled) in enumerate(zip(campaign['cells'], ['off','on','on','off'])):
        cell.update(label=f'block{i//2}-d6-restore-{enabled}', boost='on', packing='rank_prefix', complete_restores=enabled)
    (pkg/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
    p=pkg/'run.sh'; s=p.read_text()
    s=change(s, ' label=$1; boost=$2; packing=$3', ' label=$1; boost=on; packing=rank_prefix; complete=$2')
    s=change(s, '--packing "$packing" --output-dir', '--packing "$packing" --complete-restores "$complete" --output-dir')
    start=s.index('run block0-d6-packing-fit_scan'); end=s.index('echo "CAMPAIGN_FINISHED',start)
    s=s[:start]+''.join(f"run {c['label']} {c['complete_restores']}\n" for c in campaign['cells'])+s[end:];p.write_text(s)
    for p in pkg.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
    subprocess.run(['bash','-n',str(pkg/'run.sh')],check=True)
    subprocess.run(['python3',str(pkg/'run_probe.py'),'--help'],check=True,stdout=subprocess.DEVNULL)
    inventory={str(p.relative_to(pkg)):sha(p) for p in sorted(pkg.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts}
    (pkg/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in inventory.items()))
    archive=OUT/'execution.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for n in list(inventory)+['SHA256SUMS']: t.add(pkg/n,arcname='pkg/'+n,recursive=False)
    reference=BASE/'execution/readback/results/block0-d6-packing-rank_prefix'
    final=dict(status='CPU_PREPARED_NATIVE_INTERFACE_UNRUN',gpu_executions=0,uploads=0,comparison='restore_completion',
        files_sha256=inventory,archive_sha256=sha(archive),cells=campaign['cells'],expected_runtime_sources=meta['expected_runtime_sources'],
        reference_cell=str(reference),reference_engine_args_sha256=sha(reference/'engine_args.json'),
        command='python3 '+str(Path(__file__).resolve()),source_script_sha256=sha(Path(__file__)),
        parent_archive_sha256=meta['archive_sha256'],prepared_at=datetime.now(timezone.utc).isoformat(),
        scope='First-output recovery obligation on the exact executed rank-prefix/LTR200/10 backend. CPU preparation only; full costs, failures and independent state retained. Not novelty, quality or an Oracle.')
    (OUT/'preparation.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(dict(archive_sha256=final['archive_sha256'],metadata_sha256=sha(OUT/'preparation.json'),files=len(inventory))))


if __name__=='__main__': main()

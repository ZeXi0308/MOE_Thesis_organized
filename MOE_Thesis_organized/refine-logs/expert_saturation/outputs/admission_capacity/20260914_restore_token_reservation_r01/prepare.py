#!/usr/bin/env python3
"""Freeze one token-budget reservation ablation from the executed S package."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p/'AGENTS.md').is_file())
S = REPO/'refine-logs/expert_saturation/outputs/admission_capacity/20260914_restore_completion_r01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'patch anchor count={text.count(old)}: {old!r}')
    return text.replace(old, new)


def load_module(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(plan):
    return dict(tokens=plan.tokens, victims=plan.victims,
                free_after_reservation=plan.free_after_reservation)


def fixtures(old, new):
    ref = S/'execution/readback/results/block0-d6-restore-on'
    decisions = json.loads((ref/'component-decisions.json').read_text())
    ledger = json.loads((ref/'restore-obligations.json').read_text())
    raw = json.loads((ref/'raw.json').read_text())
    assert len(decisions) == 1906 and ledger['active'] == {}
    actual, prior = decisions[406], raw['scheduler_steps'][405]
    protected = actual['obligations_before'][0]
    scheduled = {row['internal_request_id']: row for row in prior['scheduled']}
    restored = scheduled[protected]
    ready_rows = [row for row in prior['scheduled']
                  if row['scheduled_tokens'] == 1 and row['recompute_tokens'] == 0]
    assert (len(ready_rows), actual['free_before'], actual['tokens'], actual['victims']) == (
        30, 148, {protected: 1024}, [])
    rows = [new.Candidate(protected, -2, 0, True,
        (restored['computed_after']+15)//16,
        (restored['prompt_tokens']+restored['output_tokens_before']+15)//16,
        restored['prompt_tokens']+restored['output_tokens_before']-restored['computed_after'])]
    for index, row in enumerate(ready_rows, 1):
        owned = (row['computed_after']+15)//16
        history = (row['computed_after']+1+15)//16
        rows.append(new.Candidate(row['internal_request_id'], 0, index, True,
                                  owned, history, 1))
    assert sum(r.history_blocks-r.owned_blocks for r in rows[1:]) == 3
    old_first = old.plan(rows, 148, 1024, 32, 0, packing='rank_prefix')
    off_first = new.plan(rows, 148, 1024, 32, 0, packing='rank_prefix',
                         reserve_ready_tokens=False)
    on_first = new.plan(rows, 148, 1024, 32, 0, packing='rank_prefix',
                        reserve_ready_tokens=True)
    assert result(old_first) == result(off_first)
    assert (on_first.tokens[protected], len(on_first.tokens),
            sum(on_first.tokens.values()), on_first.victims) == (994, 31, 1024, [])
    low = [new.Candidate('restore', -2, 0, True, 1, 1, 10)] + [
        new.Candidate(f'impossible{i}', 0, i+1, True, 0, 1, 1) for i in range(3)]
    low_on = new.plan(low, 0, 2, 4, packing='rank_prefix', reserve_ready_tokens=True)
    assert (low_on.tokens, low_on.victims, sum(low_on.tokens.values())) == (
        {'restore': 1}, [], 1)
    victim = [new.Candidate('evictor', -3, 0, False, 0, 1, 1),
              new.Candidate('restore', -2, 1, True, 1, 1, 9),
              new.Candidate('ready', 0, 2, True, 1, 1, 1),
              new.Candidate('victim', 0, 3, True, 1, 1, 1)]
    victim_on = new.plan(victim, 0, 5, 3, packing='rank_prefix',
                         reserve_ready_tokens=True)
    assert (victim_on.tokens, victim_on.victims) == (
        {'evictor': 1, 'restore': 3, 'ready': 1}, ['victim'])
    cases = [(rows, 148, 1024, 32), (low, 0, 2, 4), (victim, 0, 5, 3)]
    assert all(result(old.plan(a,b,c,d,packing='rank_prefix')) ==
               result(new.plan(a,b,c,d,packing='rank_prefix',reserve_ready_tokens=False))
               for a,b,c,d in cases)
    return dict(status='PASS', gpu_executions=0, reference_cell=str(ref),
        flag_off=dict(reference_steps=1906, reference_unresolved_obligations=0,
            differential_fixtures=3, semantic_basis='false selects the unchanged token_budget cap'),
        first_delta=dict(step=406, free_blocks=148, protected_owned_blocks=63,
            protected_history_blocks=205, protected_remaining_blocks=142,
            ready_requests=30, ready_growth_blocks=3, guard_all_tokens={protected:1024},
            guard_residual_restore_tokens=994, guard_residual_ready_tokens=30,
            total_tokens=1024, victims=[]),
        boundaries=dict(low_budget_tokens=low_on.tokens,
            impossible_ready_did_not_borrow_tokens=True,
            already_selected_victim_not_reserved=True, victim_case=result(victim_on)),
        claim='CPU plan/interface fixture only; no future runtime outcome is predicted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='preparation')
    args = parser.parse_args()
    if args.attempt != 'preparation' and not args.attempt.startswith('preparation_attempt'):
        raise ValueError('attempt must be preparation or preparation_attempt*')
    out = HERE/args.attempt
    if out.exists():
        raise FileExistsError(f'refuse overwrite: {out}')
    parent = json.loads((S/'preparation/preparation.json').read_text())
    base = S/'preparation/pkg'
    assert sha(S/'preparation/execution.tar.gz') == parent['archive_sha256']
    assert all(sha(base/name) == digest for name,digest in parent['files_sha256'].items())
    out.mkdir(); pkg = out/'pkg'; pkg.mkdir()
    for name in parent['files_sha256']:
        target = pkg/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base/name, target)
    p = pkg/'ltr_recompute_native.py'; text = p.read_text()
    text = change(text, 'packing="fit_scan"):',
                  'packing="fit_scan", reserve_ready_tokens=False):')
    text = change(text, '        amount = min(row.pending_tokens, token_budget, chunk_limit or token_budget)\n', '')
    text = change(text, '        victims.extend(trial)', '''        ready = (sum(candidate.resident and candidate.pending_tokens == 1
                     and candidate.request_id not in victims
                     and candidate.request_id not in trial for candidate in ordered[i + 1:])
                 if reserve_ready_tokens and row.priority == -2 and row.pending_tokens > 1 else 0)
        amount = min(row.pending_tokens, token_budget, chunk_limit or token_budget,
                     max(1, token_budget - ready))
        victims.extend(trial)''')
    text = change(text, 'quantum=10, packing="rank_prefix", complete_restores=False):',
                  'quantum=10, packing="rank_prefix", complete_restores=False, reserve_ready_tokens=False):')
    text = change(text, 'config.long_prefill_token_threshold, packing=packing)',
                  'config.long_prefill_token_threshold, packing=packing, reserve_ready_tokens=reserve_ready_tokens)')
    text = change(text, '                      complete_restores=complete_restores,',
                  '                      complete_restores=complete_restores,\n                      reserve_ready_tokens=reserve_ready_tokens,')
    p.write_text(text)
    p = pkg/'run_probe.py'; text = p.read_text()
    text = change(text, "    args = parser.parse_args()", "    parser.add_argument('--reserve-ready-tokens', choices=['off', 'on'], required=True)\n    args = parser.parse_args()")
    text = change(text, "    if args.packing != 'rank_prefix':\n        raise ValueError('restore comparison keeps rank-prefix fixed')", "    arm = (args.packing, args.complete_restores, args.reserve_ready_tokens)\n    if arm not in {('fit_scan','off','off'), ('rank_prefix','on','off'), ('rank_prefix','on','on')}:\n        raise ValueError('unsupported frozen token-reservation arm')")
    text = change(text, "    config['completion_policy'] = 'restore_completion_' + args.complete_restores", "    config['component']['reserve_ready_tokens'] = args.reserve_ready_tokens == 'on'\n    variant = 'fit_scan' if args.packing == 'fit_scan' else ('guard_residual' if args.reserve_ready_tokens == 'on' else 'guard_all')\n    config['variant'] = variant\n    config['completion_policy'] = 'restore_token_reservation_' + variant")
    text = change(text, "            complete_restores=args.complete_restores == 'on')", "            complete_restores=args.complete_restores == 'on',\n            reserve_ready_tokens=args.reserve_ready_tokens == 'on')")
    p.write_text(text)
    cells=[]
    for block, variants in [(0,['fit_scan','guard_all','guard_residual']),
                            (1,['guard_residual','guard_all','fit_scan'])]:
        for variant in variants:
            fit=variant=='fit_scan'; residual=variant=='guard_residual'
            cells.append(dict(label=f'block{block}-{variant}', variant=variant, boost='on',
                packing='fit_scan' if fit else 'rank_prefix', kv_cache_bytes=13960740864,
                usable_blocks=6656, complete_restores='off' if fit else 'on',
                reserve_ready_tokens='on' if residual else 'off'))
    campaign=dict(cells=cells, comparison='restore_token_reservation',
        question='Does reserving ready-token budget remove the first observed restore monopoly without changing recovery priority or resources?',
        interpretation='First-divergence old-cohort test only; no future benefit prediction or parameter scan.')
    (pkg/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
    p=pkg/'run.sh'; text=p.read_text()
    text=change(text, ' label=$1; boost=on; packing=rank_prefix; complete=$2',
                ' label=$1; boost=on; packing=$2; complete=$3; reserve=$4')
    text=change(text, '--complete-restores "$complete" --output-dir',
                '--complete-restores "$complete" --reserve-ready-tokens "$reserve" --output-dir')
    start=text.index('run block0-d6-restore-off'); end=text.index('echo "CAMPAIGN_FINISHED',start)
    runs=''.join(f"run {c['label']} {c['packing']} {c['complete_restores']} {c['reserve_ready_tokens']}\n" for c in cells)
    (pkg/'run.sh').write_text(text[:start]+runs+text[end:])
    for source in pkg.glob('*.py'):
        ast.parse(source.read_text(), filename=str(source))
    subprocess.run(['bash','-n',str(pkg/'run.sh')],check=True)
    subprocess.run(['python3',str(pkg/'run_probe.py'),'--help'],check=True,stdout=subprocess.DEVNULL)
    old=load_module('old_restore_ltr',base/'ltr_recompute_native.py')
    new=load_module('reserved_restore_ltr',pkg/'ltr_recompute_native.py')
    fixture=fixtures(old,new); (out/'fixture-results.json').write_text(json.dumps(fixture,indent=2)+'\n')
    inventory={str(x.relative_to(pkg)):sha(x) for x in sorted(pkg.rglob('*'))
               if x.is_file() and x.name!='SHA256SUMS'}
    (pkg/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in inventory.items()))
    archive=out/'execution.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        for name in [*inventory,'SHA256SUMS']:
            bundle.add(pkg/name,arcname='pkg/'+name,recursive=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip())
    metadata=dict(status='CPU_PREPARED_NATIVE_INTERFACE_UNRUN',gpu_executions=0,uploads=0,
        comparison='restore_token_reservation',repository_head=head,repository_dirty=dirty,
        parent_archive_sha256=parent['archive_sha256'],parent_files=len(parent['files_sha256']),
        files_sha256=inventory,archive_sha256=sha(archive),fixture_results_sha256=sha(out/'fixture-results.json'),
        cells=cells,expected_runtime_sources=parent['expected_runtime_sources'],
        reference_cell=fixture['reference_cell'],reference_engine_args_sha256=parent['reference_engine_args_sha256'],
        command=shlex.join([sys.executable,*sys.argv]),source_script_sha256=sha(Path(__file__)),
        prepared_at=datetime.now(timezone.utc).isoformat(),
        scope='Exact S backend and old cohort; CPU preparation only. First-divergence token-budget ablation, not a future-performance claim.')
    (out/'preparation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),archive_sha256=metadata['archive_sha256'],
                          metadata_sha256=sha(out/'preparation.json'),files=len(inventory))))


if __name__ == '__main__':
    main()

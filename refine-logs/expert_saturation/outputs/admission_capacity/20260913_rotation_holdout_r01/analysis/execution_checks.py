import csv
import hashlib
import json
from pathlib import Path

H = Path('/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_holdout_r01')
r = H/'execution03'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)
state = read(r/'execution.json')
manifest = read(r/'frozen/campaign.json')
require(state['status'] == 'COMPLETE', 'campaign is not complete')
require([c['label'] for c in state['cells']] == [c['label'] for c in manifest['cells']], 'run order differs from frozen order')
require(sha(r/'execution.tar.gz') == state['archive_sha256'] == '0ead2b4c9cf935ddda491646139e73073164f6e0250384066226f8b2fb9718f7', 'frozen upload hash differs')
rows, runtimes = [], []
for cell in state['cells']:
    label = cell['label']; d = r/'gpu_results'/label
    require(cell['status'] == 'READ_BACK' and cell['returncode'] == 0 and cell['terminal']['status'] == 'COMPLETE', label+' incomplete')
    require(sha(r/(label+'.tar.gz')) == cell['archive_sha256'], label+' archive hash differs')
    env, raw, after = [read(d/n) for n in ('environment.json','raw.json','gpu-after.json')]
    require(not env['gpu_before']['compute_processes'].strip(), label+' initialization had compute process')
    for name, expected in env['source_sha256'].items():
        require(sha(r/'frozen'/name) == expected, label+' source differs '+name)
    require((env['torch'],env['vllm'],env['transformers']) == ('2.11.0+cu130','0.26.0','5.15.1'), label+' versions differ')
    stages = [env['gpu_before'], raw['gpu_before'], after]
    require(all(state['gpu_uuid'] in s['device'] for s in stages), label+' GPU identity differs')
    before_pids = [v[0].strip() for v in csv.reader(raw['gpu_before']['compute_processes'].splitlines()) if v]
    after_pids = [v[0].strip() for v in csv.reader(after['compute_processes'].splitlines()) if v]
    require(len(before_pids) == 1 and before_pids == after_pids, label+' boundary processes differ')
    runtimes.append(env['vllm_source_sha256'])
    rows.append(dict(label=label, archive_sha256=cell['archive_sha256'], raw_sha256=sha(d/'raw.json'), source_fingerprints_match=True, boundary_process_check=True, worker_pid=int(before_pids[0])))
require(all(x == runtimes[0] for x in runtimes), 'runtime source fingerprints differ')
require(runtimes[0]['v1/core/sched/scheduler.py'] == '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941', 'scheduler source differs')
result = dict(status='PASS_CAPTURED_EXECUTION_PROVENANCE', cells=rows, frozen_archive_sha256=state['archive_sha256'], campaign_manifest_sha256=sha(r/'frozen/campaign.json'), runtime_source_sha256=runtimes[0], scope='Initialization, pre-measurement and post-measurement GPU process observations; these are not continuous exclusivity or clock-control guarantees. Performance qualification is in analysis.json. Failed earlier attempts remain separate.')
out = H/'analysis/execution_checks.json'
with out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],cells=len(rows),output=str(out))))

"""Only the new cadence/configuration and diagnostic-to-timing boundaries; no GPU."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

root=Path(__file__).resolve().parent
prior=root.parent/'20260915_natural_save_scope_timing_r01'
sys.path.insert(0,str(root/'pkg'))
from absence_rotation import RotationConfig
import analyze_cadence as analysis
import run_recovery_cadence as runner_module

for name in ('analyze_cadence.py','controller.py','pkg/run_recovery_cadence.py'):
    ast.parse((root/name).read_text(),filename=name)
subprocess.run(['bash','-n',str(root/'pkg/run.sh')],check=True)
subprocess.run([sys.executable,str(root/'pkg/run_recovery_cadence.py'),'--help'],check=True,stdout=subprocess.DEVNULL)
manifest=json.loads((prior/'manifest.json').read_text())
unchanged=[n for n in manifest if (n.startswith('pkg/') and n not in
    ('pkg/run.sh','pkg/run_save_scope_timing.py')) or n=='saved_kv_analysis_base.py']
for name in unchanged:
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest[name],name

runner=ast.parse((root/'pkg/run_recovery_cadence.py').read_text())
update=next(n for n in ast.walk(runner) if isinstance(n,ast.Call)
    and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
    and n.func.value.id=='config' and n.func.attr=='update')
expression=compile(ast.Expression(update),'actual-runner-config','eval')
configs={}
for variant,cooldown in (('current',20),('eager',0),('native_full_native',None)):
    config=json.loads((root/'pkg/inputs/config.json').read_text())
    eval(expression,dict(config=config,args=SimpleNamespace(variant=variant),cooldown=cooldown,
        diagnostic=False,rotation_enabled=variant!='native_full_native',lengths=[334,3011],KV_BYTES=8592031744,RotationConfig=RotationConfig))
    assert config['global_cooldown_steps']==cooldown
    if variant=='native_full_native':
        assert config['rotation_config'] is None and config['store_scope']=='native_full'
    else:
        assert config['rotation_config']['min_steps_between_swaps']==cooldown and config['store_scope']=='selected'
    assert config['ignore_eos'] is False and config['min_tokens']==0
    configs[variant]=config
assert analysis.normalize_config(configs['current'])==analysis.normalize_config(configs['eager'])
assert analysis.normalize_system_config(configs['current'])==analysis.normalize_system_config(configs['native_full_native'])
changed=deepcopy(configs['eager']);changed['rotation_config']['min_residency_steps']+=1
assert analysis.normalize_config(configs['current'])!=analysis.normalize_config(changed)
changed=deepcopy(configs['eager']);changed['store_scope']='native_full'
assert analysis.normalize_config(configs['current'])!=analysis.normalize_config(changed)
calls=[n for n in ast.walk(runner) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
light=[n for n in calls if n.func.id=='measure_episode']
assert len(light)==1 and any(k.arg=='record_preemptions' and k.value.value is True for k in light[0].keywords)
diagnostic=[n for n in calls if n.func.id=='capture_with_memory' and any(k.arg=='allow_preemption' for k in n.keywords)]
assert len(diagnostic)==1
shell=(root/'pkg/run.sh').read_text()
assert shell.index('--qualification-only')<shell.index('for cell in block0-native_full_native block0-current block0-eager block1-eager block1-current block1-native_full_native')
assert '--measurement-mode diagnostic' in shell and '--measurement-mode performance' in shell
assert "'diagnostic-eager','block0-native_full_native','block0-current','block0-eager','block1-eager','block1-current','block1-native_full_native'" in (root/'controller.py').read_text()
calls=[]
def installer(scheduler,**kwargs):
    calls.append(kwargs)
    return kwargs,lambda:kwargs
cs=SimpleNamespace(config=SimpleNamespace(offload_prompt_only=False))
scheduler=SimpleNamespace(connector=SimpleNamespace(connector_scheduler=cs))
for variant,cooldown in (('current',20),('eager',0)):
    value,_=runner_module.install_policy(scheduler,None,16,variant,False,cooldown,installer)
    assert value['global_cooldown_steps']==cooldown and value['store_scope']=='selected'
    assert value['population_mode']=='open' and value['save'] is True
native,uninstall=runner_module.install_policy(scheduler,None,16,'native_full_native',False,None,installer)
assert len(calls)==2 and native['status']=='NOT_APPLICABLE' and native['applied_rotations'] is None
assert native['native_calc_overridden'] is False and uninstall() is native
cs._calc_num_offloadable_tokens=lambda:0
try:runner_module.install_policy(scheduler,None,16,'native_full_native',False,None,installer)
except RuntimeError:pass
else:raise AssertionError('Overridden native saving method must fail before measurement')
commit=lambda step,reason='READY':dict(event='commit_check',step=step,reason=reason)
witness=analysis.cadence_commit_witness(dict(applied_rotations=2,events=[commit(10),commit(15,'CANCEL'),commit(29)]))
assert witness['sub20_successful_commit_intervals'][0]['steps']==19
assert not analysis.cadence_commit_witness(dict(applied_rotations=2,events=[commit(10),commit(30)]))['sub20_successful_commit_intervals']
try:analysis.cadence_commit_witness(dict(applied_rotations=1,events=[commit(10),commit(29)]))
except ValueError:pass
else:raise AssertionError('Proposed READY metadata alone cannot replace actual applied rotation count')
print(f'PASS: {len(unchanged)} unchanged F payloads; actual20/0 configs differ only in cadence; full/native mode skips installer and rejects overridden saving; fixed budgets/EOS/common sparse timing retained; actual sub20 commit witness gates timing. GPU UNRUN.')

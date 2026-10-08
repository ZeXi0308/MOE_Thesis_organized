#!/usr/bin/env python3
"""One native static-cap comparison; frozen observed child and resources."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'run_observed_group.py'
PARENT_SHA = 'd1d8325cf4796a066b151bb04b9914bce3e595782a24be4bc38d877dd0e0bb02'
CAPS = [256,224,224,256]


def load_parent():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest()!=PARENT_SHA:
        raise RuntimeError('Frozen observed group changed')
    spec=importlib.util.spec_from_file_location('simple_cap_observed_parent',PARENT)
    observed=importlib.util.module_from_spec(spec);spec.loader.exec_module(observed)
    parent=observed.load_parent();original=parent.adapted_source
    def adapted_source():
        text,sources=original()
        def replace(old,new):
            nonlocal text
            if text.count(old)!=1:raise RuntimeError('Static-cap boundary changed: '+old[:100])
            text=text.replace(old,new)
        replace("p['sequence'] != ['native', 'wait_release', 'wait_release', 'native'] or p['cap'] != 256",
                "p['sequence'] != ['native'] * 4 or p['caps'] != [256,224,224,256] or p['per_cell_seconds'] != 450")
        replace('Requires fixed normal-capacity native/recovery-start-gate ABBA',
                'Requires one fixed native static-cap256/224/224/256 group,450s per child')
        replace("tasks = [(256, mode) for mode in p['sequence']]", "tasks = [(cap, 'native') for cap in p['caps']]")
        replace("parser.add_argument('--wait-lock-seconds', type=float, default=0)",
                "parser.add_argument('--wait-lock-seconds', type=float, default=1800)")
        replace('    args = parser.parse_args()',
                "    args = parser.parse_args()\n    if args.wait_lock_seconds != 1800:\n        parser.error('This bounded static-cap group requires wait-lock-seconds1800')")
        old="""            if i == 1:
                if any(type(value) is not int for value in counts.values()):
                    raise RuntimeError('First wait_release action evidence missing; no remainder started')
                if counts['action_count'] == 0 or counts['executed_breaks'] == 0:
                    receipt.update(status='STOP_NO_ACTION', stop_reason='First wait_release had no executed start-gate break; reverse cells not started')
                    break"""
        new="""            if any(type(value) is not int or value != 0 for value in counts.values()):
                raise RuntimeError('Static-cap baselines require recorded zero B actions in every native cell')
            resolved = json.loads((cell/'output/resolved-scheduler-config.json').read_text())
            engine_args = json.loads((cell/'output/engine_args.json').read_text())
            actual_cap = resolved.get('max_num_running_reqs')
            receipt['cells'][-1].update(actual_cap=actual_cap,
                compiled_engine_max_num_seqs=engine_args.get('max_num_seqs'))
            write(session/'receipt.json', receipt)
            if type(actual_cap) is not int or actual_cap != cap or engine_args.get('max_num_seqs') != 256:
                raise RuntimeError('Actual runtime cap or fixed compiled engine maxseq differs')"""
        replace(old,new)
        return text,[*sources,Path(__file__)]
    parent.adapted_source=adapted_source
    observed.load_parent=lambda:parent
    return observed


def self_check():
    observed=load_parent();parent=observed.load_parent()
    parent.main=lambda:parent.adapted_source()
    text,_=observed.main();tree=ast.parse(text);compile(tree,'<simple-cap-controller>','exec')
    assert text.count("str(ROOT/'run_observed_fixed_cell.py'), '--inputs'")==1
    assert "tasks = [(cap, 'native') for cap in p['caps']]" in text
    assert "--max-num-seqs', str(p['engine_max_num_seqs'])" in text
    assert 'STOP_NO_ACTION' not in text.replace("or receipt['status'] == 'STOP_NO_ACTION'",'')
    assert "actual_cap != cap or engine_args.get('max_num_seqs') != 256" in text
    resource=parent.load_parent().load_parent().load_parent().load_parent().load_parent()
    assert resource.LOCK_IDENTITY==(2304,4312099778) and resource.MINIMUM_FREE_BYTES==2684354560
    # Check the original child uses signed cap AFTER original warmups, while its
    # compiled shape remains max_num_seqs. No vLLM import or native execution.
    normal_path=ROOT.parents[1]/'normal_capacity/run_cell.py'
    spec=importlib.util.spec_from_file_location('simple_cap_normal_check',normal_path)
    normal=importlib.util.module_from_spec(spec);spec.loader.exec_module(normal)
    source=normal.adapted_source()
    assert source.count("set_empty_admission_cap(engine, config['cap'])")==1
    assert source.index('PHASE APPLICATION_WARMUP_END')<source.index("set_empty_admission_cap(engine, config['cap'])")
    assert 'max_model_len=4096, max_num_seqs=args.max_num_seqs' in source
    import json
    a=json.loads((ROOT/'inputs/cap256/config.json').read_text())
    b=json.loads((ROOT/'inputs/cap224/config.json').read_text())
    assert {k for k in a if a[k]!=b[k]}=={'cap'} and b['cap']==224 and a['cap']==256
    assert (ROOT/'inputs/cap224/workload.json').read_bytes()==(ROOT/'inputs/cap256/workload.json').read_bytes()
    assert a['engine_max_num_seqs']==b['engine_max_num_seqs']==256
    print('PASS: only signed runtime cap differs; original observed child/warmups/compiled256 retained; actual-cap check and zero-native-action validation; same common lock/2.5GiB;450s children/1800s lock wait. CPU only.')


def main():
    if sys.argv[1:]==['--self-check']:
        self_check();return 0
    return load_parent().main()


if __name__=='__main__':raise SystemExit(main())

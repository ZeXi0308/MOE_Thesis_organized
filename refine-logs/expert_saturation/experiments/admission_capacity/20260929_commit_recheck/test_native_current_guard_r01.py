"""Replay recorded qualified choices through the actual off/on victim guard."""
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PKG=ROOT/'candidate_native_current_guard_r01/pkg'
sys.path.insert(0,str(PKG))
from bidkv_score import bidkv_order
tree=ast.parse((PKG/'staged_store_rotation.py').read_text())
names={'_rank_native_victim','_native_victim_current_guard'}
module=ast.fix_missing_locations(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]))
env=dict(bidkv_order=bidkv_order)
exec(compile(module,'<actual current guard>','exec'),env)
select=env['_native_victim_current_guard']
seen=changed=0
for file in (ROOT/'moe-a-native-bidkv-score-session-r01-20261002').glob('cell-*/archive/selective-store.json'):
    for row in json.loads(file.read_text())['victim_decisions']:
        rows=row['candidates'];rule=row['rule'];unknown=row['fallback_unknown']
        by_index={r['index']:r for r in rows}
        off,unrestricted,eligible=select(rows,rule,unknown,False)
        assert by_index[off]['request']==row['selected']
        on,unrestricted2,eligible2=select(rows,rule,unknown,True)
        assert unrestricted==unrestricted2 and eligible==eligible2
        if eligible:
            assert on!=rows[0]['index'] and len(rows)>1 and not unknown
            changed+=1
        else:
            assert on==off
        # Unknown state and a singleton always preserve native tail.
        assert select(rows,rule,True,True)[0]==rows[-1]['index']
        assert select(rows[-1:],rule,False,True)[0]==rows[-1]['index']
        seen+=1
assert seen>0 and changed>0
print('PASS:',seen,'recorded choices retain exact off behavior;',changed,'counterfactual current exclusions; unknown/singleton native fallback. GPU UNRUN.')

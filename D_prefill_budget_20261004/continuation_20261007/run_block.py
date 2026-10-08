"""Run one frozen four-arm block under the inherited shared-lock driver."""
import json, os, sys
from pathlib import Path
r=Path(__file__).resolve().parent
d=json.loads((r/'design_g.json').read_text())
i=int(sys.argv[1]);assert 1<=i<=3
b=d['blocks'][i-1]
cmd=['bash',str(r/'launch.sh'),'--output',sys.argv[2],'--workload',b['workload'],'--policies',','.join(b['policies']),'--warm-policies',','.join(d['warm_policies']),'--target-ms','24','--wait-lock','7200']
os.execvpe(cmd[0],cmd,os.environ)

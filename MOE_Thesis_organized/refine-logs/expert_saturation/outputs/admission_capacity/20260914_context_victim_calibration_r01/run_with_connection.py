"""Run the staged driver through this task's live foreground SSH master."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

path=Path(__file__).with_name('execute.py')
spec=importlib.util.spec_from_file_location('staged_context_driver',path)
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
control='/private/tmp/moe-context-01a0953f.sock'
driver.SSH[driver.SSH.index('-S')+1]=control
driver.SCP=[('ControlPath='+control) if s.startswith('ControlPath=') else s for s in driver.SCP]
with (path.parent/'execution/connection_run.json').open('x') as f:
    json.dump(dict(control=control,started_unix_s=time.time(),
        driver_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Transport override only; staged package and experiment settings unchanged'),f,indent=2)
sys.argv=[str(path),'run']
driver.main()

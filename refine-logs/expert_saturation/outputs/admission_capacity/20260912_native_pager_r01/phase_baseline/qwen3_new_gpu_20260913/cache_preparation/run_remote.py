import os,runpy,sys
from pathlib import Path
p=Path(__file__).resolve().parent
assert b'/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r02/run_native_pager.py' in Path('/proc/5127/cmdline').read_bytes().split(b'\0')
os.sched_setaffinity(0,{8})
sys.argv=[str(p/'collect_shards.py'),'--manifest','/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r02/qwen3.manifest.json','--worker-pid','5127','--cache-dir','/root/qwen3-compressed-cache-ad44e777','--log','/root/qwen3-compressed-cache-ad44e777/collector.jsonl']
runpy.run_path(str(p/'collect_shards.py'),run_name='__main__')

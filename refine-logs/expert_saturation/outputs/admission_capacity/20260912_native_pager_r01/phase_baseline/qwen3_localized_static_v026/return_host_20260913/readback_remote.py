import json,sys,tarfile
from pathlib import Path
stage=Path('/root/autodl-tmp/qwen3-localized-static-v026-launch-r01');source=Path('/root/autodl-tmp/qwen3-localized-static-v026-r01')
def ensure_quiescent():
 assert not Path('/proc/89182').exists(), 'original worker PID still exists'
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:argv=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  assert not any(a.startswith(str(stage).encode()+b'/') and a.endswith(b'.py') for a in argv), 'original producer still exists'
ensure_quiescent()
files=[(p,p.stat().st_size,p.stat().st_mtime_ns,prefix+'/'+p.relative_to(root).as_posix()) for root,prefix in [(stage,'launch'),(source,'results')] for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and 'shard_workspace' not in p.parts]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as a:
 for p,size,mtime,name in files:
  a.add(p,arcname=name)
  assert p.stat().st_size==size and p.stat().st_mtime_ns==mtime, str(p)
ensure_quiescent()

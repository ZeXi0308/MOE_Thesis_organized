from pathlib import Path
import hashlib,json,subprocess,sys
p=Path('/root/autodl-tmp/qwen3-localized-static-v026-launch-r02')
assert not (p/'launch.json').exists()
root=p/'detached_receipts';root.mkdir(exist_ok=False)
cmd=[sys.executable,str(p/'detached_launch.py'),'--package-dir',str(p),'--python',sys.executable,'--entry-sha256','1beb9f9605274cec102f62f2825c17cef8a89244f45275d4c149a16a472bfc59','--protocol-sha256','b6ac9329f04c6ee462e47fa7f4f1ae45cf3c50105ae48bdc337509efcad80807','--receipt-root',str(root)]
r=subprocess.run(cmd,capture_output=True,text=True);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');raise SystemExit(r.returncode)

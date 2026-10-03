#!/usr/bin/env python3
"""Read back the completed native oldest-admission block without replacing files."""
import argparse,hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--sha256",required=True)
args=parser.parse_args()
archive=root/"moe-a-native-oldest-admission-r02-20261002-readback.tar.gz"
session=root/"moe-a-native-oldest-admission-session-r02-20261002"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(archive)==args.sha256
assert not session.exists()
with tarfile.open(archive) as tar:
 members=tar.getmembers()
 assert all(m.name.startswith(session.name+"/") and ".." not in Path(m.name).parts
            and not m.issym() and not m.islnk() for m in members)
 tar.extractall(root,filter="data")
receipt=json.loads((session/"receipt.json").read_text())
assert receipt["status"]=="CELLS_COMPLETE"
count=0
for i,cell in enumerate(receipt["cells"]):
 assert cell["exit_code"]==0 and cell["archive_status"]=="VERIFIED"
 cell_dir=session/f"cell-{i:02d}-{cell['arm']}"
 for name,digest in json.loads((cell_dir/"output_sha256.json").read_text()).items():
  assert sha(cell_dir/"archive"/name)==digest,name
  count+=1
files=[f for f in session.rglob("*") if f.is_file()]
result=dict(status="LOCAL_RAW_VERIFIED",archive_sha256=args.sha256,files=len(files),
 raw_bytes=sum(f.stat().st_size for f in files),verified_output_files=count,
 started=receipt["started_unix_s"],finished=receipt["finished_unix_s"],
 elapsed=receipt["elapsed_wall_s"])
(root/"A_NATIVE_OLDEST_ADMISSION_READBACK_R02_20261002.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))


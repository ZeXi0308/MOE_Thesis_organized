"""Verify a returned block archive and extract without overwriting any evidence."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

root = Path(__file__).resolve().parent
block = sys.argv[1]
assert block in ("forward", "reverse")
record = json.loads((root / f"ARCHIVE-{block}.json").read_text())
archive = root / f"readback-{block}.tar.gz"
assert archive.stat().st_size == record["bytes"]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == record["sha256"]
destination = root / "readback"
with tarfile.open(archive) as source:
    members = source.getmembers()
    assert len(members) == record["files"]
    for member in members:
        assert member.isfile() and not member.name.startswith("/") and ".." not in Path(member.name).parts
        data = source.extractfile(member).read()
        if member.name.endswith(".json"):
            json.loads(data)
        target = destination / member.name
        if target.exists():
            assert target.read_bytes() == data, f"existing evidence differs: {target}"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as out:
                out.write(data)
assert (destination / "execution.tar.gz").read_bytes() == (root / "execution.tar.gz").read_bytes()
folder = destination / "results" / block
raw = [json.loads(p.read_text()) for p in sorted(folder.glob("*-raw.json"))]
process = json.loads((destination / "results" / f"{block}-process.json").read_text())
transfer = dict(block=block, archive_sha256=record["sha256"], archive_bytes=record["bytes"],
    archive_files=record["files"], byte_verified=True, all_json_readable=True,
    measured_episodes=sum(r["phase"] == "cell" for r in raw),
    warmup_episodes=sum(r["phase"] == "warmup" for r in raw),
    all_raw_complete=bool(raw) and all(r["status"] == "COMPLETE" for r in raw),
    process_exit_code=process["exit_code"], remote_originals_retained=True)
with (root / f"TRANSFER-{block}.json").open("x") as out:
    json.dump(transfer, out, indent=2)
    out.write("\n")
print(json.dumps(transfer, indent=2))

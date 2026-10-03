# GitHub large-data storage

The workspace snapshot published on 2026-09-12 includes all files allowed by the
existing Git ignore rules. Those rules continue to exclude local environments,
caches, materialized runtime packages, and previously excluded historical payloads.

Sixteen newly published `raw.json` files exceed 100 MiB each. Git stores their
lossless `raw.json.gz` counterparts beside the original paths. The local originals
remain unchanged and are ignored by exact path. All repeats are included.

[github-large-files.json](github-large-files.json) records the original paths,
byte counts, SHA-256 digests, and compressed sizes. The 4,498,392,971 original bytes
occupy 237,846,009 compressed bytes. Each compressed file was decompressed and
verified against its original before publication.

From the repository root, restore the original paths before running tools that
read `raw.json`. This command requires Python 3 and approximately 4.50 GB of free
space when none of the originals are present. It verifies existing files and
does not overwrite them.

```sh
python3 - <<'PY'
import gzip
import hashlib
import json
from pathlib import Path

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

manifest = json.loads(Path("docs/github-large-files.json").read_text())
for entry in manifest["files"]:
    path = Path(entry["path"])
    if path.exists():
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise RuntimeError(f"Existing file differs; left unchanged: {path}")
    else:
        target = path.open("xb")
        try:
            with target, gzip.open(entry["gzip_path"], "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise RuntimeError(f"Restored data failed verification: {path}")
        except BaseException:
            # This path was absent when restoration began.
            path.unlink(missing_ok=True)
            raise
    print(f"Verified: {path}")
PY
```

Compression changes storage only. Experiment results and scientific verdicts
remain those recorded in their original reports.

"""Queue the concrete fixed-action check, then the full same-host comparison."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
receipt = root / "compact_sequence_status.json"
groups = [
    ("results_suffix_compact_diag_r02", "n2fixed0c"),
    ("results_suffix_compact_r01",
     "ar16,n1off,n4horizon,n4horizonc,n4horizonc,n4horizon,n1off,ar16"),
]

def status(**fields):
    receipt.write_text(json.dumps(dict(unix_s=time.time(), **fields), indent=2) + "\n")

for output, arms in groups:
    status(status="QUEUED_OR_RUNNING", group=output, arms=arms)
    command = [sys.executable, str(root / "launch_suffix_compact.py"),
               "--static-cap", "16", "--wait-for-lock",
               "--arms", arms, "--output-name", output]
    with (root / (output + ".launcher.log")).open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status(status="FAILED", group=output, returncode=result.returncode)
        raise SystemExit(result.returncode)
    if "diag" in output:
        trace = json.loads((root / output / "00_n2fixed0c/compaction_trace.json").read_text())
        measured = [row for row in trace["steps"] if row.get("phase") == "measurement"]
        applied = [row for row in measured if row.get("compact_applied")]
        if not applied or not all(row.get("logits_compacted") for row in applied):
            status(status="DIAGNOSTIC_DID_NOT_EXECUTE_COMPACTION", group=output)
            raise SystemExit(65)
        status(status="DIAGNOSTIC_COMPLETE", group=output, compact_steps=len(applied),
               note="Checks actual execution only; not output equivalence or a performance claim.")
status(status="COMPLETE", groups=[name for name, _ in groups])

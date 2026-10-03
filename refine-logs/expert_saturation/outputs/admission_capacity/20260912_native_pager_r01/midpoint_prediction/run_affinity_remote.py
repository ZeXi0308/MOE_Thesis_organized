"""Launch only when explicitly scheduled on the existing authorized GPU host.

Reuses the completed affinity campaign, its unchanged WiSP source and telemetry.
Only a fresh copy gains chunk16 and a prestate check. No existing process is
stopped. Pass --config for the frozen local affinity_config.json uploaded beside
this driver; first-action payload remains unmeasured with trace disabled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("source anchor changed: " + old[:80])
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("affinity_config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["remote_root"])
    source, out = root / config["remote_reference"], root / config["remote_output"]
    prior = json.loads((source / "execution.json").read_text())
    if (prior["status"] != "COMPLETED" or len(prior["cells"]) != 4
            or any(r["status"] != "COMPLETED" or r["exit_code"] for r in prior["cells"])):
        raise RuntimeError("reference affinity campaign is not completely finished")
    for name, expected in config["source_sha256"].items():
        if digest((source / name).read_bytes()) != expected:
            raise RuntimeError("reference source changed: " + name)
    for name, calibration in config["calibration"].items():
        if digest((source / name / "result.json").read_bytes()) != calibration["raw_sha256"]:
            raise RuntimeError("calibration raw changed: " + name)
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        text=True).strip()
    if occupied:
        raise RuntimeError("GPU occupied; no cell started: " + occupied)
    out.mkdir(exist_ok=False)
    (out / "affinity_config.json").write_bytes(args.config.read_bytes())
    for name in ("run_wisp_olmoe_probe.py", "wisp_paging_trace.py", "workload.json"):
        (out / name).write_bytes((source / name).read_bytes())
    probe = replace_once((source / "run_wisp_injection_probe.py").read_text(),
                         "choices=(8, 32)", "choices=(8, 16, 32)")
    compile(probe, "chunk16_injection_probe.py", "exec")
    (out / "run_wisp_injection_probe.py").write_text(probe)
    protocol = json.loads((source / "protocol.json").read_text())
    protocol.update(cells=config["cells"], reason=config["hypothesis"],
                    holdout_scope=config["holdout_scope"], trace=False,
                    frozen_predictions=config["frozen_predictions"])
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    # Preserve the reference's taskset/NUMA observation and 100ms NVML loop.
    campaign = replace_once((source / "run_campaign.py").read_text(),
                            "import json\n", "import json\nimport hashlib\n")
    anchor = "        if code or row['status'] != 'COMPLETED':"
    check = '''        if row['status'] == 'COMPLETED':
            action = result['action']
            prestate = dict(tokens=action['old_output_tokens'], scheduler=action['before_scheduler'],
                            pager=[w['pager_execution_state'] for w in action['before_worker']])
            actual = hashlib.sha256(json.dumps(prestate, sort_keys=True).encode()).hexdigest()
            frozen = json.loads((base / 'affinity_config.json').read_text())
            row['prestate_sha256'] = actual
            if actual != frozen['preaction_sha256']:
                row.update(status='INVALID_PRESTATE', error='preaction state differs from calibration')
            save()
'''
    campaign = replace_once(campaign, anchor, check + anchor)
    compile(campaign, "chunk16_affinity_campaign.py", "exec")
    (out / "run_campaign.py").write_text(campaign)
    print(json.dumps({"status": "STARTING", "output": str(out)}), flush=True)
    with (out / "campaign.log").open("x") as log:
        code = subprocess.run([str(root / "venv/bin/python"), str(out / "run_campaign.py")],
                              stdout=log, stderr=subprocess.STDOUT).returncode
    execution = json.loads((out / "execution.json").read_text())
    print(json.dumps(execution), flush=True)
    if code or execution["status"] != "COMPLETED":
        raise RuntimeError("midpoint stopped; all outputs retained in " + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

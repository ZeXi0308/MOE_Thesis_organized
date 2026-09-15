"""Code-only preparation entry point; run on the authorized host when scheduled.

Upload this driver, config.json and phase_prefill_policy.py only. The workload
is reused on that host. No network, download, retry or process termination is
performed here. The unchanged affinity observer runs identically in both modes.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("source anchor changed: " + old[:80])
    return text.replace(old, new, 1)


def check_installed(root, expected):
    """Read installed Python source without importing vLLM or initializing CUDA."""
    found = {}
    for relative, spec in expected.items():
        matches = list((root / "venv/lib").glob("python*/site-packages/" + relative))
        if len(matches) != 1:
            raise RuntimeError("expected exactly one installed source: " + relative)
        path = matches[0]
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == spec["class"])
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        seen = {name: [] for name in spec["initial_values"]}
        for node in ast.walk(init):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "self" and target.attr in seen):
                    seen[target.attr].append(ast.dump(node.value))
        for name, value in spec["initial_values"].items():
            if ast.dump(ast.parse(repr(value), mode="eval").body) not in seen[name]:
                raise RuntimeError("installed default differs: " + relative + ":" + name)
        if spec["class"] == "Scheduler":
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
            if ([a.arg for a in method.args.args] != ["self"] or method.args.vararg
                    or method.args.kwarg or method.args.kwonlyargs):
                raise RuntimeError("unsupported Scheduler.schedule signature")
        found[str(path)] = dict(sha256=digest(path.read_bytes()), checked=spec)
    return found


def build_sources(source, config):
    probe = replace_once((source / "run_wisp_injection_probe.py").read_text(),
                         "from wisp_paging_trace import TRACE\n",
                         "from wisp_paging_trace import TRACE\nfrom phase_prefill_policy import install as install_phase_policy\n")
    probe = replace_once(probe, "    args = p.parse_args()\n", '''    p.add_argument("--policy", choices=("static8", "phase8"), required=True)
    args = p.parse_args()
    require(args.chunk == 8 and args.new_prompt_length == 128 and not args.no_new and not args.trace,
            "Phase baseline requires chunk8/long128/injection/trace=false")
''')
    probe = replace_once(probe, "        scheduler.schedule = schedule\n", '''        scheduler.schedule = schedule
        install_phase_policy(scheduler, vllm_config=engine.vllm_config, mode=args.policy,
                             is_enabled=lambda: "action" in result,
                             current_call=lambda: result["steps"][-1])
''')
    campaign = replace_once((source / "run_campaign.py").read_text(),
                            "import json\n", "import json\nimport hashlib\n")
    campaign = replace_once(campaign, "str(cell['new_prompt_length'])]",
                            "str(cell['new_prompt_length']), '--policy', cell['policy']]")
    campaign = replace_once(campaign, "        out = base / cell['cell']\n", '''        manifest = json.loads((base / 'run_manifest.json').read_text())
        actual_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                         for name in manifest['sha256']}
        if actual_hashes != manifest['sha256']:
            raise RuntimeError('Frozen code/config/workload/runtime changed before cell')
        out = base / cell['cell']
''')
    campaign = replace_once(campaign, "        state['cells'].append(row)\n",
                            "        row['source_sha256'] = actual_hashes\n        state['cells'].append(row)\n")
    anchor = "        if code or row['status'] != 'COMPLETED':"
    campaign = replace_once(campaign, anchor, '''        if row['status'] == 'COMPLETED':
            action = result['action']
            prestate = dict(tokens=action['old_output_tokens'], scheduler=action['before_scheduler'],
                            pager=[w['pager_execution_state'] for w in action['before_worker']])
            actual = hashlib.sha256(json.dumps(prestate, sort_keys=True).encode()).hexdigest()
            frozen = json.loads((base / 'config.json').read_text())
            row['prestate_sha256'] = actual
            if actual != frozen['preaction_sha256']:
                row.update(status='INVALID_PRESTATE', error='preaction differs from affinity reference')
            save()
''' + anchor)
    compile(probe, "phase_injection_probe.py", "exec")
    compile(campaign, "phase_campaign.py", "exec")
    return probe, campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--policy-source", type=Path,
                        default=Path(__file__).with_name("phase_prefill_policy.py"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["remote_root"])
    source, out = root / config["remote_reference"], root / config["remote_output"]
    for name, count in config["prerequisite_campaigns"].items():
        prior = json.loads((root / name / "execution.json").read_text())
        if (prior["status"] != "COMPLETED" or len(prior["cells"]) != count
                or any(r["status"] != "COMPLETED" or r["exit_code"] for r in prior["cells"])):
            raise RuntimeError("prerequisite campaign not completely finished: " + name)
    for name, expected in config["source_sha256"].items():
        if digest((source / name).read_bytes()) != expected:
            raise RuntimeError("reference source changed: " + name)
    for name, expected in config["reference_result_sha256"].items():
        if digest((source / name / "result.json").read_bytes()) != expected:
            raise RuntimeError("reference result changed: " + name)
    policy = args.policy_source.read_bytes()
    if digest(policy) != config["policy_sha256"]:
        raise RuntimeError("phase policy source changed")
    compile(policy, "phase_prefill_policy.py", "exec")
    installed = check_installed(root, config["installed_source_checks"])
    probe, campaign = build_sources(source, config)
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip()
    if occupied:
        raise RuntimeError("GPU occupied; no cell started: " + occupied)
    out.mkdir(exist_ok=False)
    (out / "config.json").write_bytes(args.config.read_bytes())
    (out / "phase_prefill_policy.py").write_bytes(policy)
    for name in ("run_wisp_olmoe_probe.py", "wisp_paging_trace.py", "workload.json"):
        (out / name).write_bytes((source / name).read_bytes())
    (out / "run_wisp_injection_probe.py").write_text(probe)
    (out / "run_campaign.py").write_text(campaign)
    protocol = json.loads((source / "protocol.json").read_text())
    protocol.update(cells=config["cells"], reason=config["question"], trace=False,
                    phase_policy=config["policy"], holdout_scope=config["holdout_scope"])
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    hashes = {str(p): digest(p.read_bytes()) for p in out.iterdir() if p.is_file()}
    hashes.update({p: row["sha256"] for p, row in installed.items()})
    (out / "run_manifest.json").write_text(json.dumps(dict(sha256=hashes, installed=installed,
        launch_driver_sha256=digest(Path(__file__).read_bytes())), indent=2) + "\n")
    print(json.dumps({"status": "STARTING", "output": str(out)}), flush=True)
    with (out / "campaign.log").open("x") as log:
        code = subprocess.run([str(root / "venv/bin/python"), str(out / "run_campaign.py")],
                              stdout=log, stderr=subprocess.STDOUT).returncode
    execution = json.loads((out / "execution.json").read_text())
    print(json.dumps(execution), flush=True)
    if code or execution["status"] != "COMPLETED":
        raise RuntimeError("phase baseline stopped; outputs retained in " + str(out))


if __name__ == "__main__":
    main()

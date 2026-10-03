#!/usr/bin/env python3
"""One locked, one-action recovery-admission mechanism pilot."""
import signal
import sys
import run_native_bidkv_full_running_triplet_r01 as prior

group, base, current = prior.group, prior.base, prior.current
base.PACKAGE_NAME = "candidate_native_oldest_admission_r02"
base.PACKAGE_SHA256 = "f29d62e904355cc1c2261ffe7352a791ec69a2ec772fb37f8afdfe0de8348928"
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
current.prior.RULES = ("tail",) * 3
current.inherited.MODES = ("off",) * 3
current.GUARDS = ("off",) * 3
prior.FULL_MODES = ("off",) * 3
group.ARMS = ("native", "queue_only", "queue_fund")
group.BASE = "/root/moe-a-native-oldest-admission-stage-r02-20261002"
group.SESSION = "/root/moe-a-native-oldest-admission-session-r02-20261002"
group.OUTPUTS = tuple("/root/moe-a-native-oldest-admission-output-" + label + "-r02-20261002"
                      for label in group.LABELS)
previous_private_env = base.private_model_env


def private_model_env(plan, session_dir):
    env = previous_private_env(plan, session_dir)
    found = [i for i, arm in enumerate(group.ARMS)
             if (session_dir / f"cell-{i:02d}-{arm}").exists()]
    index = found[-1] if found else 0
    env["A_NATIVE_CAPACITY_DEFERRAL"] = "off"
    env["A_NATIVE_OLDEST_ADMISSION"] = group.ARMS[index]
    if found:
        base.write_json(session_dir / f"cell-{index:02d}-{group.ARMS[index]}" / "oldest-admission-mode.json",
                        dict(mode=group.ARMS[index], selection_budget=1,
                             last_output_age_trigger_s=1.0, victim_rule="tail",
                             effective_ordinary_backfill=False, legacy_tracker=False,
                             full_running="off", current_guard="off",
                             self_preempt_continue="off", capacity_deferral="off"))
    return env


base.private_model_env = private_model_env

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print("ABORT_NATIVE_OLDEST_ADMISSION:", error, file=sys.stderr)
        raise SystemExit(75)

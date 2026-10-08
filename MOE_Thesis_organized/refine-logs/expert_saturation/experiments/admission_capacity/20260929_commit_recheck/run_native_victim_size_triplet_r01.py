#!/usr/bin/env python3
"""One locked three-arm native victim-size pilot; freeze before running."""
import signal
import sys
import run_native_bidkv_full_running_triplet_r01 as prior

group, base, current = prior.group, prior.base, prior.current
base.PACKAGE_NAME = "candidate_native_victim_size_r01"
base.PACKAGE_SHA256 = "8c9aee3e9f30257fb2feaedd317b54b56cb2571f231206e48ab810a09b5ca739"
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
current.prior.RULES = ("tail", "min_held_other", "max_held_other")
current.inherited.MODES = ("off", "off", "off")
current.GUARDS = ("off", "off", "off")
prior.FULL_MODES = ("off", "off", "off")
group.ARMS = ("tail", "min_held_other", "max_held_other")
group.BASE = "/root/moe-a-native-victim-size-stage-r01-20261002"
group.SESSION = "/root/moe-a-native-victim-size-session-r01-20261002"
group.OUTPUTS = tuple("/root/moe-a-native-victim-size-output-" + label + "-r01-20261002"
                      for label in group.LABELS)
previous_private_env = base.private_model_env

def private_model_env(plan, session_dir):
    env = previous_private_env(plan, session_dir)
    env["A_NATIVE_CAPACITY_DEFERRAL"] = "off"
    return env

base.private_model_env = private_model_env

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print("ABORT_NATIVE_VICTIM_SIZE:", error, file=sys.stderr)
        raise SystemExit(75)


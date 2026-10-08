#!/usr/bin/env python3
"""One bounded donor-comparator pilot using the existing shared-lock launcher."""
from pathlib import Path
import C_NATIVE_RECOMPUTE_PILOT_LAUNCHER_V3 as pilot

HELPER_SHA = "3c64e4866a1c3e28082158bf38b64f6a1de4517d1a33a042fb7a2b9d83322b42"
SOURCE_SHA = {
    "C_NATIVE_TEMPORAL_CAP_DONOR_CELL_V1.py": "b50e9dd37d8e58a3d763fea70f743325d547945e4e11100972bf5b1f608a878e",
    "C_NATIVE_TEMPORAL_CAP_DONOR_ADMISSION_V1.py": "6e4e109db013d0cbc963db0ba0f93328961cba912275ec38b00c6b9a84d740af",
    "C_NATIVE_RETIREMENT_FRESH_CONTRACT_V1.py": "6896c987f30106deffed0bfce22b98728cf4d78871e41ba45e06b0c1383d72a8",
    "C_NATIVE_MAX_BOUND_ADMISSION.py": "b6a601d1edc05804bee70d7f6ecdb2f00e1dd017b2280163df2db88250fb50ee",
    "C_NATIVE_RETIREMENT_ADMISSION_V1.py": "261b8f2697042f75130203019e8722c424cca3e71d82ac309c78f29ec1bc8eb6",
}
INPUT_SHA = {
    "config.json": "634e7715618879775daf312fc98d77c1b25f4cd8e97007c101604ded317738a3",
    "workload.json": "9576395c9540c71be86dd182df03ab6d39cf1c961a6a24dc851d40991cf0a985",
    "INPUT_STATS.json": "797ccd408a60ee01bed2ac370f51f7145fa4f9ea3bcaecc5a1a4535afd034ac1",
}


def main():
    if pilot.sha(Path(pilot.__file__)) != HELPER_SHA:
        raise RuntimeError("shared-lock child-lifecycle helper changed")
    for name, digest in SOURCE_SHA.items():
        if pilot.sha(pilot.BASE / name) != digest:
            raise RuntimeError(f"frozen source changed: {name}")
    for name, digest in INPUT_SHA.items():
        if pilot.sha(pilot.BASE / "20261001_c_retirement_fresh_inputs_v1" / name) != digest:
            raise RuntimeError(f"fixed viewed input changed: {name}")
    pilot.GPU_UUID = "GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36"
    pilot.LOCK_INODE = "2304:29005388732"
    pilot.CELL_SOURCE = pilot.BASE / "C_NATIVE_TEMPORAL_CAP_DONOR_CELL_V1.py"
    pilot.CELL_SOURCE_SHA = SOURCE_SHA[pilot.CELL_SOURCE.name]
    pilot.ROOT = pilot.BASE / "c-native-temporal-cap-donor-pilot-v1"
    pilot.CELL = pilot.ROOT / "native_temporal_donor_1"
    # The cell independently requires a drained donor ledger before COMPLETE.
    # V3 covers flock, owned-child cleanup, runtime, all-request completion,
    # no-offload memory and post-exit GPU release. No extra queue or retry loop.
    return pilot.main()


if __name__ == "__main__":
    raise SystemExit(main())

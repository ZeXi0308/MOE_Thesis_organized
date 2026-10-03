# Fair LTR runner successor candidate — CPU only

Base: accepted `20260915_natural_ltr_style_component_r02`, HEAD
`76d6d888de42081c63cd440a8a67623161d8181f`, interface
`RECOVERY_COMMIT_V1`. This directory is a reviewable overlay, **not** an
accepted or executable experiment package.

Apply `run_ltr_style.patch` to a fresh copy of r02 `pkg/run_ltr_style.py`, then
place the two byte-for-byte root `ltr_fair_native.py` and `ltr_fair_policy.py`
files beside it. `run_ltr_style.py` here is the exact patched result. The r02
`controller.py`, `run.sh`, manifest, input files, warmups and all backend files
were read only. Before any successor can launch, the GPU owner must issue new
documentation and a manifest including the new fair files and changed runner,
resolve the old host-specific UUID/Python/cache/lock paths, and separately
accept its native lifecycle scope. The inherited r02 `cpu_checks.json` and
`check_lifecycle.py` qualify only the old adapter.

The runner changes five seams: policy identity to `ltr_fair_selected`, adapter
import, source hash list, install call signature, and a checked compatibility
receipt translating the fair adapter's `threshold`/`quantum` to the existing
selected-save runner fields. It still checks that the scheduler schedule and
connector selected-store overrides were actually installed. The install call
stays after all three warmups, native offload drain, connector cache reset and
empty-admission check, and before measured capture. G64 input hashes, fixed
model/seed/EOS/cap, 4096 usable blocks plus null block, 16 GiB host, pinned
vLLM source checks, capture, physical qualification and drain code are unchanged.

The fair adapter is a different candidate from r02 `ltr_style_native.py`: it
does not reserve KV for one active target or force target-only waiting, and it
records native peer preemption. A pass for r02 does not qualify this behavior.
The native store/load/flush, first new output and positive-call quantum after
output remain GPU unrun for this successor. With 31 running peers, token budget
can allow only partial target recompute despite enough KV and a free slot;
qualification must record this outcome rather than assume one-call output.

CPU check from `/private/tmp/moe-research-c-20260929`:

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.13 -m unittest -v \
  research_c/ltr_fair_successor/test_runner_wiring.py
```

Result: one targeted runner seam/order test passed. The patch also applied
exactly to r02 in a disposable temporary directory, and all four Python files
compiled without imports or GPU initialization. The existing 27 fair policy,
adapter and comparison tests passed separately; they do not qualify a new GPU
package.

## A integration receipt

Root copied this overlay into the main research workspace and repeated the
targeted wiring test: 1/1 passed. Root independently applied the patch to a
fresh accepted r02 runner in a temporary directory and obtained the exact
`run_ltr_style.py` bytes in this overlay. The fair adapter/policy copies match
the shared CPU source SHA-256 values `a1ceccd3…a5c64b6` and
`1150e69b…f433e3c`. This is still CPU-only integration. The controller,
manifest, machine paths, native lifecycle and resource authorization have not
been re-frozen or executed.

# KV comparison on the newly supplied instance

2026-09-12. User supplied existing SSH instance port 23478. Local HEAD 7059fc98e0d98a127ff4edda86d44f7f634d26f4; pre-existing dirty expert_union files are excluded. Public branch checked live remains d65416c5cd09eb30aa6e46875339b16ebf7f5901.

Exact scientific execution.tar.gz reused from 20260910_kv_budget_r01: e9649d8314d593d9c0b5a879483e471d8ddda7bd57482aa3acf9ead6d9c2880b. No scientific source/config changes. Only driver endpoint, new output root and transient SSH socket differ.

Question: does increased actual KV pool at cap32 remove recovery pauses and improve full-request outcomes? Order .95/.90/.90/.95; 32 WikiText prompts, 3072 input, 1024 output, 50ms arrivals, budget1024, native preemption in both arms, three retained warmups per cell. Budget95 qualification requires actual usable blocks >=32*ceil(4096/block_size), a sufficient full-residency condition, not necessary for native completion. Stop if first .95 initialization/qualification fails; retain failures and do not retune. No expert route capture.

Authority read: docs/current/README.md, docs/ideas/README.md, expert_saturation/EXPERIMENT_TRACKER.md, DECODE_CAP_BRANCH_GATE.md, admission_capacity/README.md, old KV DECISIONS and latest feasibility report. Historical mechanism verdicts inherited; user explicitly selects this ordinary-KV question without requiring unrelated conformance work.

New GPU UUID GPU-389be666-aeaa-c602-1504-85bf1dd3ac9f; 32607MiB, idle at inspection. vLLM0.26.0, Torch2.11.0, Transformers5.15.1. Host reports 754.54GiB total but cgroup memory.max=98784247808 bytes (92GiB), so host total is not job allocation. Available data disk about29.4GiB: insufficient to assume a new Qwen3 BF16 snapshot fits alongside this environment.

Claim ceiling: one-model native in-process request measurement; configuration intervention changes reserved memory budget, not same-budget scheduler improvement. No paid resource provisioning or new dependencies.

## Live execution discovered after upload

User explicitly authorized transferring repository code/config/request inputs. Transfer of the original bundle then succeeded. Unpacking with exist_ok=False found an existing /root/autodl-tmp/moe-kv-budget-20260912-r01. Read-only inspection showed another concurrent workspace task was already running the same four cells with identical bundle hash. We did not launch a second campaign or alter that run. Canonical local readback is ../20260912_kv_budget_r01 (relative to campaign parent), tracked by its execution.json. Our unused duplicate driver/bundle were removed; original immutable source remains in 20260910_kv_budget_r01.

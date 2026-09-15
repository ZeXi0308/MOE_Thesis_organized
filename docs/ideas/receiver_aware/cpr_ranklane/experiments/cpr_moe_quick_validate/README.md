# CPR-MoE RTX 5090 quick validation

This directory contains two necessary-condition gates for the conditional
CPR-MoE direction. It does **not** implement CPR-MoE and cannot validate the
optimized EP return-path hypothesis.

## Experiments

1. `quality`: re-analyzes the existing paired, held-out OLMoE/LLM-jp documents.
   It tests only whether the lowest gate rank is consistently less quality
   sensitive than rank 1 at equal byte budget.
2. `codec`: measures a connected same-stream `pack -> unpack` path for per-row
   INT8/INT4 on a CUDA GPU. Only INT4 is a decision gate because the paired
   quality evidence is INT4; INT8 is characterization only. The transfer saving
   is a zero-start analytical byte bound. There is no H2D proxy and no claim of
   NCCL, EP, overlap, or fusion.

Passing both experiments means only `NOT_FALSIFIED_SINGLE_GPU`; the 8×A100
optimized return-path precedence-DAG Gate 0 remains mandatory.

## Environment

```bash
cd /Users/leandrozhao/Desktop/毕设论文资料
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==2.0.2 pandas==2.3.3
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import triton; print(triton.__version__)"
```

The CUDA 12.8 PyTorch wheel brings its compatible Triton dependency on Linux.
The frozen environment expects Torch 2.8.0, NumPy 2.0.2 and Pandas 2.3.3. On
the RTX 5090 host, do not silently fall back to CPU for the codec experiment.

## Quality provenance prerequisite

The older quality CSVs did not record the producer/dependency hashes or exact
rounding contract. They remain useful numeric evidence, but the formal matched
INT4 gate refuses them. Run the frozen producer once per model into new output
directories; it now emits `quantization_provenance.json` during the same
forward run:

```bash
python docs/ideas/A_rank_tail_fp8/experiments/run_idea_a_rank_lut_gpu_rigorous_verify.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --model-key olmoe \
  --samples 128 --offset 600 --seq-len 128 --dtype bfloat16 \
  --seed 20260720 --n-bootstrap 5000 --offline \
  --output-dir docs/ideas/A_rank_tail_fp8/outputs/cpr_quality_refresh_2026-07-25_olmoe

python docs/ideas/A_rank_tail_fp8/experiments/run_idea_a_rank_lut_gpu_rigorous_verify.py \
  --model llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M \
  --model-revision 1d5983076dfc67aee4a77ec06a27027f5bab6055 \
  --model-key llmjp \
  --samples 128 --offset 600 --seq-len 128 --dtype bfloat16 \
  --seed 20260720 --n-bootstrap 5000 --offline \
  --output-dir docs/ideas/A_rank_tail_fp8/outputs/cpr_quality_refresh_2026-07-25_llmjp
```

Do not manufacture sidecars for old CSVs. The quick-validation runner checks
the CSV, producer, `fake_quant.py`, `policies.py`, and `capture_moe.py` hashes.

## Commands

Validate configuration and local inputs without running an experiment:

```bash
python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/run_experiment.py \
  --config docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/configs/quick_validate.json \
  --experiment all \
  --output-dir docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/validate_20260725 \
  --validate-only
```

CPU-compatible quality smoke:

```bash
python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/run_experiment.py \
  --config docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/configs/quick_validate.json \
  --experiment quality \
  --output-dir docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/smoke_quality_20260725 \
  --smoke
python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/analyze.py \
  --input-dir docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/smoke_quality_20260725
```

Formal quality necessary gate:

```bash
python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/run_experiment.py \
  --config docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/configs/quick_validate.json \
  --experiment quality \
  --output-dir docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/quality_seed20260725
```

RTX 5090 codec smoke and formal run:

```bash
python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/run_experiment.py \
  --config docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/configs/quick_validate.json \
  --experiment codec \
  --output-dir docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/smoke_codec_20260725 \
  --smoke

python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/run_experiment.py \
  --config docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/configs/quick_validate.json \
  --experiment all \
  --seed 20260725 \
  --output-dir docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/formal_seed20260725
```

The formal path deliberately uses `--experiment all`, so quality and INT4
codec decisions share one immutable run directory and manifest.

Run all three pre-registered seeds:

```bash
for CPR_SEED in 20260725 20260726 20260727; do
  python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/run_experiment.py \
    --config docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/configs/quick_validate.json \
    --experiment all \
    --seed "$CPR_SEED" \
    --output-dir "docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/formal_seed${CPR_SEED}"
done
```

Analyze one run:

```bash
python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/analyze.py \
  --input-dir docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/formal_seed20260725
```

Aggregate the frozen multi-seed protocol. It fails closed unless all and only
the configured seeds are present, all used the same config hash, all ran
`--experiment all`, and every manifest is `COMPLETE`:

```bash
python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/analyze.py \
  --input-dirs \
    docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/formal_seed20260725 \
    docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/formal_seed20260726 \
    docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/formal_seed20260727 \
  --output-dir docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/formal_aggregate_20260725
```

Never reuse an output directory after a failure. The failed directory retains
`run_manifest.json` with `status=FAILED`; diagnose it, then use a new directory.

## Expected files

- `resolved_plan.json`
- `quality_paired_raw.csv`, `quality_summary.csv`, `quality_decision.json`
- `codec_raw_samples.csv`, `codec_summary.csv`, `codec_decision.json`
- `run_manifest.json`
- after analysis: `analysis.json`, `report.md`

No result values are pre-populated.

## Tests

```bash
cd docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate
python -m unittest -v test_quick_validate.py
python -m unittest -v test_codec_kernels_gpu.py
```

The second command is mandatory on the RTX 5090 before a formal run. It checks
packed bytes, scales, and reconstructed BF16 values against a PyTorch reference
for zeros, extremes, ties-to-even, signed INT4 nibbles, and both configured
hidden sizes. It is skipped, not passed, when CUDA/Triton is unavailable.

## Cleanup

Inspect targets first, then remove only an explicit generated directory:

```bash
find docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results -mindepth 1 -maxdepth 1 -type d -print
rm -r docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/smoke_quality_20260725
```

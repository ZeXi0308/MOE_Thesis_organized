# Energy-SLO FP8-Compute Quality Confound Gate (llmjp, scope=all)

- documents: 24 (dataset=wikitext103_docs, offset=0)
- reference corpus PPL: 13.4852
- candidate (FP8-compute) corpus PPL: 13.6195
- mean token KL: 0.009054  (95% CI [0.008240, 0.010351])
- acceptable KL threshold (this script's judgment call): 0.05
- known-bad communication-INT4 KL anchor (qualitative only, NOT directly comparable): 0.257
- VERDICT: GO

Evidence boundary: single-GPU real FP8 tensor-core compute on expert FFN matmuls only (attention/router/LM-head untouched); per-tensor absmax scaling; routing divergence at deeper layers is a real, expected downstream effect and is included in the KL number.
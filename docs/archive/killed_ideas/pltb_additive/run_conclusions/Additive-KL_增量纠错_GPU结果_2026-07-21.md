# Additive-KL Modeling Audit (Corrected 2026-07-21) — GPU Result

Hardware: RTX 5090 (AutoDL). Model: local OLMoE snapshot. Dataset: `wikitext2_docs` cal offset0 n16 / test offset20 n32.

## Verdict

**WITHDRAW_ADDITIVITY_FALSIFIED**

| predictor | ratio | 95% CI | CI contains 1? |
|---|---:|---|---|
| locked_incremental (CORRECT) | **1.076** | [0.983, 1.165] | **Yes** |
| free_incremental (CORRECT) | 0.899 | [0.682, 1.130] | Yes |
| locked_additive DEPRECATED | 3.772 | [3.669, 3.869] | No |

Prior **3.77×** claim reproduced exactly under the deprecated double-count formula and is an accounting artifact, not evidence of residual-stream non-additivity.

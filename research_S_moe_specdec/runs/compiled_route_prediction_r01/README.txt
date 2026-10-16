Reproduce: python3 research_S_moe_specdec/runs/compiled_route_prediction_r01/analyze.py

Using P instead of S removes much of the omitted physical expert-set work, especially at B=1. It supports P as a necessary correction for a physical-row occupancy approximation, but does not establish that padding rows obey the AR marginals or within-step independence assumptions. At B=1 q+P still has MAE 3.132707 and positive bias 2.561535 experts (MAPE 12.863844%). The recorded padding histograms recur exactly over consecutive padded steps and their edge marginals differ from AR; temporal recurrence alone does not determine within-step independence. This is descriptive evidence about route sets, not a CPU/GPU cache-hit observation, timing decomposition, or service benefit.

B | q+S -> valid D MAE | q+P -> physical D MAE | q+S -> physical D MAE | q+P bias | q+P MAPE (%)
1 | 0.565913 | 3.132707 | 17.674742 | 2.561535 | 12.863844
8 | 3.437925 | 2.898445 | 6.307189 | 0.394678 | 6.839818
32 | 1.945560 | 1.463901 | 1.818043 | 0.292922 | 2.536831

B1 padding diagnostic: 1569 padded steps per layer, four padding rows per step. Each of 16 layers has 134 distinct padding histograms/support sets; its modal histogram accounts for 46.653920% of padded steps. All 1370 adjacent padded-step pairs within waves per layer have identical padding histograms (support Jaccard=1). For the valid row on those same pairs, equal-support fraction ranges from 0 to 1.094891%, with mean Jaccard by layer 0.177427 to 0.355602. Padding versus AR route-edge marginal total variation ranges from 0.404341 to 0.572274. These are histogram/support observations, not a reconstruction of padding token routes.

Fixed fit partition: gamma=0, nominal B={1,8,32}, all 64 requests and all target decode rows. q_e=expert inclusion count/token count, pooled by layer. Each B contributes 1984 decode tokens per layer, total 5952; no fitting to gamma=4.

Fixed test partition: gamma=4, nominal B={1,8,32}, all target decode layer-steps. SHA-disjoint files but the same 64 prompt subset: held-out executed gamma conditions, not an independent workload validation.

Comparison predictions all use the identical fitted q_e and formula sum_e[1-(1-q_e)^n], with n=S or P. P is model-forward row count, excluding expert GEMM tile padding.

Metrics weight decode layer-steps equally; small B therefore contributes more steps to the pooled summary. Per-B metrics are the primary comparison. Terminal shrinking batches and every executed valid verification row are retained.

Steps, layers and waves are correlated, with one instrumented condition run; no independent confidence interval or significance claim is made.

B1 padding repetition uses exact recorded padding histograms/support sets, not token-level padding routes. Adjacent means consecutive integer steps within the same wave, both padded; no across-wave or across-gap pair is included.

Padding route-edge marginal distance is a descriptive comparison to AR, not a fitted alternate predictor. Repeated support/histogram does not establish identical ordered token routes or their mechanism.

Instrumented physical working-set prediction only, not performance or measured HBM traffic; route-return/native histogram overhead remains active. Main eager F1 data and analysis are unchanged.

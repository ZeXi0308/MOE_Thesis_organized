# Routing Predictability P0-B: Lookahead-Distance Decay Curve

How much does top1_transition's advantage over freq_baseline decay as lookahead k increases?

| model | k | mean_accuracy_diff_pp | ci_low_pp | ci_high_pp | predictor_mean_accuracy | baseline_mean_accuracy | n_docs | p_value_holm | passes_5pp_gate |
|---|---|---|---|---|---|---|---|---|---|
| llmjp | 1 | 13.8529 | 13.3286 | 14.3993 | 0.2624 | 0.1238 | 60 | 0.0000 | True |
| llmjp | 2 | 13.6477 | 13.1296 | 14.1566 | 0.2544 | 0.1179 | 60 | 0.0000 | True |
| llmjp | 3 | 11.9276 | 11.4768 | 12.4119 | 0.2394 | 0.1201 | 60 | 0.0000 | True |
| llmjp | 4 | 12.3351 | 11.8576 | 12.8402 | 0.2473 | 0.1240 | 60 | 0.0000 | True |
| llmjp | 6 | 10.1510 | 9.4881 | 10.8177 | 0.2286 | 0.1271 | 60 | 0.0000 | True |
| llmjp | 8 | 8.1429 | 7.4689 | 8.8200 | 0.2188 | 0.1373 | 60 | 0.0000 | True |
| olmoe | 1 | 19.0191 | 18.3787 | 19.6233 | 0.2738 | 0.0836 | 45 | 0.0000 | True |
| olmoe | 2 | 18.5268 | 18.0221 | 19.0290 | 0.2710 | 0.0857 | 45 | 0.0000 | True |
| olmoe | 3 | 17.8586 | 17.3502 | 18.4008 | 0.2655 | 0.0869 | 45 | 0.0000 | True |
| olmoe | 4 | 16.7520 | 16.0843 | 17.4494 | 0.2566 | 0.0891 | 45 | 0.0000 | True |
| olmoe | 6 | 14.5486 | 13.8776 | 15.1843 | 0.2317 | 0.0862 | 45 | 0.0000 | True |
| olmoe | 8 | 14.1102 | 13.3549 | 14.7819 | 0.2314 | 0.0903 | 45 | 0.0000 | True |
# Routing Predictability P0: Gate Results

Predictor accuracy - baseline accuracy (percentage points), paired bootstrap over TEST documents, Holm-corrected across all cells.
Practical bar: mean_diff_pp >= 5.0 AND ci_low_pp > 0 AND p_value_holm < 0.05.

| model | bucket | predictor | baseline | mean_accuracy_diff_pp | ci_low_pp | ci_high_pp | predictor_mean_accuracy | baseline_mean_accuracy | n_docs | p_value_holm | passes_5pp_gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llmjp | all | neighbor_same_layer | freq_baseline_same_layer | 9.7185 | 9.2079 | 10.2594 | 0.2366 | 0.1394 | 60 | 0.0000 | True |
| llmjp | all | top1_transition | freq_baseline | 13.8529 | 13.3286 | 14.3993 | 0.2624 | 0.1238 | 60 | 0.0000 | True |
| llmjp | all | topk_transition | freq_baseline | 1.6172 | 1.5165 | 1.7201 | 0.1400 | 0.1238 | 60 | 0.0000 | False |
| llmjp | early | neighbor_same_layer | freq_baseline_same_layer | 8.6315 | 8.0988 | 9.2273 | 0.2273 | 0.1410 | 60 | 0.0000 | True |
| llmjp | early | top1_transition | freq_baseline | 14.7412 | 14.1821 | 15.2809 | 0.2592 | 0.1118 | 60 | 0.0000 | True |
| llmjp | early | topk_transition | freq_baseline | 1.9735 | 1.8375 | 2.1143 | 0.1315 | 0.1118 | 60 | 0.0000 | False |
| llmjp | late | neighbor_same_layer | freq_baseline_same_layer | 10.8056 | 9.9214 | 11.6708 | 0.2459 | 0.1379 | 60 | 0.0000 | True |
| llmjp | late | top1_transition | freq_baseline | 12.8376 | 11.8833 | 13.7630 | 0.2660 | 0.1377 | 60 | 0.0000 | True |
| llmjp | late | topk_transition | freq_baseline | 1.2100 | 1.1291 | 1.2891 | 0.1498 | 0.1377 | 60 | 0.0000 | False |
| olmoe | all | neighbor_same_layer | freq_baseline_same_layer | 13.3971 | 12.5827 | 14.2125 | 0.2156 | 0.0817 | 45 | 0.0000 | True |
| olmoe | all | top1_transition | freq_baseline | 19.0191 | 18.3787 | 19.6233 | 0.2738 | 0.0836 | 45 | 0.0000 | True |
| olmoe | all | topk_transition | freq_baseline | 6.0174 | 5.5978 | 6.4503 | 0.1437 | 0.0836 | 45 | 0.0000 | True |
| olmoe | early | neighbor_same_layer | freq_baseline_same_layer | 13.5098 | 12.5566 | 14.4739 | 0.2078 | 0.0727 | 45 | 0.0000 | True |
| olmoe | early | top1_transition | freq_baseline | 17.7203 | 16.9386 | 18.4212 | 0.2593 | 0.0821 | 45 | 0.0000 | True |
| olmoe | early | topk_transition | freq_baseline | 3.9041 | 3.5970 | 4.2274 | 0.1212 | 0.0821 | 45 | 0.0000 | False |
| olmoe | late | neighbor_same_layer | freq_baseline_same_layer | 13.2843 | 12.2025 | 14.3791 | 0.2235 | 0.0906 | 45 | 0.0000 | True |
| olmoe | late | top1_transition | freq_baseline | 20.5035 | 19.5063 | 21.4139 | 0.2903 | 0.0852 | 45 | 0.0000 | True |
| olmoe | late | topk_transition | freq_baseline | 8.4325 | 7.7182 | 9.1692 | 0.1696 | 0.0852 | 45 | 0.0000 | True |
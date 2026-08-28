# Expert-Precision Persistence & Shadow-Verify Controller (llmjp)

> **SUPERSEDED FOR H2 VERDICT (2026-07-22):** the table below masks an independently collected all-INT4 KL trajectory; it does not execute the mixed-precision policy's KV-state evolution or charge low+high shadow compute. H1 remains descriptive. H2 requires an in-loop rerun.

- documents: 32 (calib=12, test=20), decode_steps=48, fp_scope=all
- escalate threshold tau (calibrated at 0.75 quantile of calib KL): 0.078719

## H1: lag persistence (primary, decisive; GO iff CI_low > 0.2)
 lag  spearman    ci_low  ci_high  n_documents  n_pairs go_no_go
   1  0.203277  0.125370 0.276640           20      940    NO-GO
   2  0.068850 -0.019643 0.172274           20      920    NO-GO
   3  0.139776  0.066783 0.208259           20      900    NO-GO
   4  0.115803  0.048323 0.184128           20      880    NO-GO
   6  0.034071 -0.026799 0.092139           20      840    NO-GO
   8  0.024803 -0.054523 0.098926           20      800    NO-GO

## H2: causal shadow-verify controller simulation (secondary, operational)
 period  threshold_tau  n_documents  always_low_kl  no_escalate_kl  no_escalate_high_frac  reactive_kl  reactive_high_frac  oracle_kl  oracle_high_frac  reactive_reduction_vs_always_low  reactive_reduction_ci_low  reactive_reduction_ci_high  reactive_vs_oracle_kl_ratio go_no_go
      4       0.078719           20      55.875195       42.475097                 0.2500    29.605498            0.403125  23.461825          0.188542                          0.470150                   0.401931                    0.534016                     1.261858    NO-GO
      8       0.078719           20      55.875195       49.299119                 0.1250    39.437865            0.314583  23.461825          0.188542                          0.294179                   0.198547                    0.401997                     1.680938    NO-GO
     16       0.078719           20      55.875195       52.397223                 0.0625    41.824728            0.281250  23.461825          0.188542                          0.251462                   0.159311                    0.339294                     1.782672    NO-GO

## Same-step router-feature diagnostics (exploratory ONLY, not part of any GO/NO-GO)
                               feature  same_step_spearman_vs_kl
           full_route_top1_weight_mean                  0.078328
            full_route_top1_weight_std                  0.091955
      full_route_top1_top2_margin_mean                  0.092397
             full_route_tail_mass_mean                 -0.013285
       full_route_routing_entropy_mean                 -0.057872
             full_route_rank1_hhi_mean                  0.000000
full_route_active_expert_fraction_mean                  0.000000
full_route_same_id_adjacent_layer_rate                 -0.067103

OVERALL: H1 NO-GO (at least one lag) / H2 NO-GO (at least one verify period).

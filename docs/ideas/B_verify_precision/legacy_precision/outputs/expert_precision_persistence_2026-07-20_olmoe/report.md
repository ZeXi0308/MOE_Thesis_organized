# Expert-Precision Persistence & Shadow-Verify Controller (olmoe)

> **SUPERSEDED FOR H2 VERDICT (2026-07-22):** the table below masks an independently collected all-INT4 KL trajectory; it does not execute the mixed-precision policy's KV-state evolution or charge low+high shadow compute. H1 remains descriptive. H2 `GO` is invalid and requires an in-loop rerun.

- documents: 32 (calib=12, test=20), decode_steps=48, fp_scope=all
- escalate threshold tau (calibrated at 0.75 quantile of calib KL): 0.066958

## H1: lag persistence (primary, decisive; GO iff CI_low > 0.2)
 lag  spearman    ci_low  ci_high  n_documents  n_pairs go_no_go
   1  0.180405  0.121711 0.227556           20      940    NO-GO
   2  0.103191  0.042270 0.159482           20      920    NO-GO
   3  0.099512  0.030521 0.163724           20      900    NO-GO
   4  0.098623  0.039641 0.141689           20      880    NO-GO
   6  0.022381 -0.049048 0.086991           20      840    NO-GO
   8  0.068001  0.012413 0.114873           20      800    NO-GO

## H2: causal shadow-verify controller simulation (secondary, operational)
 period  threshold_tau  n_documents  always_low_kl  no_escalate_kl  no_escalate_high_frac  reactive_kl  reactive_high_frac  oracle_kl  oracle_high_frac  reactive_reduction_vs_always_low  reactive_reduction_ci_low  reactive_reduction_ci_high  reactive_vs_oracle_kl_ratio go_no_go
      4       0.066958           20      57.854028       43.776442                 0.2500    28.881235            0.434375  17.418401          0.251042                          0.500791                   0.421280                    0.566153                     1.658088       GO
      8       0.066958           20      57.854028       50.452892                 0.1250    33.725936            0.358333  17.418401          0.251042                          0.417051                   0.319734                    0.506175                     1.936225    NO-GO
     16       0.066958           20      57.854028       54.628077                 0.0625    35.337006            0.343750  17.418401          0.251042                          0.389204                   0.225269                    0.528815                     2.028717    NO-GO

## Same-step router-feature diagnostics (exploratory ONLY, not part of any GO/NO-GO)
                               feature  same_step_spearman_vs_kl
           full_route_top1_weight_mean                  0.150987
            full_route_top1_weight_std                  0.132280
      full_route_top1_top2_margin_mean                  0.100159
             full_route_tail_mass_mean                  0.095775
       full_route_routing_entropy_mean                 -0.133528
             full_route_rank1_hhi_mean                  0.000000
full_route_active_expert_fraction_mean                  0.000000
full_route_same_id_adjacent_layer_rate                 -0.047018

OVERALL: H1 NO-GO (at least one lag) / H2 GO (at least one verify period).

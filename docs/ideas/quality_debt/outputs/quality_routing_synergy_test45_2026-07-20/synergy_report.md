# Synergy check: routing predictability vs quality-degradation risk (same 45 documents)

n_docs = 45

Spearman(top1_hit_rate, mean_token_kl under fixed_tail4) = -0.1770 (p=0.2449)
Spearman(mean_routing_entropy, mean_token_kl) = -0.0211 (p=0.8907)
Spearman(token_count, mean_token_kl) = nan (p=nan)  [length confound check]
Spearman(top1_hit_rate, mean_routing_entropy) = -0.1344 (p=0.3787)  [sanity: hit_rate and entropy should be strongly anti-correlated]
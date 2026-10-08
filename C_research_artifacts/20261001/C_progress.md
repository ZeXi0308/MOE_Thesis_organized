# C progress — 2026-10-02

Independent C line; active/incomplete, not ready for paper review.

Completed: whole/q1/q2 fixed-grain450 on MultiFieldQA150; q1/q2 mean completion worsened46.05%/17.23% with measured cache-hit loss. Common warmup removed the prior0.7s Triton-JIT step; no fixed-quantum tuning follows. Classical Horn prompt-only exact comparator improves whole only0.0653%, not a GPU-time bound.

New actual QMSum200 whole baseline on primary GPUe4434c32: all200,199EOS/1cap,19454outputtokens, mean completion8.929s/p9515.195s, full-outputRouge-L21.414%. 927steps, peakKV4096,4preempts/273alloc failures. Only2requests have211ms host gaps; others<=37.029ms. First prompt compute347028 exactly matches unique prefix-trie work; total347688 adds660 recovery tokens. No new-method benefit claimed.

Next: exact native allocator first-feasible backfill beyond a blocked WAITING head, same density rank and native128/1024/4096/EOS512. Prior observation shows only one shorter candidate may fit, so expected opportunity is small. V1 failed before formal inference due source-anchor indentation, child/stage/GPU released; immutable failure preserved. V2 anchor repair source816b1ef; spare same-host fallback frozen in15461dc. Primary twice and spare once returned shared-lock busy at extraction. Both V2 uploads exist; neither V2 package installed/run. No C GPU process/held lock/queued retry. Will run one complete backfill→control pair on one host only when lock available.

Artifacts: primary /root/autodl-tmp/c-research-20260930/c-longqa-primary-20261002-v1/qmsum-whole-development-v1; separate qmsum-whole-postqual-v1.json preserves launcher list-reader error. Local stable /Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001, qmsum_whole_analysis_v1.json/qmsum_pressure_v1.json/qmsum_prompt_work_v1.json plus plots and checkpoint. Source branch agent/research-c-20260929-v2; no push/publication.

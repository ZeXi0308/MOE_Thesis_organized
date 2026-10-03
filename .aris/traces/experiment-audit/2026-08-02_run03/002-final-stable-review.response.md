# Final verdict: PASS_CANDIDATE_ONLY

Same-family review remains provisional. No formal result exists, and none is
inferred.

Both prior blockers are fixed:

- The emitted request ledger now calls the same compact canonical JSON
  integer-list hash helper as manifest validation.
- The regression independently reconstructs that encoding and checks every
  emitted `capture.request_rows[*]["prompt_token_ids_sha256"]`.
- The audit now correctly distinguishes two authorized, materialized cells
  from execution readiness.
- This agrees with `formal_execution_authorized=true` and the two manifest
  bindings.

No new blocking defect was found.

## Test evidence

- Targeted producer/input suite: **10/10 PASS**
- Full BCRD suite: **72/72 PASS**
- Initial/final hashes matched: **concurrent mutation detected: NO**

## Formal execution

- Permitted now: **NO**.
- HEAD remains `8fe396078ca365afb9ea5d06d8b88c9c01e7a825`; the entire checkout is dirty
  and the reviewed producer/config files remain untracked.
- Formal execution becomes permissible only after these exact reviewed bytes
  are committed, the entire checkout is clean, and exactly one visible RTX
  5090 satisfies the frozen environment guards. This authorizes only the two
  producer-qualification cells, not Gate 0/Gate 1 or scientific claims.

## Final SHA-256

```text
583caa0d99db30e708d94da858a561e42830eb7ede94ebfe4e4bba3a4d664b67  build_continuous_workloads.py
564d9fb6734462789eaca9bf0cf5cfd1ff8a04271a923cacf021015c6893b2db  capture_continuous_decode.py
50101d96e57591abca342e4dea7ace26e02cf85f2fd59355cab5a04767710105  test_continuous_decode.py
5664e1e457548b6564a1bf3d24af5c3d2d98c1d1ddbd6510a93556ea49042de4  gate0_continuous_decode_v1.json
2bf4b4897c15b165fea90d730ed9136d0777535daab7f6807336c09a7c70cdbe  olmoe.formal.json
83a8b410b2d9c22bd2c13a760cb242f5a40691dc156cb24a3fcd9de725674f79  llmjp.formal.json
8e12a7ab15be1f4eea61333dc62f9c92157a1e0579ff2f5be1b0431dee80bcc2  gate0_continuous_decode_experiment_card_2026-08-02.md
a5cc0a9d83c148f1604479139cac6f89b66eca22257c93a53d48ce8c0b9af16e  spec-gate0-continuous-decode-producer-2026-08-02.md
ca1d63b489497730bb46f949181f206c9a3f61d2762808970895cabdb901ea13  gate0_audit_2026-08-02.md
0342fb23e47fbbe7a280a8563ab90dfd57baac497818e648a590321f123c981f  code-review-gate0-continuous-decode-2026-08-02.md
f98269bd3084988cc952a272c8d6eec97f50e189d7689b9c581c2c170c4a623e  experiments/shared/modeling.py
853e60ce71c4c70af83f889a09105b9a31cad29d7bf10d95c9bdeaf16ba1a8c2  pasted-text.txt
```

# N0d Post-Result Experiment-Integrity Audit

Date: 2026-08-24  
Reviewer: fresh same-family sub-agent `/root/n0d_result_integrity_audit`  
Fresh reviewer verdict: `WARN`, P0 = 0, P1 = 1  
Current resolution: `P1_PARTIALLY_CLOSED / EXACT_TIE_BREAK_PROVENANCE_WARN /
LOCAL_DIRECTORY_PROVENANCE_WARN / NARROW_CLAIM_SUPPORTED`

The reviewer accepted the scientific payload and the narrow
`PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED` result. The `WARN` was not
a failed scientific reproduction: at review time the tracker/report were not
closed, and evaluator v1 did not independently derive the retained top-k set
from each row's router logits.

## A-F verdicts

| Area | Fresh verdict | Resolution and bounded conclusion |
|---|---|---|
| A. Reference provenance | `PASS` | Tokens are reconstructed from a sealed request ledger. This is a same-model conformance reference, not real ground truth. Exact loaded weight-file digests are absent and remain a provenance ceiling. |
| B. Normalization | `PASS` | Classification uses raw logit and Expert-set differences with fixed tolerances; no output-derived self-normalization was found. |
| C. File/key/number existence | `WARN` | All manifest-listed bundle/capture/campaign hashes and the frozen evaluator replay passed. This report and tracker close the publication-state mismatch, but two post-sentinel unmanifested local `.pyc` files keep exact-directory provenance at WARN. |
| D. Dead code/fail-open validation | `PASS` with one P1 | Core paths were live. Evaluator v3 validates every retained Expert set for top-k value consistency and has tamper regressions; exact GPU tie-break identity is not reconstructible from tied logits. |
| E. Scope versus language | `PASS` | One model, one RTX 5090, one fixed four-request/eight-step cell, three fresh processes. No broader serving or mechanism claim is made. |
| F. Evaluation type | `PASS` | `self_supervised_proxy` plus real-GPU custom-runtime execution-conformance measurement; neither real GT nor simulation. |

## P1 and append-only remediation

Evaluator v1 validated Expert count, range, and uniqueness but trusted the
stored IDs after that. All 4,608 router rows satisfy the top-k value-ordering
condition. Of those, 48 have an exact tie at the selection boundary; a CPU
`torch.topk` replay chooses a different, equally valued set in 18 tied rows.
The exact GPU backend tie-break identity therefore cannot be recovered from the
retained logits alone. The original narrow result is not invalidated, but exact
selected-ID provenance on tied rows remains `WARN`.

Evaluator v3 adds a fail-closed, softmax-equivalent ranking invariant for every
row:

```text
min(selected router logits) >= max(unselected router logits)
```

It allows legitimate exact boundary ties and rejects replacement by a
lower-logit Expert. This proves value consistency, not the backend's exact
choice among tied Experts. The tamper and exact-tie tests pass. The full local
experiment suite is `202/202 PASS`.

The v3 evaluator replay is external to the immutable campaign:

- original verdict SHA-256:
  `0775016fc43a8d4649e4b8c715cf10ad867255ebde146e0023c170ab7597e616`;
- v3 evaluator SHA-256:
  `29e82f46ff70ea69c3a63b936085503597a51be5d4298e61bd4ffc62ab494922`;
- v3 replay verdict SHA-256:
  `8d7f3855f1817a0f000112be2b59befd8941ebb718412980d598dd7c77036d66`;
- v3 status:
  `PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED`;
- `selected_experts_topk_value_consistent`: `true`;
- `selected_experts_exact_tie_break_recomputed`: `false`;
- `exact_topk_boundary_tie_rows`: `48`;
- structural errors: none.

The append-only v2 verdict remains as historical audit evidence, but its
`selected_experts_topk_recomputed` field name overstates the invariant and is
superseded by the v3 scope fields.

The original `CAMPAIGN_FILES.sha256` still passes after remediation. No
manifest-listed bundle, original verdict, capture, or campaign sentinel was
edited. However, the downloaded local campaign contains two unmanifested
`frozen/__pycache__/*.pyc` files whose mtimes are later than
`CAMPAIGN_COMPLETE.json`. They are consistent with a local post-seal replay,
were not part of the remote scientific campaign, and are retained as an
explicit provenance warning. Consequently, the evidence claim is
"manifest-listed scientific bytes unchanged", not "the local directory was
literally untouched".

## Claim impact

Supported:

> In one frozen OLMoE BF16 custom-Transformers matched-prestate cell, batch-4
> versus serial execution produces a three-process-repeatable Expert-assignment
> divergence whose same-request/same-step router-logit difference is already
> visible at an earlier layer.

Unsupported and still locked: a router-GEMM causal mechanism, a specific
Attention/KV/padding/companion mechanism, native serving transfer, latency or
capacity improvement, action Oracle, Controller, SLO-goodput, and multi-GPU EP.

The complete reviewer request, response, and metadata are retained under
`.aris/traces/experiment-audit/2026-08-24_run01/`.

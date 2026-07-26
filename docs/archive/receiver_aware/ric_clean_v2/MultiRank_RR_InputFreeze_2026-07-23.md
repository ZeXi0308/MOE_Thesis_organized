# Multi-rank RR-credit input freeze (2026-07-23)

## Status

`INPUT_FROZEN / PHYSICAL_EXECUTION_NOT_RUN`

This freeze closes trace-selection bias for the first physical receiver census.
It is not a temporal-incast, transport, topology, or scheduling result.

## Selection contract

- Source: the complete clean-v2 native calibration route trace for each model.
- Unit: one receiver wave identified by model, request, forward, phase,
  decode-step, layer, token, and receiver identity.
- Rule: retain the 64 smallest SHA-256 values under the frozen salt
  `ric-clean-v2-multirank-rr-census-20260723-v1`.
- Eligibility does not read sender multiplicity, timing, queue state, credit
  behavior, policy outcome, or candidate gain.
- Every source row is streamed and validated.  Missing/duplicate top-k slots,
  invalid rows, or non-contiguous repeated waves fail closed.

## Frozen artifacts

| Model | Source rows | Native waves | Selected | Plan artifact hash | Plan file SHA-256 |
|---|---:|---:|---:|---|---|
| OLMoE | 1,048,576 | 131,072 | 64 | `1e6adcc1d5d3478bcd5041b554a2a9cca1a6327ea84e361a08575c9509afc1f6` | `f18612a0f20240a44929d68cd1cc1365b2595fb8bd4b1ee58fee63afe3ae3ff8` |
| LLM-jp | 2,097,152 | 131,072 | 64 | `f31cc522541089f77d414ee077054f99c7520ddc4db4d2967eb27e92142632d8` | `8a8f2060ef39a3fd914ec46a9a67ad3a80db84e6e8faf8a32cb4425b26caf186` |

The bound route-file SHA-256 values are:

- OLMoE: `48b4f4afbe20e3ba9aba20ec18993ee3a922ccb8183e32f21f001bea90e71afc`;
- LLM-jp: `74dec1ece776e642cbd892005926e0f9cfb48ddb5f0c1780a7ae1c09e05c716d`.

## Post-selection structural diagnostics

These diagnostics were computed only after the 64 waves were frozen.

- OLMoE has 3--7 distinct remote virtual senders per selected wave; all 64
  waves have at least three.
- LLM-jp has 6--7 distinct remote virtual senders per selected wave; all 64
  waves have at least three.
- OLMoE contains 512 frozen contributions at 4,096 BF16 payload bytes each.
- LLM-jp contains 1,024 frozen contributions at 1,024 bytes each.

This establishes route-level fan-in support only.  It says nothing about
whether expert readiness overlaps in time, whether a receiver becomes busy, or
whether multiple legal first-credit actions coexist.

## Execution boundary

- One GPU may run only the local schema/hook smoke with all ranks folded and
  `LOCAL_CLONE` transport.
- A functional distributed P2P dry run needs two independent GPUs.
- The physical incast existence gate needs at least four independent GPUs;
  eight are preferred because the frozen virtual EP8 identities remain exact.
- The first physical policy is RR-credit only, with `B={1,2,4,8}` and 20 warmup
  plus 100 measured waves per cell.  Candidate scheduling is forbidden until
  the natural temporal-headroom gates pass on both models.


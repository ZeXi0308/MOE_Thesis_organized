# LongBench MultiFieldQA-en first-16 output note (2026-10-01)

This is a reading of the completed fixed first-16 native run, not a new score or an official full LongBench result. Row numbers below are zero-based source indices. The frozen scorer gives **52.00435% mean English QA F1** over all 16, with 4 normalized exact matches, 13 qualified natural-EOS finishes, and three 64-token length finishes. F1 measures overlap between the **entire** generated answer and the best reference; it is not a fraction of questions answered correctly. I leave the reference answers, generated text, cap, and scoring unchanged.

## Three interpretation traps

| Row | Frozen output and reference | What the recorded score means |
|---:|---|---|
| #13 | Output `Joule`; reference `Watt, one joule per second.` | **F1 = 1/3**, because `joule` is one overlapping token in the five-token normalized reference. The answer to the SI unit of **power** is watt; joule alone names energy and is factually wrong. The relevant source sentence was cut from the model's retained input, as the earlier input-evidence note records. F1 credit does not establish correct reasoning or access to that sentence. |
| #8 | Output `1-0`; reference says the Buckeyes won **15-3**. | **F1 = 0**. The untruncated first schedule row contains `15–3 || 1–0`: the first is the game score, the second the cumulative win-loss record after that game. The question's word “record” can invite the second reading, but the frozen reference chooses the game result. This is an output/reference interpretation difference; it does not authorize changing the gold or score. |
| #12 | Output `90-120 mcg/day`; reference `90 μg for women and 120 μg for men.` | **F1 = 0** under the official normalization: the output becomes `90120 mcgday`, while the reference keeps separate `90`, `120`, `μg`, and the sex assignments. The output conveys a range and a plausible unit paraphrase but omits which value belongs to women versus men. The supporting sentence was removed by the fixed middle truncation, so this is not evidence that the model read it. |

## The three length finishes

All three reached exactly 64 generated tokens and ended with `finish_reason=length`, not qualified natural EOS. Their visible text is a relevant explanation that is **cut mid-thought**, with no visible repeated loop:

- **#3, ICD:** defines the device and describes monitoring/shock, then stops after “if an abnormal”. The exact glossary support was removed from retained input.
- **#5, Kondo effect:** states early that superconductivity tends to suppress it, adds a condition and further context, then stops at “For”. The relevant support was retained.
- **#14, horizontal mobile business model:** states “flexibility” early, elaborates on vendor options, then stops at “carrier-”. The relevant support was retained.

These are observations of the first 64 output tokens. They do not establish how any answer would have ended at a larger cap, nor do they justify trimming generated prose before the official full-output F1. The prior input-evidence reading classified 12/16 as support retained, 3/16 as removed, and 1/16 as unclear; that input reading is separate from output scoring.

**Read-only sources.** Frozen workload SHA-256 `504e28d18cc3fd959cedd5763df169d2e17ec74f768d69e7456fdc752d09c1b4`; native `measured-outputs.json` SHA-256 `753de081d61611d6b27f4a5408f0e216018d000a7c11ac42f68c8f935e6a80d9`; `long_document_qa_qualification_v1.json` SHA-256 `ef930d7404663b3cdf560c6c6c604a92ed8bd160f8e6e1ded4a62bd74397cada`. Input-retention judgments are from `C_LONG_DOCUMENT_QA_INPUT_EVIDENCE_20261001.md`.

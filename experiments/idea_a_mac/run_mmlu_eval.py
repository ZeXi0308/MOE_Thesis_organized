"""Lightweight MMLU zero-shot eval for the FP8+tail-INT4 thesis.

Compares task accuracy (not just PPL) under:
  - full (BF16 baseline)
  - uniform_fp8 (50% saving)
  - fp8top4_rest_int4 (62.5% saving, rank-aware tail)

Uses log-likelihood scoring: for each question, compute LL of each answer
letter (A/B/C/D) and pick argmax. Samples N questions across a few subjects
to keep Mac CPU runtime tractable.

Addresses 短板 2: does the +0.30 PPL degradation translate to task accuracy loss?
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset

from capture_moe import patch_mixtral_moe
from modeling import load_model, load_tokenizer


SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_chemistry",
    "computer_security", "conceptual_physics", "high_school_geography",
    "high_school_mathematics", "high_school_us_history", "machine_learning",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--num-per-subject", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=320)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--num-receiver-groups", type=int, default=4)
    p.add_argument("--output-dir", default="outputs/main_experiments/olmoe_mmlu")
    return p.parse_args()


def format_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = [f"The following is a multiple choice question. Answer with the letter.",
             "", question, ""]
    for l, c in zip(letters, choices):
        lines.append(f"{l}. {c}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def choice_lls(model, tokenizer, prompt: str, seq_len: int) -> list[float]:
    """Log-likelihood of each answer letter (A/B/C/D) given the prompt."""
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len)
    prompt_ids = prompt_enc["input_ids"]
    prompt_len = prompt_ids.shape[1]
    letters = ["A", "B", "C", "D"]
    lls = []
    for letter in letters:
        letter_ids = tokenizer(letter, add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = torch.cat([prompt_ids, letter_ids], dim=1)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        # LL of the first letter token, predicted at position prompt_len-1
        log_probs = F.log_softmax(logits[0, prompt_len - 1, :].float(), dim=-1)
        lls.append(float(log_probs[letter_ids[0, 0].item()].item()))
    return lls


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("loading MMLU...", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    rng = random.Random(42)
    questions = []
    for subj in SUBJECTS:
        subj_rows = [r for r in ds if r["subject"] == subj]
        rng.shuffle(subj_rows)
        questions.extend(subj_rows[: args.num_per_subject])
    print(f"sampled {len(questions)} questions from {len(SUBJECTS)} subjects", flush=True)

    tokenizer = load_tokenizer(args.model)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype)
    print(f"model loaded in {load_seconds:.1f}s", flush=True)

    strategies = ["full", "uniform_fp8", "fp8top4_rest_int4"]
    results = []

    for strat in strategies:
        patch_mixtral_moe(model, strat, num_receiver_groups=args.num_receiver_groups,
                          receiver_mapping="contiguous")
        correct = 0
        for qi, q in enumerate(questions):
            prompt = format_prompt(q["question"], q["choices"])
            lls = choice_lls(model, tokenizer, prompt, args.seq_len)
            pred = lls.index(max(lls))
            if pred == q["answer"]:
                correct += 1
            if (qi + 1) % 20 == 0:
                print(f"  {strat}: {qi+1}/{len(questions)} acc={correct/(qi+1):.3f}", flush=True)
        acc = correct / len(questions)
        row = {"strategy": strat, "n_questions": len(questions), "correct": correct, "accuracy": acc}
        results.append(row)
        print(row, flush=True)
        pd.DataFrame(results).to_csv(out / "mmlu_results.partial.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv(out / "mmlu_results.csv", index=False)
    print(f"\nsaved to {out}/mmlu_results.csv", flush=True)
    print("\n=== MMLU Summary ===")
    for _, r in df.iterrows():
        print(f"  {r['strategy']:25s}  acc={r['accuracy']:.3f}  ({r['correct']}/{r['n_questions']})")


if __name__ == "__main__":
    main()

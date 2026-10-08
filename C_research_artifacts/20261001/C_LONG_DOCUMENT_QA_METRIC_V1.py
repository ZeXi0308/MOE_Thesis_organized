#!/usr/bin/env python3
"""English QA metric copied verbatim from the pinned official LongBench code.

Official commit: 2e00731f8d0bff23dc4325161044d0ed8af94c1e
metrics.py Git blob: 657cf773dcca6ef2aaeae96353055f208e95b35f
Source: https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py
Only the English QA functions and their standard-library dependencies are used.
The multifieldqa_en evaluator scores complete output against all references and
averages the maximum reference F1 over every planned request.
"""
import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def multi_reference_f1(prediction, answers):
    if not isinstance(prediction, str) or not answers or any(not isinstance(a, str) for a in answers):
        raise ValueError("expected prediction text and nonempty reference list")
    return max(qa_f1_score(prediction, answer) for answer in answers)

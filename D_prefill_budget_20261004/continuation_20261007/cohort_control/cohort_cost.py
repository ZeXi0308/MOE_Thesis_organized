"""Conditional decode-only costs for the current cohort, not feasibility bounds.

No arrivals or decoders created by subsequent prefills are included. Each current
request advances once per step and leaves after its declared remaining R tokens.
C is its context feature at the first projected step, matching the fitted model.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def remaining_costs(cohort, model) -> dict[int, float]:
    """Return H -> seconds to execute H decode-only steps, for distinct R values.

    cohort contains (positive integer R, nonnegative context C) pairs. The caller
    uses the native current cohort (D <= 192) and the existing le512 model whose
    coefficients are [P, D, Cd, Cp]. At P=Cp=0, step cost is a+b*D+c*Cd.
    Require a>0 and b,c>=0 so every nonempty projected step has positive cost.

    With m_j=min(H,R_j), cost is a*H + b*sum(m_j) +
    c*sum(C_j*m_j + m_j*(m_j-1)/2). Sorting and prefix/suffix sums take
    O(n log n) time and O(n) space; work does not scale with output length.
    Empty cohorts return {}. Invalid inputs/nonfinite costs raise ValueError.
    """
    try:
        rows = list(cohort)
    except TypeError as exc:
        raise ValueError("cohort must contain (R, C) pairs") from exc
    if not rows:
        return {}
    if not isinstance(model, Mapping):
        raise ValueError("model must be a calibration model mapping")
    try:
        coefficients = list(model["coefficients"])
        a = _finite_number(model["intercept_s"], "intercept_s")
    except (KeyError, TypeError) as exc:
        raise ValueError("model requires intercept_s and four coefficients") from exc
    if len(coefficients) != 4:
        raise ValueError("model coefficients must be [P, D, Cd, Cp]")
    _, b, c, _ = [_finite_number(x, "coefficient") for x in coefficients]
    if a <= 0 or b < 0 or c < 0:
        raise ValueError("decode model requires a>0 and b,c>=0")

    validated = []
    for row in rows:
        try:
            r, context = row
        except (TypeError, ValueError) as exc:
            raise ValueError("cohort entries must be (R, C) pairs") from exc
        if isinstance(r, bool) or not isinstance(r, Integral) or r <= 0:
            raise ValueError("R must be a positive integer")
        context = _finite_number(context, "C")
        if context < 0:
            raise ValueError("C must be nonnegative")
        validated.append((int(r), context))
    validated.sort(key=lambda row: row[0])
    n = len(validated)
    suffix_context = [0.0] * (n + 1)
    try:
        for i in range(n - 1, -1, -1):
            suffix_context[i] = suffix_context[i + 1] + validated[i][1]
        prefix_r = prefix_triangle = 0
        prefix_context_r = 0.0
        result = {}
        for i, (h, context) in enumerate(validated):
            prefix_r += h
            prefix_triangle += h * (h - 1) // 2
            prefix_context_r += context * h
            if i + 1 < n and validated[i + 1][0] == h:
                continue
            remaining = n - i - 1
            tokens = prefix_r + remaining * h
            contexts = math.fsum((prefix_context_r, h * suffix_context[i + 1],
                                  prefix_triangle + remaining * (h * (h - 1) // 2)))
            cost = math.fsum((a * h, b * tokens, c * contexts))
            if not math.isfinite(contexts) or not math.isfinite(cost) or cost <= 0:
                raise ValueError("projected cost must be finite and positive")
            result[h] = cost
    except OverflowError as exc:
        raise ValueError("projected cost overflowed") from exc
    return result

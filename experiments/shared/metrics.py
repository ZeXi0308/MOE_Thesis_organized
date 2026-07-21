from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SampleMetric:
    sample_id: int
    nll_sum: float
    token_count: int
    kl_sum: float = 0.0

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / max(self.token_count, 1)

    @property
    def mean_token_kl(self) -> float:
        return self.kl_sum / max(self.token_count, 1)


@dataclass
class MetricAccumulator:
    samples: list[SampleMetric] = field(default_factory=list)

    def add(
        self,
        sample_id: int,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        baseline_logits: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> SampleMetric:
        nll_sum, token_count = token_nll(logits, input_ids, attention_mask)
        kl_sum = 0.0
        if baseline_logits is not None:
            kl_sum, kl_count = token_kl(baseline_logits, logits, attention_mask)
            if kl_count != token_count:
                raise RuntimeError(f"KL token count {kl_count} != NLL token count {token_count}")
        row = SampleMetric(sample_id, nll_sum, token_count, kl_sum)
        self.samples.append(row)
        return row

    @property
    def nll_sum(self) -> float:
        return sum(row.nll_sum for row in self.samples)

    @property
    def kl_sum(self) -> float:
        return sum(row.kl_sum for row in self.samples)

    @property
    def token_count(self) -> int:
        return sum(row.token_count for row in self.samples)

    @property
    def corpus_ppl(self) -> float:
        return math.exp(self.nll_sum / max(self.token_count, 1))

    @property
    def mean_token_kl(self) -> float:
        return self.kl_sum / max(self.token_count, 1)

    def bootstrap_summary(self, n_bootstrap: int = 1000, seed: int = 42) -> dict[str, float]:
        base = {
            "corpus_ppl": self.corpus_ppl,
            "mean_token_kl": self.mean_token_kl,
            "token_count": self.token_count,
            "sample_count": len(self.samples),
        }
        if len(self.samples) < 2 or n_bootstrap <= 0:
            base.update(
                {
                    "corpus_ppl_ci_low": self.corpus_ppl,
                    "corpus_ppl_ci_high": self.corpus_ppl,
                    "mean_token_kl_ci_low": self.mean_token_kl,
                    "mean_token_kl_ci_high": self.mean_token_kl,
                }
            )
            return base

        rng = np.random.default_rng(seed)
        n = len(self.samples)
        ppl_values = np.empty(n_bootstrap, dtype=np.float64)
        kl_values = np.empty(n_bootstrap, dtype=np.float64)
        for idx in range(n_bootstrap):
            chosen = rng.integers(0, n, size=n)
            nll = sum(self.samples[i].nll_sum for i in chosen)
            kl = sum(self.samples[i].kl_sum for i in chosen)
            tokens = sum(self.samples[i].token_count for i in chosen)
            ppl_values[idx] = math.exp(nll / max(tokens, 1))
            kl_values[idx] = kl / max(tokens, 1)

        base.update(
            {
                "corpus_ppl_ci_low": float(np.quantile(ppl_values, 0.025)),
                "corpus_ppl_ci_high": float(np.quantile(ppl_values, 0.975)),
                "mean_token_kl_ci_low": float(np.quantile(kl_values, 0.025)),
                "mean_token_kl_ci_high": float(np.quantile(kl_values, 0.975)),
            }
        )
        return base

    def sample_rows(self, strategy: str) -> list[dict[str, float | int | str]]:
        return [
            {
                "strategy": strategy,
                "sample_id": row.sample_id,
                "token_count": row.token_count,
                "nll_sum": row.nll_sum,
                "mean_nll": row.mean_nll,
                "sample_ppl": math.exp(row.mean_nll),
                "kl_sum": row.kl_sum,
                "mean_token_kl": row.mean_token_kl,
            }
            for row in self.samples
        ]


def _shift_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if input_ids.shape[1] < 2:
        return torch.zeros_like(input_ids[:, :0], dtype=torch.bool)
    if attention_mask is None:
        return torch.ones_like(input_ids[:, 1:], dtype=torch.bool)
    return attention_mask[:, 1:].bool()


def token_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    if logits.shape[1] < 2:
        return 0.0, 0
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    mask = _shift_mask(input_ids, attention_mask)
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    return float(losses[mask].sum().item()), int(mask.sum().item())


def token_kl(
    baseline_logits: torch.Tensor,
    approx_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    if baseline_logits.shape != approx_logits.shape:
        raise ValueError(f"logit shape mismatch: {baseline_logits.shape} vs {approx_logits.shape}")
    if baseline_logits.shape[1] < 2:
        return 0.0, 0

    baseline = baseline_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    log_p = F.log_softmax(baseline, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    p = log_p.exp()
    per_token = (p * (log_p - log_q)).sum(dim=-1)
    fake_ids = torch.zeros(
        (baseline_logits.shape[0], baseline_logits.shape[1]),
        dtype=torch.long,
        device=baseline_logits.device,
    )
    mask = _shift_mask(fake_ids, attention_mask)
    return float(per_token[mask].sum().item()), int(mask.sum().item())

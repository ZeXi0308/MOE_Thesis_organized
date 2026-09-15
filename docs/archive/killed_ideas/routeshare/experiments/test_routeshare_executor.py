from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routeshare_executor import (  # noqa: E402
    closure_error,
    execute_expert_stage,
    execute_tenants_separately,
)


class ScaleExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


class RouteShareExecutorTests(unittest.TestCase):
    def test_coalition_equals_separate(self) -> None:
        experts = [ScaleExpert(1.0), ScaleExpert(2.0), ScaleExpert(3.0), ScaleExpert(4.0)]
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        selected = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [0, 3]], dtype=torch.long
        )
        weights = torch.full((6, 2), 0.5)
        tenants = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        coalition = execute_expert_stage(experts, hidden, selected, weights)
        separate = execute_tenants_separately(experts, hidden, selected, weights, tenants)
        max_abs, max_rel = closure_error(coalition, separate)
        self.assertEqual(max_abs, 0.0)
        self.assertEqual(max_rel, 0.0)

    def test_rejects_duplicate_shape_and_out_of_range(self) -> None:
        experts = [ScaleExpert(1.0), ScaleExpert(2.0)]
        hidden = torch.ones(2, 4)
        selected = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        weights = torch.full((2, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "out of range"):
            execute_expert_stage(experts, hidden, selected, weights)


if __name__ == "__main__":
    unittest.main()

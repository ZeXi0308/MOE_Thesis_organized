from __future__ import annotations

from pathlib import Path
import sys
import unittest

from torch import nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_joulequeue_expert_inputs_gpu import _expert_modules, select_experts


class _Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)
        self.act_fn = nn.SiLU()


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_Expert() for _ in range(8)])


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(6)])


class ExpertInputCaptureTest(unittest.TestCase):
    def test_expert_identity_and_selection_are_stable(self) -> None:
        modules = _expert_modules(_Model())
        self.assertEqual(len(modules), 48)
        selected = select_experts(set(modules), 20260722)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len({layer for layer, _ in selected}), 4)
        self.assertTrue(all(key in modules for key in selected))
        self.assertEqual(selected, select_experts(set(modules), 20260722))


if __name__ == "__main__":
    unittest.main()

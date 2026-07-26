from __future__ import annotations

import unittest

try:
    import torch
    import triton  # noqa: F401

    from codec_kernels import assert_codec_matches_reference
except ImportError:
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA+Triton required")
class CodecKernelGpuTests(unittest.TestCase):
    def test_zero_extreme_ties_and_config_shapes_match_reference(self) -> None:
        for hidden in (512, 2048):
            for mode, qmax in (("int8", 127), ("int4", 7)):
                source = torch.zeros((3, hidden), device="cuda", dtype=torch.bfloat16)
                source[0, :10] = torch.tensor(
                    [
                        -qmax,
                        -6.5,
                        -3.5,
                        -2.5,
                        -0.5,
                        0.5,
                        2.5,
                        3.5,
                        6.5,
                        qmax,
                    ],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                source[1, :8] = torch.tensor(
                    [-qmax, -1.0, -0.0, 0.0, 1.0, qmax, -qmax, qmax],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                finfo = torch.finfo(torch.bfloat16)
                source[2, :4] = torch.tensor(
                    [-finfo.max, -finfo.tiny, finfo.tiny, finfo.max],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                assert_codec_matches_reference(source, mode)


if __name__ == "__main__":
    unittest.main()

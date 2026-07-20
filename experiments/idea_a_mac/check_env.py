from __future__ import annotations

import platform
import sys

import torch
import transformers


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"mps_available: {torch.backends.mps.is_available()}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print("selected_device: cpu")


if __name__ == "__main__":
    main()


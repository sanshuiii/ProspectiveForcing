#!/usr/bin/env python3
"""Print a compact environment report and fail on missing core dependencies."""

from __future__ import annotations

import importlib
import platform

import torch


def main() -> None:
    required = ["omegaconf", "einops", "safetensors", "transformers", "diffusers"]
    for name in required:
        importlib.import_module(name)
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_version={torch.version.cuda}")
    print(f"gpu_count={torch.cuda.device_count()}")


if __name__ == "__main__":
    main()

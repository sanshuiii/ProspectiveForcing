#!/usr/bin/env python3
"""Validate the public checkpoint contract without constructing the model."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from utils.checkpoint_compat import normalize_checkpoint_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    payload = torch.load(args.checkpoint, map_location="cpu")
    if "generator_ema" not in payload or not isinstance(payload["generator_ema"], dict):
        raise RuntimeError("checkpoint must contain a generator_ema state dictionary")
    state = normalize_checkpoint_keys(payload["generator_ema"])
    if not any(key.startswith("prospective_draft.") for key in state):
        raise RuntimeError("checkpoint does not contain the Prospective Forcing draft")
    if args.stage == "stage2" and not any(
        key.startswith("m1_controller.") for key in state
    ):
        raise RuntimeError("Stage 2 checkpoint does not contain the M1 controller")
    print(f"PASS stage={args.stage} tensors={len(state)} sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()

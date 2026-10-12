"""Strict M1 checkpoint loading with a narrow source-smoke exception."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn


def load_m1_checkpoint(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    method: str,
    allow_untrained_controller: bool,
) -> dict[str, Any]:
    method = str(method).upper()
    if method not in {"R1", "M1", "RM1"}:
        raise ValueError(f"unsupported M1 method: {method}")
    incompatible = module.load_state_dict(state_dict, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    controller_missing = [
        key for key in missing if key.startswith("m1_controller.")
    ]
    invalid_missing = [
        key for key in missing if not key.startswith("m1_controller.")
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"M1 checkpoint mismatch: missing={invalid_missing}, "
            f"unexpected={unexpected}"
        )
    if allow_untrained_controller:
        if not controller_missing or len(controller_missing) != len(missing):
            raise RuntimeError(
                "precalibration smoke must initialize only declared "
                "m1_controller keys from config"
            )
    elif missing:
        raise RuntimeError(
            "formal M1 inference requires a complete trained controller: "
            f"missing={missing}"
        )
    return {
        "status": "PASS",
        "m1_method": method,
        "source_controller_initialized": bool(controller_missing),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "invalid_missing_keys": invalid_missing,
    }

"""Checkpoint compatibility helpers for the public Prospective Forcing release."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping


_LEGACY_PREFIXES = (
    ("eagle_draft.", "prospective_draft."),
    ("tab16_controller.", "m1_controller."),
)


def normalize_checkpoint_keys(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Normalize wrappers and translate pre-release checkpoint prefixes.

    The published code uses Prospective Forcing names. Some earlier checkpoints
    used two older module prefixes. Translating them here accepts either format
    without changing tensors.
    """

    normalized: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        new_key = (
            key.replace("._fsdp_wrapped_module.", ".")
            .replace("._checkpoint_wrapped_module.", ".")
            .replace("._orig_mod.", ".")
            .replace("_fsdp_wrapped_module.", "")
        )
        for old_prefix, public_prefix in _LEGACY_PREFIXES:
            if new_key.startswith(old_prefix):
                new_key = public_prefix + new_key[len(old_prefix):]
                break
        if new_key in normalized:
            raise RuntimeError(
                f"checkpoint key collision after normalization: {new_key}"
            )
        normalized[new_key] = value
    return normalized

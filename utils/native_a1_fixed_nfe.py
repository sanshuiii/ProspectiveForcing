"""A1 fixed-main-backbone-NFE and auxiliary interval contracts.

Only actual calls into the causal Wan backbone are recorded here.  Auxiliary
head, gate, fusion, VAE, and metric work are deliberately excluded from the
main-backbone trace and must be reported separately by their callers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


ALLOWED_MAIN_BACKBONE_CALL_KINDS = frozenset({"sampling", "clean_cache"})


def _canonical_timestep_trace(timestep: torch.Tensor) -> list[list[float]]:
    if timestep.ndim == 1:
        timestep = timestep.unsqueeze(0)
    if timestep.ndim != 2:
        raise ValueError("timestep trace must have shape [batch, frames]")
    return [
        [float(value) for value in row]
        for row in timestep.detach().to(device="cpu", dtype=torch.float64).tolist()
    ]


@dataclass(frozen=True)
class MainBackboneTraceEntry:
    ordinal: int
    call_kind: str
    current_start_token: int
    input_frames: int
    logical_chunks: int
    updating_cache: bool
    timestep_trace: list[list[float]]

    def to_dict(self) -> dict:
        return {
            "ordinal": self.ordinal,
            "call_kind": self.call_kind,
            "current_start_token": self.current_start_token,
            "input_frames": self.input_frames,
            "logical_chunks": self.logical_chunks,
            "updating_cache": self.updating_cache,
            "timestep_trace": self.timestep_trace,
        }


class MainBackboneTraceRecorder:
    """Record the exact call/timestep/chunk trace of the real backbone."""

    def __init__(self) -> None:
        self._entries: list[MainBackboneTraceEntry] = []

    def record(self, call_kind: str, logical_chunks: int, kwargs: Mapping) -> None:
        if call_kind not in ALLOWED_MAIN_BACKBONE_CALL_KINDS:
            raise RuntimeError(f"forbidden main-backbone call kind: {call_kind!r}")
        noisy = kwargs.get("noisy_image_or_video")
        timestep = kwargs.get("timestep")
        if not isinstance(noisy, torch.Tensor) or noisy.ndim != 5:
            raise ValueError("backbone input must have shape [batch, frames, channels, h, w]")
        if not isinstance(timestep, torch.Tensor):
            raise ValueError("every backbone call must expose its timestep tensor")
        entry = MainBackboneTraceEntry(
            ordinal=len(self._entries),
            call_kind=call_kind,
            current_start_token=int(kwargs.get("current_start") or 0),
            input_frames=int(noisy.shape[1]),
            logical_chunks=int(logical_chunks),
            updating_cache=bool(kwargs.get("updating_cache", False)),
            timestep_trace=_canonical_timestep_trace(timestep),
        )
        if entry.call_kind == "sampling" and entry.updating_cache:
            raise RuntimeError("sampling call may not mutate the clean cache")
        if entry.call_kind == "clean_cache" and not entry.updating_cache:
            # Initial-latent cache priming is legal, but A1 is T2V and must
            # therefore never exercise that special path.
            raise RuntimeError("A1 clean-cache calls must set updating_cache=True")
        self._entries.append(entry)

    @property
    def count(self) -> int:
        return len(self._entries)

    def entries(self) -> list[dict]:
        return [entry.to_dict() for entry in self._entries]

    def digest(self) -> str:
        payload = json.dumps(
            self.entries(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def summary(self) -> dict:
        entries = self.entries()
        return {
            "main_backbone_forward_count": len(entries),
            "main_backbone_trace_sha256": self.digest(),
            "main_backbone_trace": entries,
            "main_backbone_sampling_calls": sum(
                entry["call_kind"] == "sampling" for entry in entries
            ),
            "main_backbone_recache_calls": sum(
                entry["call_kind"] == "clean_cache" for entry in entries
            ),
        }


def assert_paired_main_backbone_traces(control: Mapping, treatment: Mapping) -> None:
    """Hard fairness gate: count, timestep trace, and chunk trace are exact."""

    required = (
        "main_backbone_forward_count",
        "main_backbone_trace_sha256",
        "main_backbone_trace",
        "main_backbone_sampling_calls",
        "main_backbone_recache_calls",
    )
    for key in required:
        if control.get(key) != treatment.get(key):
            raise RuntimeError(f"paired main-backbone mismatch at {key}")


def auxiliary_interval_coverage(
    accept_probabilities: Sequence[float],
    interval_count: int,
    tau: float = 0.5,
) -> dict:
    """Apply the approved L=1/L=2 coverage controller without a backbone."""

    if interval_count < 1:
        raise ValueError("interval_count must be positive")
    if not math.isclose(float(tau), 0.5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("A1 freezes tau at exactly 0.5")
    cursor = 0
    decision_index = 0
    spans: list[int] = []
    covered: list[int] = []
    while cursor < interval_count:
        remaining = interval_count - cursor
        if remaining == 1:
            span = 1
        else:
            if decision_index >= len(accept_probabilities):
                raise ValueError("insufficient frozen gate probabilities")
            probability = float(accept_probabilities[decision_index])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("gate probabilities must be finite and in [0, 1]")
            span = 2 if probability >= tau else 1
            decision_index += 1
        spans.append(span)
        covered.extend(range(cursor, cursor + span))
        cursor += span
    if covered != list(range(interval_count)):
        raise RuntimeError("auxiliary intervals were duplicated, omitted, or exited early")
    return {
        "interval_count": interval_count,
        "tau": float(tau),
        "spans": spans,
        "covered_intervals": covered,
        "gate_decisions_consumed": decision_index,
    }


def assert_q1_only_legal_commit(committed_horizons: Iterable[int]) -> None:
    horizons = list(committed_horizons)
    if any(horizon != 0 for horizon in horizons):
        raise RuntimeError("A1 permits q1 commits only; q2 must never commit")

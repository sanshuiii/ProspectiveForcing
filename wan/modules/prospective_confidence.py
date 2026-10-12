"""M1 confidence routing and K=2 verification primitives.

This module deliberately does not own the native Prospective Forcing q1 head.  Keeping
the controller separate preserves every native checkpoint key and lets the
caller account for each actual q1 candidate invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


METHODS = {"R1", "M1", "RM1"}


def _as_batch_scalar(value: torch.Tensor, batch: int) -> torch.Tensor:
    value = value.detach().float()
    if value.ndim == 0:
        return value.expand(batch)
    return value.reshape(batch, -1).mean(dim=1)


def detached_pooled_stats(value: torch.Tensor) -> torch.Tensor:
    """Return detached mean/std/RMS statistics, one row per sample."""

    if value.ndim < 2:
        raise ValueError("confidence features require a batch dimension")
    detached = value.detach().float()
    normalized = F.layer_norm(detached, (detached.shape[-1],))
    flat = normalized.reshape(normalized.shape[0], -1)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    rms = flat.square().mean(dim=1).clamp_min(1e-12).sqrt()
    return torch.stack((mean, std, rms), dim=1)


def detached_raw_pooled_stats(value: torch.Tensor) -> torch.Tensor:
    """Return detached raw-feature mean/std/RMS statistics per sample."""
    if value.ndim < 2:
        raise ValueError("confidence features require a batch dimension")
    detached = value.detach().float()
    flat = detached.reshape(detached.shape[0], -1)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    rms = flat.square().mean(dim=1).clamp_min(1e-12).sqrt()
    return torch.stack((mean, std, rms), dim=1)


def normalized_feature_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-sample E = ||LN(pred)-LN(target)||^2 / D over all tokens."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    pred = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    truth = F.layer_norm(target.detach().float(), (target.shape[-1],))
    return (pred - truth).square().reshape(pred.shape[0], -1).mean(dim=1)


def feature_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return feature_prediction_loss_per_sample(prediction, target).mean()


def feature_prediction_loss_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the paper's normalized feature distance for each batch item."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    pred = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    truth = F.layer_norm(target.detach().float(), (target.shape[-1],))
    mse = (pred - truth).square().reshape(pred.shape[0], -1).mean(dim=1)
    cosine = F.cosine_similarity(pred, truth, dim=-1).reshape(
        pred.shape[0], -1
    ).mean(dim=1)
    return 0.5 * mse + 1.0 - cosine


class CalibratedLogErrorHead(nn.Module):
    """Predict a log-error distribution and its threshold probability."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        features: torch.Tensor,
        delta: float,
    ) -> Dict[str, torch.Tensor]:
        # The native generator runs in BF16 while freshly attached M1
        # controller modules may still be FP32 (and vice versa after a model
        # dtype conversion).  LayerNorm requires its input and affine weights
        # to agree, so make that boundary explicit instead of relying on
        # autocast promotion rules.
        parameter = next(self.network.parameters())
        features = features.to(device=parameter.device, dtype=parameter.dtype)
        raw = self.network(features)
        mean = raw[:, 0]
        log_scale = raw[:, 1].clamp(-6.0, 3.0)
        scale = log_scale.exp()
        threshold = features.new_tensor(float(delta)).clamp_min(1e-12).log()
        z = (threshold - mean) / scale
        probability = 0.5 * (1.0 + torch.erf(z / (2.0 ** 0.5)))
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        return {
            "mean_log_error": mean,
            "log_scale": log_scale,
            "probability": probability,
        }


class LearnedCandidateVerifier(nn.Module):
    """Learned candidate-1/candidate-2/reject classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        parameter = next(self.network.parameters())
        features = features.to(device=parameter.device, dtype=parameter.dtype)
        return self.network(features)


@dataclass
class RouteDecision:
    probability: torch.Tensor
    execute: torch.Tensor


@dataclass
class VerificationDecision:
    logits: torch.Tensor
    selected_index: torch.Tensor
    reject: torch.Tensor


class M1ConfidenceController(nn.Module):
    """Separate additive controller for R1, M1, or RM1.

    All confidence/verifier statistics are detached.  Candidate embeddings are
    the only inputs added before independent native q1 candidate calls.
    """

    def __init__(
        self,
        dim: int,
        method: str,
        delta: float = 0.12,
        tau_run: float = 0.5,
        tau_accept: float = 0.5,
        interval_vocab: int = 64,
        interval_embedding_dim: int = 8,
        candidate_seed: int = 20260816,
    ) -> None:
        super().__init__()
        method = str(method).upper()
        if method not in METHODS:
            raise ValueError(f"unsupported M1 method: {method!r}")
        if dim < 1 or interval_vocab < 1:
            raise ValueError("dim and interval_vocab must be positive")
        self.dim = int(dim)
        self.method = method
        self.delta = float(delta)
        self.tau_run = float(tau_run)
        self.tau_accept = float(tau_accept)
        self.candidate_count = 2 if method in {"M1", "RM1"} else 1
        self.has_pre_router = method in {"R1", "RM1"}
        self.has_post_router = method == "R1"
        self.has_verifier = method in {"M1", "RM1"}

        self.interval_embedding = (
            nn.Embedding(interval_vocab, interval_embedding_dim)
            if self.has_pre_router or self.has_post_router
            else None
        )
        context_dim = 3 + 1 + interval_embedding_dim
        if self.has_pre_router:
            self.g1_pre = CalibratedLogErrorHead(context_dim)
        if self.has_post_router:
            self.c1_post = CalibratedLogErrorHead(context_dim + 3)
        if self.has_verifier:
            # Raw mean/std/RMS for state, candidates, and residuals;
            # pair cosine/distance. No timestep or rollout interval input.
            verifier_dim = 3 + 6 + 6 + 2
            self.verifier = LearnedCandidateVerifier(verifier_dim)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(candidate_seed))
                embeddings = torch.empty(2, dim)
                nn.init.normal_(embeddings, mean=0.0, std=0.02)
            embeddings[0].zero_()
            self.candidate_embeddings = nn.Parameter(embeddings)

    def _context_features(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        interval: torch.Tensor,
    ) -> torch.Tensor:
        batch = state.shape[0]
        if self.interval_embedding is None:
            raise RuntimeError("rollout interval embedding is unavailable")
        controller_device = self.interval_embedding.weight.device
        if state.device != controller_device:
            raise RuntimeError(
                "M1 controller placement mismatch: "
                f"state is on {state.device}, controller is on {controller_device}"
            )
        timestep_value = _as_batch_scalar(timestep, batch).div(1000.0).unsqueeze(1)
        timestep_value = timestep_value.to(device=controller_device)
        interval_index = interval.detach().to(
            device=controller_device, dtype=torch.long
        ).reshape(batch, -1)[:, 0]
        interval_index = interval_index.clamp(0, self.interval_embedding.num_embeddings - 1)
        interval_features = self.interval_embedding(interval_index)
        return torch.cat(
            (detached_pooled_stats(state), timestep_value, interval_features),
            dim=1,
        )

    def pre_route(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        interval: torch.Tensor,
        force_execute: bool = False,
    ) -> RouteDecision:
        batch = state.shape[0]
        if not self.has_pre_router:
            probability = state.new_ones(batch, dtype=torch.float32)
        else:
            output = self.g1_pre(
                self._context_features(state, timestep, interval),
                self.delta,
            )
            probability = output["probability"]
        execute = probability >= self.tau_run
        if force_execute:
            execute = torch.ones_like(execute, dtype=torch.bool)
        return RouteDecision(probability=probability, execute=execute)

    def candidate_noise(
        self,
        future_noise: torch.Tensor,
        candidate_index: int,
    ) -> torch.Tensor:
        if not 0 <= candidate_index < self.candidate_count:
            raise ValueError("candidate index outside configured range")
        if self.candidate_count == 1:
            return future_noise
        embedding = self.candidate_embeddings[candidate_index].to(
            device=future_noise.device, dtype=future_noise.dtype
        )
        return future_noise + embedding.view(1, 1, -1)

    def execute_candidates(
        self,
        route: RouteDecision,
        future_noise: torch.Tensor,
        candidate_call,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Execute exactly the candidate calls authorized by a pre-route.

        Formal M1 inference is batch-size one.  Keeping the zero-call path
        inside this helper makes a pre-route skip observable and testable rather
        than merely adjusting counters after candidates have already run.
        """

        execute_any = bool(route.execute.any())
        execute_all = bool(route.execute.all())
        if execute_any != execute_all:
            raise RuntimeError(
                "mixed M1 routes require batch_size=1 inference"
            )
        if not execute_any:
            return [], []

        candidate_features: List[torch.Tensor] = []
        candidate_flows: List[torch.Tensor] = []
        for candidate_index in range(self.candidate_count):
            candidate_noise = self.candidate_noise(
                future_noise, candidate_index
            )
            candidate_state, candidate_flow = candidate_call(candidate_noise)
            candidate_features.append(candidate_state)
            candidate_flows.append(candidate_flow)
        return candidate_features, candidate_flows

    def post_route(
        self,
        state: torch.Tensor,
        candidate: torch.Tensor,
        timestep: torch.Tensor,
        interval: torch.Tensor,
    ) -> RouteDecision:
        if not self.has_post_router:
            raise RuntimeError("post router is only defined for R1")
        context = self._context_features(state, timestep, interval)
        residual = candidate - state[:, -candidate.shape[1] :]
        features = torch.cat((context, detached_pooled_stats(residual)), dim=1)
        output = self.c1_post(features, self.delta)
        probability = output["probability"]
        return RouteDecision(
            probability=probability,
            execute=probability >= self.tau_accept,
        )

    def _verifier_features(
        self,
        state: torch.Tensor,
        candidates: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(candidates) != 2 or candidates[0].shape != candidates[1].shape:
            raise ValueError("verifier requires two shape-matched candidates")
        tail = state[:, -candidates[0].shape[1] :]
        residuals = [candidate - tail for candidate in candidates]
        flat0 = candidates[0].detach().float().reshape(candidates[0].shape[0], -1)
        flat1 = candidates[1].detach().float().reshape(candidates[1].shape[0], -1)
        cosine = F.cosine_similarity(flat0, flat1, dim=1)
        distance = (flat0 - flat1).square().mean(dim=1).sqrt()
        features = torch.cat(
            (
                detached_raw_pooled_stats(state),
                detached_raw_pooled_stats(candidates[0]),
                detached_raw_pooled_stats(candidates[1]),
                detached_raw_pooled_stats(residuals[0]),
                detached_raw_pooled_stats(residuals[1]),
                cosine.unsqueeze(1),
                distance.unsqueeze(1),
            ),
            dim=1,
        )
        return features, cosine, distance

    def verify(
        self,
        state: torch.Tensor,
        candidates: Sequence[torch.Tensor],
    ) -> VerificationDecision:
        if not self.has_verifier:
            raise RuntimeError("verifier is only defined for M1/RM1")
        features, _, _ = self._verifier_features(state, candidates)
        logits = self.verifier(features)
        selected = logits.argmax(dim=1)
        reject = selected == 2
        return VerificationDecision(
            logits=logits,
            selected_index=selected,
            reject=reject,
        )

    @staticmethod
    def select_top1(
        candidates: Sequence[torch.Tensor],
        decision: VerificationDecision,
    ) -> Optional[torch.Tensor]:
        if len(candidates) != 2:
            raise ValueError("top-1 selection requires two candidates")
        if bool(decision.reject.all()):
            return None
        stacked = torch.stack(tuple(candidates), dim=1)
        gather_index = decision.selected_index.clamp_max(1).view(
            stacked.shape[0], 1, *([1] * (stacked.ndim - 2))
        )
        gather_index = gather_index.expand(
            stacked.shape[0], 1, *stacked.shape[2:]
        )
        selected = stacked.gather(1, gather_index).squeeze(1)
        if bool(decision.reject.any()):
            selected = selected.masked_fill(
                decision.reject.view(-1, *([1] * (selected.ndim - 1))), 0.0
            )
        return selected

    def supervised_loss(
        self,
        state: torch.Tensor,
        candidates: Sequence[torch.Tensor],
        target: torch.Tensor,
        timestep: torch.Tensor,
        interval: torch.Tensor,
        step: int,
        candidate_alignment_beta: float = 0.05,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(candidates) != self.candidate_count:
            raise ValueError("candidate count does not match controller method")
        if not 0.0 <= candidate_alignment_beta <= 1.0:
            raise ValueError("candidate_alignment_beta must be in [0, 1]")
        errors = torch.stack(
            [normalized_feature_error(candidate, target) for candidate in candidates],
            dim=1,
        )
        per_candidate_loss = torch.stack(
            [
                feature_prediction_loss_per_sample(candidate, target)
                for candidate in candidates
            ],
            dim=1,
        )
        best_error, best_index = errors.min(dim=1)
        acceptable = best_error <= self.delta
        best_loss = per_candidate_loss.gather(1, best_index[:, None]).squeeze(1)
        if self.candidate_count > 1:
            other_loss = (
                per_candidate_loss.sum(dim=1) - best_loss
            ) / (self.candidate_count - 1)
            prediction_loss = (
                (1.0 - candidate_alignment_beta) * best_loss
                + candidate_alignment_beta * other_loss
            ).mean()
        else:
            prediction_loss = best_loss.mean()
        # R1 preserves the native q1 objective exactly: the existing Prospective Forcing
        # aligned feature/flow loss trains q1, while detached router losses
        # train only g1_pre/c1_post. M1/RM1 use explicit candidate feature
        # alignment here.
        total = (
            prediction_loss
            if self.candidate_count == 2
            else prediction_loss.new_zeros(())
        )
        logs: Dict[str, torch.Tensor] = {
            "candidate_alignment_loss": prediction_loss.detach(),
            "oracle_best_error": best_error.detach().mean(),
            "oracle_accept_rate": acceptable.float().mean().detach(),
        }

        if self.has_pre_router:
            context = self._context_features(state, timestep, interval)
            pre = self.g1_pre(context, self.delta)["probability"]
            label = acceptable.float()
            pre_bce = F.binary_cross_entropy(pre, label)
            pre_brier = (pre - label).square().mean()
            cost_weight = 0.0 if step < 200 else min(1.0, (step - 200) / 500.0)
            cost_regularization = pre.mean() * (0.01 * cost_weight)
            total = total + pre_bce + pre_brier + cost_regularization
            logs.update(
                {
                    "g1_pre_bce": pre_bce.detach(),
                    "g1_pre_brier": pre_brier.detach(),
                    "g1_pre_probability": pre.detach().mean(),
                    "router_cost_regularization": cost_regularization.detach(),
                }
            )

        if self.has_post_router:
            context = self._context_features(state, timestep, interval)
            residual = candidates[0] - state[:, -candidates[0].shape[1] :]
            post_features = torch.cat(
                (context, detached_pooled_stats(residual)), dim=1
            )
            post = self.c1_post(post_features, self.delta)["probability"]
            label = (errors[:, 0] <= self.delta).float()
            post_bce = F.binary_cross_entropy(post, label)
            post_brier = (post - label).square().mean()
            total = total + post_bce + post_brier
            logs.update(
                {
                    "c1_post_bce": post_bce.detach(),
                    "c1_post_brier": post_brier.detach(),
                    "c1_post_probability": post.detach().mean(),
                }
            )

        if self.has_verifier:
            verifier_features, cosine, distance = self._verifier_features(
                state, candidates
            )
            logits = self.verifier(verifier_features)
            reject_index = torch.full_like(best_index, 2)
            label = torch.where(acceptable, best_index, reject_index)
            ranking = F.cross_entropy(logits, label)
            reject_probability = logits.softmax(dim=1)[:, 2]
            reject_brier = (
                reject_probability - (~acceptable).float()
            ).square().mean()
            total = total + ranking + reject_brier
            predicted = logits.argmax(dim=1)
            logs.update(
                {
                    "verifier_cross_entropy": ranking.detach(),
                    "verifier_reject_brier": reject_brier.detach(),
                    "candidate_cosine": cosine.detach().mean(),
                    "candidate_distance": distance.detach().mean(),
                    "verifier_top1_accuracy": (predicted == label).float().mean().detach(),
                    "verifier_reject_rate": (predicted == 2).float().mean().detach(),
                }
            )

        logs["m1_auxiliary_loss"] = total.detach()
        return total, logs


def curriculum_stage(method: str, step: int) -> str:
    method = str(method).upper()
    if method not in METHODS or not 0 <= step < 1000:
        raise ValueError("method must be R1/M1/RM1 and step must be in [0, 999]")
    if method == "R1":
        return "fixed_a1_calibration" if step < 200 else (
            "confidence_cost" if step < 700 else "runtime_routing"
        )
    if method == "M1":
        return "joint_candidate_verifier"
    if step < 200:
        return "candidates"
    if step < 500:
        return "verifier"
    if step < 700:
        return "router_fixed_a1"
    return "runtime_router_k2"

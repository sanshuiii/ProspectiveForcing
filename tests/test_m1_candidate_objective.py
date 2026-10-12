import unittest
from types import SimpleNamespace

import torch

from wan.modules.prospective_confidence import (
    M1ConfidenceController,
    detached_raw_pooled_stats,
    feature_prediction_loss_per_sample,
    normalized_feature_error,
)
from pipeline.rolling_forcing_training import RollingForcingTrainingPipeline


class M1CandidateObjectiveTest(unittest.TestCase):
    def test_verifier_uses_raw_stats_without_timestep_or_interval(self):
        torch.manual_seed(11)
        batch, tokens, dim = 2, 5, 8
        state = torch.randn(batch, tokens, dim)
        candidates = [torch.randn(batch, 3, dim), torch.randn(batch, 3, dim)]
        controller = M1ConfidenceController(dim=dim, method="M1")

        features, _, _ = controller._verifier_features(state, candidates)
        decision = controller.verify(state, candidates)

        expected = torch.cat(
            (
                detached_raw_pooled_stats(state),
                detached_raw_pooled_stats(candidates[0]),
                detached_raw_pooled_stats(candidates[1]),
                detached_raw_pooled_stats(candidates[0] - state[:, -3:]),
                detached_raw_pooled_stats(candidates[1] - state[:, -3:]),
                torch.nn.functional.cosine_similarity(
                    candidates[0].flatten(1), candidates[1].flatten(1), dim=1
                ).unsqueeze(1),
                (candidates[0] - candidates[1]).square().flatten(1)
                .mean(dim=1)
                .sqrt()
                .unsqueeze(1),
            ),
            dim=1,
        )
        torch.testing.assert_close(features, expected)
        self.assertEqual(features.shape, (batch, 17))
        self.assertEqual(decision.logits.shape, (batch, 3))
        self.assertIsNone(controller.interval_embedding)

    def test_candidate_alignment_uses_best_candidate_weight(self):
        torch.manual_seed(7)
        batch, tokens, dim = 3, 4, 8
        target = torch.randn(batch, tokens, dim)
        candidates = [
            target + 0.03 * torch.randn_like(target),
            target + 0.30 * torch.randn_like(target),
        ]
        beta = 0.05
        controller = M1ConfidenceController(dim=dim, method="M1")

        _, logs = controller.supervised_loss(
            state=target,
            candidates=candidates,
            target=target,
            timestep=torch.full((batch,), 500),
            interval=torch.zeros(batch, dtype=torch.long),
            step=0,
            candidate_alignment_beta=beta,
        )

        errors = torch.stack(
            [normalized_feature_error(candidate, target) for candidate in candidates],
            dim=1,
        )
        per_candidate = torch.stack(
            [
                feature_prediction_loss_per_sample(candidate, target)
                for candidate in candidates
            ],
            dim=1,
        )
        best = errors.argmin(dim=1)
        best_loss = per_candidate.gather(1, best[:, None]).squeeze(1)
        other_loss = per_candidate.sum(dim=1) - best_loss
        expected = ((1 - beta) * best_loss + beta * other_loss).mean()

        torch.testing.assert_close(logs["candidate_alignment_loss"], expected)
        self.assertNotIn("candidate_diversity_loss", logs)

    def test_native_alignment_uses_oracle_best_candidate_not_verifier_choice(self):
        torch.manual_seed(13)
        target = torch.randn(2, 4, 8)
        candidate_0 = target.clone()
        candidate_1 = target.clone()
        candidate_1[0] += torch.randn_like(candidate_1[0])
        candidate_0[1] += torch.randn_like(candidate_0[1])
        target_flow = torch.zeros(2, 3)
        flow_0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        flow_1 = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

        class DummyController:
            def supervised_loss(self, **kwargs):
                return kwargs["target"].new_zeros(()), {}

        pipeline = object.__new__(RollingForcingTrainingPipeline)
        pipeline.generator = SimpleNamespace(m1_controller=DummyController())
        pipeline.prospective_loss_weight = 1.0
        pipeline.prospective_feature_weight = 1.0
        pipeline.prospective_flow_weight = 1.0
        pipeline.m1_loss_weight = 0.0
        pipeline.m1_candidate_alignment_beta = 0.05
        pipeline.m1_training_step = 0
        pipeline.last_prospective_log_dict = {}
        pipeline.last_prospective_loss = None

        draft_metadata = {
            # Emulate a verifier that currently picks candidate 1 for both samples.
            "draft_features": [candidate_1],
            "draft_flows": [flow_1],
            "m1": {
                "candidate_features": [[candidate_0, candidate_1]],
                "candidate_flows": [[flow_0, flow_1]],
                "supervision": {
                    "state": target,
                    "timestep": torch.full((2,), 500),
                    "interval": torch.zeros(2, dtype=torch.long),
                },
            },
        }

        pipeline._finalize_prospective_loss(
            draft_metadata, {0: (target, target_flow)}
        )

        self.assertAlmostEqual(
            pipeline.last_prospective_log_dict["prospective_feature_loss"].item(),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            pipeline.last_prospective_log_dict["prospective_flow_loss"].item(),
            0.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()

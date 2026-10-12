import unittest

from utils.checkpoint_compat import normalize_checkpoint_keys


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_public_keys_are_unchanged(self):
        marker = object()
        result = normalize_checkpoint_keys({"prospective_draft.weight": marker})
        self.assertIs(result["prospective_draft.weight"], marker)

    def test_wrappers_and_legacy_prefixes_are_normalized(self):
        state = {
            "_fsdp_wrapped_module.eagle_draft.weight": 1,
            "_fsdp_wrapped_module.tab16_controller.bias": 2,
        }
        result = normalize_checkpoint_keys(state)
        self.assertEqual(
            result,
            {
                "prospective_draft.weight": 1,
                "m1_controller.bias": 2,
            },
        )

    def test_collisions_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "collision"):
            normalize_checkpoint_keys(
                {
                    "eagle_draft.weight": 1,
                    "prospective_draft.weight": 2,
                }
            )


if __name__ == "__main__":
    unittest.main()

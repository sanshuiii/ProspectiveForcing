import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTest(unittest.TestCase):
    def test_checkpoint_slots_are_empty(self):
        for name in ("base", "stage1", "stage2"):
            files = [path.name for path in (ROOT / "checkpoints" / name).iterdir()]
            self.assertEqual(files, [".gitkeep"])

    def test_artifact_manifest_is_machine_readable(self):
        manifest = json.loads(
            (ROOT / "checkpoints" / "artifact_manifest.json").read_text()
        )
        self.assertEqual(manifest["schema_version"], "prospective_forcing_artifacts_v1")
        self.assertEqual(len(manifest["artifacts"]), 3)

    def test_scientific_configs_keep_registered_contract(self):
        stage1 = (ROOT / "configs" / "training" / "stage1_cm_init.yaml").read_text()
        stage2 = (ROOT / "configs" / "training" / "stage2_m1.yaml").read_text()
        for text in (stage1, stage2):
            self.assertIn("denoising_step_list: [1000, 800, 600, 400, 200]", text)
            self.assertIn("expected_world_size: 8", text)
            self.assertIn("batch_size: 1", text)
            self.assertIn("total_batch_size: 8", text)
        self.assertIn("max_steps: 3000", stage1)
        self.assertIn("max_steps: 1000", stage2)
        self.assertIn("m1_method: M1", stage2)
        self.assertIn("m1_candidate_alignment_beta: 0.05", stage2)
        self.assertIn("m1_delta: 0.12", stage2)
        self.assertNotIn("m1_tau_run", stage2)
        self.assertNotIn("m1_tau_accept", stage2)

    def test_no_private_absolute_paths_or_credentials(self):
        excluded = {ROOT / "LICENSE", ROOT / "THIRD_PARTY_NOTICES.md"}
        tokens = [
            "/" + "Users" + "/",
            "/" + "home" + "/",
            "wandb" + "_api_key",
            "github" + "_token",
            "hf" + "_token",
        ]
        for path in ROOT.rglob("*"):
            if not path.is_file() or path in excluded:
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            for token in tokens:
                self.assertNotIn(token.lower(), text.lower(), str(path))
            self.assertIsNone(
                re.search(r"(?:ssh|https?)://[^\s/]+@", text, flags=re.IGNORECASE),
                str(path),
            )


if __name__ == "__main__":
    unittest.main()

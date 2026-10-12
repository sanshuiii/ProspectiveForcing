import unittest

from utils.prospective_eval_contract import assert_prospective_frame_contract


class EvaluationFrameContractTest(unittest.TestCase):
    def test_formal_contract_is_126_latent_and_480_rgb(self):
        assert_prospective_frame_contract(
            require_prospective=True,
            short_smoke=False,
            num_output_frames=126,
            num_save_frames=480,
        )
        with self.assertRaises(AssertionError):
            assert_prospective_frame_contract(
                require_prospective=True,
                short_smoke=False,
                num_output_frames=21,
                num_save_frames=80,
            )

    def test_short_smoke_contract_is_exact(self):
        assert_prospective_frame_contract(
            require_prospective=True,
            short_smoke=True,
            num_output_frames=21,
            num_save_frames=80,
        )


if __name__ == "__main__":
    unittest.main()

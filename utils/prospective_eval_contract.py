from __future__ import annotations


def assert_prospective_frame_contract(
    *,
    require_prospective: bool,
    short_smoke: bool,
    num_output_frames: int,
    num_save_frames: int | None,
) -> None:
    """Keep formal PROSPECTIVE dimensions immutable while admitting one strict smoke shape."""
    if not require_prospective:
        assert not short_smoke, "short PROSPECTIVE smoke requires --require_prospective"
        return

    if short_smoke:
        expected_output_frames = 21
        expected_save_frames = 80
        contract_name = "strict short smoke"
    else:
        expected_output_frames = 126
        expected_save_frames = 480
        contract_name = "formal"

    assert num_output_frames == expected_output_frames, (
        f"{contract_name} PROSPECTIVE output frames must be {expected_output_frames}; "
        f"got {num_output_frames}"
    )
    assert num_save_frames == expected_save_frames, (
        f"{contract_name} PROSPECTIVE saved frames must be {expected_save_frames}; "
        f"got {num_save_frames}"
    )

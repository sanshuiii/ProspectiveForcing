from typing import List, Optional
import time
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.native_a1_fixed_nfe import (
    MainBackboneTraceRecorder,
    assert_q1_only_legal_commit,
)
from wan.modules.prospective_confidence import normalized_feature_error


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache_clean = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.last_inference_stats = {}

    @staticmethod
    def _draft_output_num_chunks(draft_module, rolling_window_num_chunks):
        """Return the configured head output width without changing its compute."""
        if getattr(draft_module, "window_prediction", False):
            return rolling_window_num_chunks
        if getattr(draft_module, "suffix_prediction", False):
            output_num_chunks = int(draft_module.output_num_chunks)
            if not 1 <= output_num_chunks <= rolling_window_num_chunks:
                raise RuntimeError(
                    "invalid Prospective Forcing suffix output width: "
                    f"{output_num_chunks}"
                )
            return output_num_chunks
        return 1

    @staticmethod
    def _stitch_last_chunk_window(
        main_window,
        committed_draft_chunks,
        horizon,
        frames_per_chunk,
    ):
        """Advance Rolling by retaining overlap and appending q1/q2 tail chunks."""
        if horizon < 0 or horizon >= len(committed_draft_chunks):
            raise RuntimeError("invalid last-chunk draft horizon")
        retained_start = (horizon + 1) * frames_per_chunk
        retained_main = main_window[:, retained_start:]
        stitched = torch.cat(
            [retained_main, *committed_draft_chunks[:horizon + 1]],
            dim=1,
        )
        if stitched.shape != main_window.shape:
            raise RuntimeError(
                "last-chunk Rolling stitch shape mismatch: "
                f"{tuple(stitched.shape)} vs {tuple(main_window.shape)}"
            )
        return stitched

    def inference_rolling_forcing(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        auxiliary_window_heads: int = 0,
        m1_force_execute: bool = False,
        m1_calibration_shadow: bool = False,
        decode_output: bool = True,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        if auxiliary_window_heads not in {0, 1, 2}:
            raise ValueError("auxiliary_window_heads must be 0, 1, or 2")
        if auxiliary_window_heads and initial_latent is not None:
            raise ValueError("window-draft evaluation currently supports T2V only")
        if auxiliary_window_heads and not hasattr(self.generator, "prospective_draft"):
            raise ValueError("auxiliary window heads were requested but are absent")
        if m1_force_execute and auxiliary_window_heads != 1:
            raise ValueError("M1 force-execute requires the q1-only A1 path")
        if m1_calibration_shadow:
            if auxiliary_window_heads != 1 or not m1_force_execute:
                raise ValueError(
                    "M1 calibration shadow requires forced q1-only candidates"
                )
            if decode_output:
                raise ValueError("M1 calibration shadow must skip RGB decode")

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        torch.cuda.reset_peak_memory_stats(noise.device)
        counters = {
            "physical_backbone_forward_calls": 0,
            "logical_chunk_nfe": 0,
            "serial_backbone_rounds": 0,
            "sampling_backbone_calls": 0,
            "clean_cache_backbone_calls": 0,
            "draft_head_forward_calls": 0,
            "draft_rounds": 0,
            "proposed_chunks": 0,
            "computed_draft_output_chunks": 0,
            "committed_draft_chunks": 0,
            "accepted_q1": 0,
            "accepted_q2": 0,
            "verifier_backbone_calls": 0,
            "fallback_backbone_calls": 0,
            "pdd_l1_projection_calls": 0,
            "pdd_l2_projection_calls": 0,
            "pdd_gate_calls": 0,
            "pdd_fusion_calls": 0,
            "pdd_estimated_projection_flops": 0,
            "m1_g1_pre_calls": 0,
            "m1_c1_post_calls": 0,
            "m1_candidate_calls": 0,
            "m1_verifier_calls": 0,
            "m1_b0_routes": 0,
            "m1_a1_routes": 0,
            "m1_rejects": 0,
            "m1_saved_candidate_calls": 0,
            "m1_selected_candidate_1": 0,
            "m1_selected_candidate_2": 0,
        }
        accepted_histogram = {"0": 0, "1": 0, "2": 0}
        m1_route_records = []
        backbone_events = []
        draft_events = []
        m1_controller_events = []
        m1_calibration_records = []
        pending_m1_calibration = None
        main_backbone_trace = MainBackboneTraceRecorder()

        def call_generator(call_kind, logical_chunks=0, **kwargs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = self.generator(**kwargs)
            end_event.record()
            main_backbone_trace.record(call_kind, logical_chunks, kwargs)
            backbone_events.append((call_kind, start_event, end_event))
            counters["physical_backbone_forward_calls"] += 1
            counters["serial_backbone_rounds"] += 1
            counters["logical_chunk_nfe"] += int(logical_chunks)
            if call_kind == "sampling":
                counters["sampling_backbone_calls"] += 1
            elif call_kind == "clean_cache":
                counters["clean_cache_backbone_calls"] += 1
            return result

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_clean is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_clean)):
                self.kv_cache_clean[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_clean[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                call_generator(
                    "clean_cache",
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                call_generator(
                    "clean_cache",
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        # implementing rolling forcing 
        # construct the rolling forcing windows
        num_denoising_steps = len(self.denoising_step_list)
        rolling_window_length_blocks = num_denoising_steps
        window_start_blocks = []
        window_end_blocks = []
        window_num = num_blocks + rolling_window_length_blocks - 1

        for window_index in range(window_num):
            start_block = max(0, window_index - rolling_window_length_blocks + 1)
            end_block = min(num_blocks - 1, window_index)
            window_start_blocks.append(start_block)
            window_end_blocks.append(end_block)

        # init noisy cache
        noisy_cache = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # init denosing timestep, same accross windows
        shared_timestep = torch.ones(
            [batch_size, rolling_window_length_blocks * self.num_frame_per_block],
            device=noise.device,
            dtype=torch.float32)
        
        for index, current_timestep in enumerate(reversed(self.denoising_step_list)): # from clean to noisy 
            shared_timestep[:, index * self.num_frame_per_block:(index + 1) * self.num_frame_per_block] *= current_timestep


        def window_spec(window_index):
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]
            current_start_frame = start_block * self.num_frame_per_block
            current_end_frame = (end_block + 1) * self.num_frame_per_block
            current_num_frames = current_end_frame - current_start_frame
            if (
                current_num_frames
                == rolling_window_length_blocks * self.num_frame_per_block
                or current_start_frame == 0
            ):
                noisy_input = torch.cat([
                    noisy_cache[
                        :, current_start_frame:
                        current_end_frame - self.num_frame_per_block
                    ],
                    noise[
                        :, current_end_frame - self.num_frame_per_block:
                        current_end_frame
                    ],
                ], dim=1)
            else:
                noisy_input = noisy_cache[
                    :, current_start_frame:current_end_frame
                ].clone()
            if (
                current_num_frames
                == rolling_window_length_blocks * self.num_frame_per_block
            ):
                current_timestep = shared_timestep
            elif current_start_frame == 0:
                current_timestep = shared_timestep[:, -current_num_frames:]
            elif current_end_frame == num_frames:
                current_timestep = shared_timestep[:, :current_num_frames]
            else:
                raise ValueError("invalid rolling window length")
            return (
                start_block,
                end_block,
                current_start_frame,
                current_end_frame,
                current_num_frames,
                noisy_input,
                current_timestep,
            )

        def update_noisy_cache(
            start_block,
            end_block,
            current_start_frame,
            denoised_pred,
            current_timestep,
        ):
            with torch.no_grad():
                for block_idx in range(start_block, end_block + 1):
                    local_start = (
                        block_idx - start_block
                    ) * self.num_frame_per_block
                    local_end = local_start + self.num_frame_per_block
                    block_time_step = current_timestep[
                        :, local_start:local_end
                    ].mean().item()
                    matches = (
                        torch.abs(
                            self.denoising_step_list - block_time_step
                        ) < 1e-4
                    )
                    block_timestep_index = torch.nonzero(
                        matches, as_tuple=True
                    )[0]
                    if (
                        block_timestep_index
                        == len(self.denoising_step_list) - 1
                    ):
                        continue
                    next_timestep = self.denoising_step_list[
                        block_timestep_index + 1
                    ].to(noise.device)
                    renoised = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * denoised_pred.shape[1]],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                    noisy_cache[
                        :, block_idx * self.num_frame_per_block:
                        (block_idx + 1) * self.num_frame_per_block
                    ] = renoised[:, local_start:local_end]

        def refresh_clean_cache(
            current_start_frame,
            denoised_pred,
            current_timestep,
        ):
            with torch.no_grad():
                context_timestep = (
                    torch.ones_like(current_timestep) * self.args.context_noise
                )
                clean_chunk = denoised_pred[:, :self.num_frame_per_block]
                clean_timestep = context_timestep[
                    :, :self.num_frame_per_block
                ]
                call_generator(
                    "clean_cache",
                    noisy_image_or_video=clean_chunk,
                    conditional_dict=conditional_dict,
                    timestep=clean_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    updating_cache=True,
                )

        def commit_window(spec, denoised_pred):
            (
                start_block,
                end_block,
                current_start_frame,
                current_end_frame,
                _,
                _,
                current_timestep,
            ) = spec
            output[:, current_start_frame:current_end_frame] = denoised_pred
            update_noisy_cache(
                start_block,
                end_block,
                current_start_frame,
                denoised_pred,
                current_timestep,
            )
            refresh_clean_cache(
                current_start_frame, denoised_pred, current_timestep
            )

        sampling_wall_start = time.perf_counter()
        window_index = 0
        full_window_frames = (
            rolling_window_length_blocks * self.num_frame_per_block
        )
        draft_output_num_chunks = 0
        if hasattr(self.generator, "prospective_draft"):
            draft_output_num_chunks = self._draft_output_num_chunks(
                self.generator.prospective_draft,
                rolling_window_length_blocks,
            )
        draft_output_num_frames = (
            draft_output_num_chunks * self.num_frame_per_block
        )
        last_full_window_index = num_blocks - 1
        while window_index < window_num:
            print("window_index:", window_index)
            spec = window_spec(window_index)
            (
                _,
                _,
                current_start_frame,
                _,
                current_num_frames,
                noisy_input,
                current_timestep,
            ) = spec
            available_future_full_windows = max(
                0, last_full_window_index - window_index
            )
            heads_to_use = 0
            if (
                auxiliary_window_heads
                and current_num_frames == full_window_frames
                and available_future_full_windows > 0
            ):
                heads_to_use = min(
                    auxiliary_window_heads,
                    available_future_full_windows,
                )

            generator_kwargs = dict(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=current_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            if m1_calibration_shadow:
                generator_kwargs["return_features"] = True
            if heads_to_use:
                current_end_frame = spec[3]
                generator_kwargs["m1_force_execute"] = m1_force_execute
                generator_kwargs["prospective_interval_index"] = window_index
                generator_kwargs["prospective_future_noises"] = [
                    noise[
                        :, current_end_frame + horizon * self.num_frame_per_block:
                        current_end_frame
                        + (horizon + 1) * self.num_frame_per_block
                    ]
                    for horizon in range(heads_to_use)
                ]
                generator_kwargs["prospective_future_timesteps"] = [
                    shared_timestep[:, -draft_output_num_frames:]
                    for _ in range(heads_to_use)
                ]

            generator_output = call_generator(
                "sampling",
                logical_chunks=current_num_frames // self.num_frame_per_block,
                **generator_kwargs,
            )
            if heads_to_use or m1_calibration_shadow:
                _, denoised_pred, metadata = generator_output
            else:
                _, denoised_pred = generator_output
                metadata = None

            if m1_calibration_shadow and pending_m1_calibration is not None:
                target = metadata.get("last_chunk_features")
                if target is None:
                    raise RuntimeError("M1 calibration target features are missing")
                candidate_features = pending_m1_calibration.pop(
                    "candidate_features"
                )
                errors = [
                    float(
                        normalized_feature_error(candidate, target)
                        .mean().detach().cpu()
                    )
                    for candidate in candidate_features
                ]
                if not errors or not all(torch.isfinite(torch.tensor(errors))):
                    raise RuntimeError("M1 calibration produced non-finite errors")
                best_error = min(errors)
                best_index = errors.index(best_error)
                m1_calibration_records.append(
                    {
                        **pending_m1_calibration,
                        "target_interval_index": int(window_index),
                        "candidate_errors": errors,
                        "oracle_best_error": best_error,
                        "oracle_best_index": best_index,
                        "target_shape": list(target.shape),
                    }
                )
                pending_m1_calibration = None

            if heads_to_use:
                m1 = metadata.get("m1")
                if m1 is None:
                    if len(metadata["draft_flows"]) != heads_to_use:
                        raise RuntimeError("draft flow count does not match mode")
                    committed_heads = heads_to_use
                else:
                    if heads_to_use != 1 or len(m1["records"]) != 1:
                        raise RuntimeError("M1 is a q1-only inference mode")
                    raw_record = m1["records"][0]
                    calibration_candidates = None
                    if m1_calibration_shadow:
                        groups = m1.get("candidate_features", [])
                        if len(groups) != 1 or not groups[0]:
                            raise RuntimeError(
                                "M1 calibration shadow did not expose candidates"
                            )
                        calibration_candidates = tuple(
                            candidate.detach() for candidate in groups[0]
                        )
                    record = {
                        **raw_record,
                        "pre_probability": float(
                            raw_record["pre_probability"].detach().float().mean().cpu()
                        ),
                        "post_probability": (
                            None if raw_record.get("post_probability") is None
                            else float(
                                raw_record["post_probability"].detach().float().mean().cpu()
                            )
                        ),
                        "verifier_logits": (
                            None if raw_record.get("verifier_logits") is None
                            else raw_record["verifier_logits"].detach().float().cpu().tolist()
                        ),
                        "interval_index": window_index,
                    }
                    m1_route_records.append(record)
                    m1_controller_events.extend(
                        m1.get("controller_timing_events", [])
                    )
                    counters["m1_g1_pre_calls"] += int(
                        m1["method"] in {"R1", "RM1"}
                    )
                    counters["m1_c1_post_calls"] += int(
                        m1["method"] == "R1" and record["candidate_calls"] > 0
                    )
                    counters["m1_verifier_calls"] += int(
                        m1["method"] in {"M1", "RM1"}
                        and record["candidate_calls"] > 0
                    )
                    counters["m1_candidate_calls"] += int(
                        record["candidate_calls"]
                    )
                    if m1_calibration_shadow:
                        pending_m1_calibration = {
                            "source_interval_index": int(window_index),
                            "candidate_features": calibration_candidates,
                            "pre_probability": record["pre_probability"],
                            "post_probability": record["post_probability"],
                            "verifier_logits": record["verifier_logits"],
                            "model_selected_index": int(record["selected_index"]),
                            "model_rejected": bool(record["rejected"]),
                            "candidate_count": int(record["candidate_calls"]),
                            "method": str(m1["method"]),
                        }
                    rejected = bool(record["rejected"])
                    committed_heads = 0 if rejected else 1
                    if m1_calibration_shadow:
                        committed_heads = 0
                    expected_flows = committed_heads
                    if (not m1_calibration_shadow
                            and len(metadata["draft_flows"]) != expected_flows):
                        raise RuntimeError(
                            "M1 draft flows do not match the route decision"
                        )
                    if rejected:
                        counters["m1_b0_routes"] += 1
                        counters["m1_rejects"] += 1
                        if record["candidate_calls"] == 0:
                            counters["m1_saved_candidate_calls"] += int(
                                2 if m1["method"] == "RM1" else 1
                            )
                    else:
                        counters["m1_a1_routes"] += 1
                        selected_index = int(record["selected_index"])
                        if selected_index == 0:
                            counters["m1_selected_candidate_1"] += 1
                        elif selected_index == 1:
                            counters["m1_selected_candidate_2"] += 1
                expected_main_shape = (
                    batch_size,
                    full_window_frames,
                    num_channels,
                    height,
                    width,
                )
                if tuple(denoised_pred.shape) != expected_main_shape:
                    raise RuntimeError(
                        "main Rolling head did not return its full window: "
                        f"{tuple(denoised_pred.shape)} vs {expected_main_shape}"
                    )
                draft_events.extend(
                    metadata.get("draft_head_timing_events", [])
                )
                pdd_stats = metadata.get("adaptive_pdd2", {"enabled": False})
                if pdd_stats.get("enabled", False):
                    counters["pdd_l1_projection_calls"] += int(
                        pdd_stats["l1_projection_calls"]
                    )
                    counters["pdd_l2_projection_calls"] += int(
                        pdd_stats["l2_projection_calls"]
                    )
                    counters["pdd_gate_calls"] += int(pdd_stats["gate_calls"])
                    counters["pdd_fusion_calls"] += int(
                        pdd_stats["fusion_calls"]
                    )
                    counters["pdd_estimated_projection_flops"] += int(
                        pdd_stats["estimated_projection_flops"]
                    )
                    if pdd_stats["main_backbone_forward_calls"] != 0:
                        raise RuntimeError("PDD head attempted a hidden backbone call")
                    if pdd_stats["recache_calls"] != 0:
                        raise RuntimeError("PDD head attempted an extra recache")
                    if pdd_stats["committed_horizon"] != 0:
                        raise RuntimeError("PDD head attempted to commit q2")
                counters["draft_rounds"] += 1
                candidate_calls = (
                    heads_to_use if m1 is None
                    else int(m1["records"][0]["candidate_calls"])
                )
                counters["draft_head_forward_calls"] += candidate_calls
                counters["proposed_chunks"] += candidate_calls
                counters["computed_draft_output_chunks"] += (
                    candidate_calls * draft_output_num_chunks
                )
            else:
                committed_heads = 0

            commit_window(spec, denoised_pred)

            committed_draft_chunks = []
            for horizon in range(committed_heads if heads_to_use else 0):
                draft_flow = metadata["draft_flows"][horizon]
                expected_shape = (
                    batch_size,
                    draft_output_num_frames,
                    num_channels,
                    height,
                    width,
                )
                if tuple(draft_flow.shape) != expected_shape:
                    raise RuntimeError(
                        "draft output shape does not match configured scope: "
                        f"{tuple(draft_flow.shape)} vs {expected_shape}"
                    )
                draft_tail_flow = draft_flow[
                    :, -self.num_frame_per_block:
                ]
                future_start = (
                    spec[3] + horizon * self.num_frame_per_block
                )
                future_end = future_start + self.num_frame_per_block
                actual_xt = noise[:, future_start:future_end]
                tail_timestep = shared_timestep[
                    :, -self.num_frame_per_block:
                ]
                draft_tail_x0 = self.generator._convert_flow_pred_to_x0(
                    flow_pred=draft_tail_flow.flatten(0, 1),
                    xt=actual_xt.flatten(0, 1),
                    timestep=tail_timestep.flatten(0, 1),
                ).unflatten(0, draft_tail_flow.shape[:2])
                if not torch.isfinite(draft_tail_x0).all():
                    raise RuntimeError("non-finite last-chunk draft conversion")
                committed_draft_chunks.append(draft_tail_x0)
                target_window_index = window_index + horizon + 1
                target_spec = window_spec(target_window_index)
                stitched = self._stitch_last_chunk_window(
                    denoised_pred,
                    committed_draft_chunks,
                    horizon,
                    self.num_frame_per_block,
                )
                if stitched.shape != target_spec[5].shape:
                    raise RuntimeError(
                        "last-chunk target window shape mismatch: "
                        f"{tuple(stitched.shape)} vs "
                        f"{tuple(target_spec[5].shape)}"
                    )
                commit_window(target_spec, stitched)
                counters["committed_draft_chunks"] += 1
                if horizon == 0:
                    counters["accepted_q1"] += 1
                elif horizon == 1:
                    counters["accepted_q2"] += 1
            if heads_to_use:
                accepted_histogram[str(committed_heads)] += 1
            window_index += 1 + (committed_heads if heads_to_use else 0)
        if pending_m1_calibration is not None:
            raise RuntimeError("M1 calibration ended without its next-window target")

        torch.cuda.synchronize(noise.device)
        sampling_wall_seconds = time.perf_counter() - sampling_wall_start
        backbone_cuda_ms = {
            "sampling": 0.0,
            "clean_cache": 0.0,
        }
        for call_kind, start_event, end_event in backbone_events:
            backbone_cuda_ms[call_kind] += start_event.elapsed_time(end_event)
        draft_head_cuda_ms = sum(
            start_event.elapsed_time(end_event)
            for start_event, end_event in draft_events
        )
        m1_controller_cuda_ms = {
            "g1_pre": 0.0,
            "c1_post": 0.0,
            "verifier": 0.0,
        }
        for label, start_event, end_event in m1_controller_events:
            m1_controller_cuda_ms[label] += start_event.elapsed_time(end_event)
        route_switches = sum(
            int(
                bool(previous["rejected"])
                != bool(current["rejected"])
            )
            for previous, current in zip(
                m1_route_records, m1_route_records[1:]
            )
        )

        if decode_output:
            # Step 4: Decode the output
            vae_wall_start = time.perf_counter()
            video = self.vae.decode_to_pixel(output, use_cache=False)
            # Five-minute RGB tensors are ~21.5 GiB in FP32.  Normalize in place
            # to preserve the exact arithmetic while avoiding a second full-size
            # CUDA allocation after VAE decode.
            video.mul_(0.5).add_(0.5).clamp_(0, 1)
            torch.cuda.synchronize(noise.device)
            vae_wall_seconds = time.perf_counter() - vae_wall_start
        else:
            video = None
            vae_wall_seconds = 0.0

        accepted_total = (
            counters["accepted_q1"] + counters["accepted_q2"]
        )
        assert_q1_only_legal_commit(
            [0] * counters["accepted_q1"] + [1] * counters["accepted_q2"]
        )
        if main_backbone_trace.count != counters["physical_backbone_forward_calls"]:
            raise RuntimeError("main-backbone recorder lost a physical forward")
        draft_rounds = counters["draft_rounds"]
        wasted_draft_output_chunks = (
            counters["computed_draft_output_chunks"]
            - counters["committed_draft_chunks"]
        )
        assert wasted_draft_output_chunks >= 0
        self.last_inference_stats = {
            **counters,
            **main_backbone_trace.summary(),
            "auxiliary_window_heads": auxiliary_window_heads,
            "rolling_window_num_chunks": rolling_window_length_blocks,
            "main_head_commit_num_chunks": rolling_window_length_blocks,
            "main_head_commit_policy": "full_rolling_window_v1",
            "draft_output_num_chunks": draft_output_num_chunks,
            "draft_commit_num_chunks_per_head": 1,
            "draft_commit_policy": "last_chunk_only_rolling_stitch_v1",
            "accepted_length_histogram": accepted_histogram,
            "q1_accept_rate": (
                counters["accepted_q1"] / draft_rounds
                if draft_rounds else 0.0
            ),
            "q2_conditional_accept_rate": (
                counters["accepted_q2"] / counters["accepted_q1"]
                if counters["accepted_q1"] else 0.0
            ),
            "mean_accepted_chunks_per_round": (
                accepted_total / draft_rounds if draft_rounds else 0.0
            ),
            "fallback_round_rate": (
                counters["m1_b0_routes"] / draft_rounds
                if draft_rounds and m1_route_records else 0.0
            ),
            "wasted_draft_output_chunks": wasted_draft_output_chunks,
            "wasted_draft_ratio": (
                1.0
                - counters["committed_draft_chunks"]
                / counters["computed_draft_output_chunks"]
                if counters["computed_draft_output_chunks"] else 0.0
            ),
            "sampling_wall_seconds": sampling_wall_seconds,
            "sampling_backbone_cuda_ms": backbone_cuda_ms["sampling"],
            "clean_cache_backbone_cuda_ms": backbone_cuda_ms["clean_cache"],
            "draft_head_cuda_ms": draft_head_cuda_ms,
            "m1_g1_pre_cuda_ms": m1_controller_cuda_ms["g1_pre"],
            "m1_c1_post_cuda_ms": m1_controller_cuda_ms["c1_post"],
            "m1_verifier_cuda_ms": m1_controller_cuda_ms["verifier"],
            "m1_route_switches": route_switches,
            "vae_wall_seconds": vae_wall_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                noise.device
            ),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                noise.device
            ),
            "m1_route_records": m1_route_records,
            "m1_force_execute": m1_force_execute,
            "m1_calibration_shadow": m1_calibration_shadow,
            "m1_calibration_record_count": len(m1_calibration_records),
            "m1_calibration_records": m1_calibration_records,
        }
        if profile:
            print("Profiling results:", self.last_inference_stats)

        if return_latents:
            return video, output
        else:
            return video



    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_clean = []
        # if self.local_attn_size != -1:
        #     # Use the local attention size to compute the KV cache size
        #     kv_cache_size = self.local_attn_size * self.frame_seq_length
        # else:
        #     # Use the default KV cache size
        kv_cache_size = 1560 * 24

        for _ in range(self.num_transformer_blocks):
            kv_cache_clean.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_clean = kv_cache_clean  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

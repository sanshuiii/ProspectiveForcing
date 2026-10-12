from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F


class RollingForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache_clean = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.prospective_enabled = bool(kwargs.get("prospective_enabled", False))
        self.prospective_feature_weight = float(kwargs.get("prospective_feature_weight", 1.0))
        self.prospective_flow_weight = float(kwargs.get("prospective_flow_weight", 0.1))
        self.prospective_loss_weight = float(kwargs.get("prospective_loss_weight", 0.1))
        self.m1_loss_weight = float(kwargs.get("m1_loss_weight", 0.1))
        self.m1_candidate_alignment_beta = float(
            kwargs.get("m1_candidate_alignment_beta", 0.05)
        )
        self.m1_training_step = 0
        self.prospective_future_horizons = int(kwargs.get("prospective_future_horizons", 2))
        self.prospective_anchor_policy = str(kwargs.get("prospective_anchor_policy", "start"))
        if self.prospective_future_horizons < 1:
            raise ValueError("prospective_future_horizons must be positive")
        if self.prospective_anchor_policy not in {"start", "tail"}:
            raise ValueError(
                "prospective_anchor_policy must be either 'start' or 'tail'"
            )
        self.last_prospective_loss = None
        self.last_prospective_log_dict = {}

    @staticmethod
    def prospective_window_indices(
        num_blocks: int,
        rolling_window_length_blocks: int,
        future_horizons: int,
        anchor_policy: str,
    ):
        """Return the full anchor window and its aligned future target windows."""
        if future_horizons < 1:
            raise ValueError("future_horizons must be positive")
        if num_blocks < rolling_window_length_blocks + future_horizons:
            return None
        if anchor_policy == "start":
            anchor_window = rolling_window_length_blocks - 1
        elif anchor_policy == "tail":
            anchor_window = num_blocks - future_horizons - 1
        else:
            raise ValueError(f"unknown prospective anchor policy: {anchor_policy!r}")
        target_windows = tuple(
            range(anchor_window + 1, anchor_window + future_horizons + 1)
        )
        return anchor_window, target_windows


    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()
    
    def generate_list(self, num_blocks, num_denoising_steps, device):

        # Generate random indices
        indices = torch.randint(
            low=0,
            high=num_denoising_steps,
            size=(num_blocks,),
            device=device
        )
        if self.last_step_only:
            indices = torch.ones_like(indices) * (num_denoising_steps - 1)

        return indices.tolist()    


    def _finalize_prospective_loss(
        self,
        draft_metadata,
        targets,
    ):
        if draft_metadata is None or len(targets) != len(draft_metadata["draft_features"]):
            raise RuntimeError("Prospective Forcing did not collect both aligned future targets")

        m1_metadata = draft_metadata.get("m1")
        draft_features = list(draft_metadata["draft_features"])
        draft_flows = list(draft_metadata["draft_flows"])
        if m1_metadata is not None:
            candidate_groups = m1_metadata.get("candidate_features", [])
            candidate_flow_groups = m1_metadata.get("candidate_flows", [])
            if (
                not candidate_groups
                or not candidate_flow_groups
                or not candidate_groups[0]
                or len(candidate_groups[0]) != len(candidate_flow_groups[0])
            ):
                raise RuntimeError("M1 candidate outputs are incomplete")

            # The paper defines the native feature/velocity alignment for q1
            # using the oracle-best candidate m*, not the verifier's current
            # prediction. Select m* per sample from the detached target error.
            target_feature = targets[0][0]
            target_norm = F.layer_norm(
                target_feature.detach().float(),
                (target_feature.shape[-1],),
            )
            candidate_errors = []
            for candidate in candidate_groups[0]:
                candidate_norm = F.layer_norm(
                    candidate.float(), (candidate.shape[-1],)
                )
                candidate_errors.append(
                    (candidate_norm - target_norm)
                    .square()
                    .flatten(start_dim=1)
                    .mean(dim=1)
                )
            best_index = torch.stack(candidate_errors, dim=1).argmin(dim=1)

            def select_per_sample(outputs):
                stacked = torch.stack(outputs, dim=1)
                gather_index = best_index.view(
                    best_index.shape[0], 1,
                    *([1] * (stacked.ndim - 2)),
                ).expand(-1, 1, *stacked.shape[2:])
                return stacked.gather(1, gather_index).squeeze(1)

            draft_features[0] = select_per_sample(candidate_groups[0])
            draft_flows[0] = select_per_sample(candidate_flow_groups[0])

        feature_losses = []
        flow_losses = []
        for horizon_index, (target_feature, target_flow) in sorted(targets.items()):
            draft_feature = draft_features[horizon_index]
            draft_flow = draft_flows[horizon_index]
            draft_norm = F.layer_norm(draft_feature.float(), (draft_feature.shape[-1],))
            target_norm = F.layer_norm(target_feature.float(), (target_feature.shape[-1],))
            feature_losses.append(
                0.5 * F.mse_loss(draft_norm, target_norm)
                + 1.0 - F.cosine_similarity(draft_norm, target_norm, dim=-1).mean()
            )
            flow_losses.append(F.mse_loss(draft_flow.float(), target_flow.float()))

        feature_loss = torch.stack(feature_losses).mean()
        flow_loss = torch.stack(flow_losses).mean()
        native_prospective_loss = self.prospective_loss_weight * (
            self.prospective_feature_weight * feature_loss
            + self.prospective_flow_weight * flow_loss
        )
        m1_auxiliary_loss = feature_loss.new_zeros(())
        m1_log_dict = {}
        weighted_m1_loss = feature_loss.new_zeros(())
        if m1_metadata is not None:
            controller = getattr(self.generator, "m1_controller", None)
            supervision = m1_metadata.get("supervision")
            candidate_groups = m1_metadata.get("candidate_features", [])
            if controller is None or supervision is None or not candidate_groups:
                raise RuntimeError("M1 aligned supervision metadata is incomplete")
            m1_auxiliary_loss, m1_log_dict = controller.supervised_loss(
                state=supervision["state"],
                candidates=candidate_groups[0],
                target=targets[0][0],
                timestep=supervision["timestep"],
                interval=supervision["interval"],
                step=int(self.m1_training_step),
                candidate_alignment_beta=self.m1_candidate_alignment_beta,
            )
            weighted_m1_loss = self.m1_loss_weight * m1_auxiliary_loss
        prospective_loss = native_prospective_loss + weighted_m1_loss
        self.last_prospective_loss = prospective_loss
        self.last_prospective_log_dict = {
            "prospective_feature_loss": feature_loss.detach(),
            "prospective_flow_loss": flow_loss.detach(),
            "prospective_loss": prospective_loss.detach(),
        }
        if m1_metadata is not None:
            self.last_prospective_log_dict.update({
                "native_prospective_loss": native_prospective_loss.detach(),
                "weighted_m1_loss": weighted_m1_loss.detach(),
                **m1_log_dict,
            })
        for horizon_index, (feature_value, flow_value) in enumerate(
            zip(feature_losses, flow_losses), start=1
        ):
            self.last_prospective_log_dict[
                f"prospective_feature_loss_h{horizon_index}"
            ] = feature_value.detach()
            self.last_prospective_log_dict[
                f"prospective_flow_loss_h{horizon_index}"
            ] = flow_value.detach()
        return prospective_loss


    def inference_with_rolling_forcing(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        self.last_prospective_loss = noise.new_zeros(())
        self.last_prospective_log_dict = {}
        prospective_active = self.prospective_enabled and torch.is_grad_enabled()
        prospective_draft_metadata = None
        prospective_targets = {}

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
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # implementing rolling forcing 
        # construct the rolling forcing windows
        num_denoising_steps = len(self.denoising_step_list)
        rolling_window_length_blocks = num_denoising_steps
        prospective_windows = self.prospective_window_indices(
            num_blocks=num_blocks,
            rolling_window_length_blocks=rolling_window_length_blocks,
            future_horizons=self.prospective_future_horizons,
            anchor_policy=self.prospective_anchor_policy,
        )
        prospective_active = (
            prospective_active and prospective_windows is not None
            and not self.independent_first_frame
        )
        if prospective_active:
            prospective_anchor_window, prospective_target_windows = prospective_windows
        else:
            prospective_anchor_window, prospective_target_windows = -1, ()
        window_start_blocks = []
        window_end_blocks = []
        window_num = num_blocks + rolling_window_length_blocks - 1

        for window_index in range(window_num):
            start_block = max(0, window_index - rolling_window_length_blocks + 1)
            end_block = min(num_blocks - 1, window_index)
            window_start_blocks.append(start_block)
            window_end_blocks.append(end_block)

        # exit_flag indicates the window at which the model will backpropagate gradients.
        exit_flag = torch.randint(high=rolling_window_length_blocks, device=noise.device, size=())
        start_gradient_frame_index = num_output_frames - 21
        
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


        # Denoising loop with rolling forcing
        for window_index in range(window_num):
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index] # include

            current_start_frame = start_block * self.num_frame_per_block
            current_end_frame = (end_block + 1) * self.num_frame_per_block # not include
            current_num_frames = current_end_frame - current_start_frame

            # noisy_input: new noise and previous denoised noisy frames, only last block is pure noise
            if current_num_frames == rolling_window_length_blocks * self.num_frame_per_block or current_start_frame == 0:
                noisy_input = torch.cat([
                    noisy_cache[:, current_start_frame : current_end_frame - self.num_frame_per_block],
                    noise[:, current_end_frame - self.num_frame_per_block : current_end_frame ]
                ], dim=1)
            else: # at the end of the video
                noisy_input = noisy_cache[:, current_start_frame:current_end_frame].clone()

            # init denosing timestep
            if current_num_frames == rolling_window_length_blocks * self.num_frame_per_block:
                current_timestep = shared_timestep
            elif current_start_frame == 0:
                current_timestep = shared_timestep[:,-current_num_frames:]
            elif current_end_frame == num_frames:
                current_timestep = shared_timestep[:,:current_num_frames]
            else:
                raise ValueError("current_num_frames should be equal to rolling_window_length_blocks * self.num_frame_per_block, or the first or last window.")
            
            require_grad = window_index % rolling_window_length_blocks == exit_flag
            if current_end_frame <= start_gradient_frame_index:
                require_grad = False

            # Keep the official DMD exit window untouched.  The unique full
            # anchor window only gains gradient when Prospective Forcing is active.
            is_prospective_anchor = (
                prospective_active and window_index == prospective_anchor_window
            )
            is_prospective_target = (
                prospective_active and window_index in prospective_target_windows
            )
            call_requires_grad = require_grad or is_prospective_anchor
            generator_kwargs = dict(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=current_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            if is_prospective_anchor:
                generator_kwargs["prospective_interval_index"] = window_index
                future_start = current_end_frame
                generator_kwargs["prospective_future_noises"] = [
                    noise[:, future_start + horizon * self.num_frame_per_block:
                          future_start + (horizon + 1) * self.num_frame_per_block]
                    for horizon in range(self.prospective_future_horizons)
                ]
                generator_kwargs["prospective_future_timesteps"] = [
                    shared_timestep[:, -self.num_frame_per_block:]
                    for _ in range(self.prospective_future_horizons)
                ]
            if is_prospective_target:
                generator_kwargs["return_features"] = True

            if call_requires_grad:
                generator_output = self.generator(**generator_kwargs)
            else:
                with torch.no_grad():
                    generator_output = self.generator(**generator_kwargs)

            if len(generator_output) == 3:
                flow_pred, denoised_pred, prospective_metadata = generator_output
            else:
                flow_pred, denoised_pred = generator_output
                prospective_metadata = None

            if require_grad:
                output[:, current_start_frame:current_end_frame] = denoised_pred

            if is_prospective_anchor:
                prospective_draft_metadata = prospective_metadata
            if is_prospective_target:
                horizon_index = window_index - prospective_anchor_window - 1
                prospective_targets[horizon_index] = (
                    prospective_metadata["last_chunk_features"].detach(),
                    flow_pred[:, -self.num_frame_per_block:].detach(),
                )
                if len(prospective_targets) == self.prospective_future_horizons:
                    self._finalize_prospective_loss(prospective_draft_metadata, prospective_targets)


            # update noisy_cache, which is detached from the computation graph
            with torch.no_grad():
                for block_idx in range(start_block, end_block + 1):
                    
                    block_time_step = current_timestep[:, 
                                    (block_idx - start_block)*self.num_frame_per_block : 
                                    (block_idx - start_block+1)*self.num_frame_per_block].mean().item()
                    matches = torch.abs(self.denoising_step_list - block_time_step) < 1e-4
                    block_timestep_index = torch.nonzero(matches, as_tuple=True)[0]

                    if block_timestep_index == len(self.denoising_step_list) - 1:
                        continue
                    
                    next_timestep = self.denoising_step_list[block_timestep_index + 1].to(noise.device)

                    noisy_cache[:, block_idx * self.num_frame_per_block:
                                    (block_idx+1) * self.num_frame_per_block] = \
                        self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])[:, (block_idx - start_block)*self.num_frame_per_block:
                                                                    (block_idx - start_block+1)*self.num_frame_per_block]


            # rerun with timestep zero to update the clean cache, which is also detached from the computation graph
            with torch.no_grad():
                context_timestep = torch.ones_like(current_timestep) * self.context_noise
                # # add context noise
                # denoised_pred = self.scheduler.add_noise(
                #     denoised_pred.flatten(0, 1),
                #     torch.randn_like(denoised_pred.flatten(0, 1)),
                #     context_timestep * torch.ones(
                #         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                # ).unflatten(0, denoised_pred.shape[:2])

                # only cache the first block
                denoised_pred = denoised_pred[:,:self.num_frame_per_block]
                context_timestep = context_timestep[:,:self.num_frame_per_block]
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    updating_cache=True,
                )

        # Step 3.5: Return the denoised timestep
        # can ignore since not used
        denoised_timestep_from, denoised_timestep_to = None, None

        return output, denoised_timestep_from, denoised_timestep_to



    def inference_with_self_forcing(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        self.last_prospective_loss = noise.new_zeros(())
        self.last_prospective_log_dict = {}

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
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # if self.kv_cache_clean is None:
        #     self._initialize_kv_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device,
        #     )
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device
        #     )
        # else:
        #     # reset cross attn cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index]["is_init"] = False
        #     # reset kv cache
        #     for block_index in range(len(self.kv_cache_clean)):
        #         self.kv_cache_clean[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache_clean[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache_clean,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache_clean,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache_clean,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    updating_cache=True,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_clean = []

        for _ in range(self.num_transformer_blocks):
            kv_cache_clean.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
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

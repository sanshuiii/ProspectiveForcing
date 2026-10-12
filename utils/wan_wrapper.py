import types
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.prospective import (
    ProspectiveParallelContextFutureDraft,
    ProspectiveFutureDraft,
    ProspectiveParallelFutureDraft,
    ProspectiveParallelFutureSuffixDraft,
    ProspectiveParallelFutureWindowDraft,
    ProspectiveParallelMultiLayerFutureDraft,
)
from wan.modules.prospective_confidence import M1ConfidenceController
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            prospective_enabled=False,
            prospective_draft_architecture="recurrent_shared",
            prospective_num_heads=12,
            prospective_mlp_ratio=2.0,
            prospective_future_horizons=2,
            prospective_num_frames_per_chunk=3,
            prospective_context_num_chunks=1,
            prospective_output_num_chunks=1,
            prospective_feature_tap_layers=(5, 16, 27, 40),
            prospective_window_num_chunks=5,
            prospective_init_seed=0,
            m1_method=None,
            m1_delta=0.12,
            m1_tau_run=0.5,
            m1_tau_accept=0.5,
            m1_candidate_seed=20260816,
    ):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")
        self.model.eval()
        self.prospective_enabled = bool(prospective_enabled)
        self.prospective_draft_architecture = str(prospective_draft_architecture)
        self.prospective_future_horizons = int(prospective_future_horizons)
        self.prospective_num_frames_per_chunk = int(prospective_num_frames_per_chunk)
        self.prospective_context_num_chunks = int(prospective_context_num_chunks)
        self.prospective_output_num_chunks = int(prospective_output_num_chunks)
        self.prospective_feature_tap_layers = tuple(
            int(layer) for layer in prospective_feature_tap_layers
        )
        if self.prospective_enabled:
            if not is_causal:
                raise ValueError("Prospective Forcing is only defined for the causal generator")
            draft_classes = {
                "recurrent_shared": ProspectiveFutureDraft,
                "parallel_heads": ProspectiveParallelFutureDraft,
                "parallel_context_heads": ProspectiveParallelContextFutureDraft,
                "parallel_suffix_heads": ProspectiveParallelFutureSuffixDraft,
                "parallel_multilayer_heads": ProspectiveParallelMultiLayerFutureDraft,
                "parallel_window_heads": ProspectiveParallelFutureWindowDraft,
            }
            if self.prospective_draft_architecture not in draft_classes:
                raise ValueError(
                    "unsupported prospective_draft_architecture: "
                    f"{self.prospective_draft_architecture!r}"
                )
            draft_kwargs = dict(
                dim=self.model.dim,
                num_heads=prospective_num_heads,
                mlp_ratio=prospective_mlp_ratio,
                max_horizons=self.prospective_future_horizons,
                num_frames_per_chunk=self.prospective_num_frames_per_chunk,
            )
            if self.prospective_draft_architecture == "parallel_window_heads":
                draft_kwargs["window_num_chunks"] = prospective_window_num_chunks
            elif self.prospective_draft_architecture == "parallel_context_heads":
                draft_kwargs["context_num_chunks"] = (
                    self.prospective_context_num_chunks
                )
            elif self.prospective_draft_architecture == "parallel_suffix_heads":
                draft_kwargs["output_num_chunks"] = self.prospective_output_num_chunks
            elif self.prospective_draft_architecture == "parallel_multilayer_heads":
                draft_kwargs["feature_tap_layers"] = (
                    self.prospective_feature_tap_layers
                )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(prospective_init_seed))
                self.prospective_draft = draft_classes[self.prospective_draft_architecture](
                    **draft_kwargs
                )

        self.m1_method = (
            None if m1_method in (None, "", "none")
            else str(m1_method).upper()
        )
        if self.m1_method is not None:
            if not self.prospective_enabled or not is_causal:
                raise ValueError("M1 requires the causal native Prospective Forcing draft")
            if self.prospective_draft_architecture != "parallel_heads":
                raise ValueError("M1 is defined only for native parallel-head A1")
            self.m1_controller = M1ConfidenceController(
                dim=self.model.dim,
                method=self.m1_method,
                delta=m1_delta,
                tau_run=m1_tau_run,
                tau_accept=m1_tau_accept,
                candidate_seed=m1_candidate_seed,
            )


        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        This follows directly by rearranging the flow-matching equation.
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        updating_cache: Optional[bool] = False,
        return_features: bool = False,
        prospective_future_noises: Optional[List[torch.Tensor]] = None,
        prospective_future_timesteps: Optional[List[torch.Tensor]] = None,
        prospective_interval_index: int = 0,
        m1_force_execute: bool = False,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        metadata = None
        need_features = return_features or prospective_future_noises is not None
        if prospective_future_noises is not None:
            if not self.prospective_enabled:
                raise ValueError("prospective_future_noises requires prospective_enabled=True")
            if kv_cache is None:
                raise ValueError("Prospective Forcing drafts require the causal KV-cache path")
            if prospective_future_timesteps is None or len(prospective_future_noises) != len(prospective_future_timesteps):
                raise ValueError("future noises and timesteps must have matching lengths")
            if not 1 <= len(prospective_future_noises) <= self.prospective_future_horizons:
                raise ValueError(
                    "future-noise count must be between one and "
                    "prospective_future_horizons"
                )

        # X0 prediction
        if kv_cache is not None:
            model_output = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                updating_cache=updating_cache,
                return_features=need_features,
                prospective_draft=getattr(self, "prospective_draft", None),
                m1_controller=getattr(self, "m1_controller", None),
                prospective_future_noises=prospective_future_noises,
                prospective_future_timesteps=prospective_future_timesteps,
                prospective_num_frames_per_chunk=self.prospective_num_frames_per_chunk,
                prospective_interval_index=prospective_interval_index,
                m1_force_execute=m1_force_execute,
            )
            if need_features:
                flow_pred, metadata = model_output
            else:
                flow_pred = model_output
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        if metadata is not None:
            return flow_pred, pred_x0, metadata
        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

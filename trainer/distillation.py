import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import PrecomputedTextEmbeddingDataset, TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
from safetensors.torch import load_file as load_safetensors
from utils.checkpoint_compat import normalize_checkpoint_keys
from torch.utils.tensorboard import SummaryWriter
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        expected_world_size = getattr(config, "expected_world_size", None)
        if expected_world_size is not None and \
                self.world_size != int(expected_world_size):
            raise RuntimeError(
                f"Expected {expected_world_size} distributed ranks, "
                f"got {self.world_size}"
            )
        if getattr(config, "require_full_shard", False) and \
                config.sharding_strategy != "full":
            raise RuntimeError(
                f"This experiment requires FULL_SHARD, "
                f"got {config.sharding_strategy}"
            )

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal

        if getattr(config, "require_fixed_seed", False) and config.seed == 0:
            raise ValueError("A non-zero fixed seed is required for this experiment")

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process:
            self.writer = SummaryWriter(
                log_dir=os.path.join(config.logdir, "tensorboard"),
                flush_secs=10
            )

        self.output_path = config.logdir

        if getattr(config, "require_flash_attention", False):
            import flash_attn
            if self.is_main_process:
                print(f"FlashAttention {flash_attn.__version__} is available")

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        generator_force_wrap_classes = None
        if hasattr(self.model.generator, "prospective_draft"):
            generator_force_wrap_classes = (type(self.model.generator.prospective_draft),)
        generator_ignored_modules = None
        if hasattr(self.model.generator, "m1_controller"):
            controller = self.model.generator.m1_controller
            # ignored_modules are not moved by FSDP.  Place the replicated
            # controller explicitly on this rank's CUDA device before the
            # parent is wrapped; otherwise its interval embedding remains on
            # CPU while generator states are on CUDA at the first train step.
            controller.to(device=self.device, dtype=torch.float32)
            # The controller is used both inside generator.forward() and by
            # the aligned auxiliary loss after that forward returns. Keep its
            # tiny parameter set replicated instead of exposing FSDP shards
            # across those two execution boundaries. Gradients are explicitly
            # synchronized immediately after backward below.
            generator_ignored_modules = (controller,)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            force_wrap_module_classes=generator_force_wrap_classes,
            # Wan 1.3B blocks have 46.4M parameters, just below the former
            # 50M default. Wrap them individually to bound reduce-scatter
            # workspace without changing the model or optimization objective.
            min_num_params=getattr(config, "generator_fsdp_min_num_params", int(5e7)),
            backward_prefetch=getattr(config, "generator_backward_prefetch", "pre"),
            ignored_modules=generator_ignored_modules,
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "real_score_cpu_offload", False)
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "fake_score_cpu_offload", False)
        )

        self.use_precomputed_text_embeddings = bool(
            getattr(config, "precomputed_text_embeddings", None)
        )
        if self.use_precomputed_text_embeddings:
            if self.model.text_encoder is not None:
                raise RuntimeError(
                    "Text Encoder must not be instantiated for precomputed training"
                )
            if self.is_main_process:
                print("Using precomputed text embeddings; Text Encoder is not loaded")
        else:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
            )

        if (not getattr(config, "vae_cpu_offload", False)) and \
                (not config.no_visualize or config.load_raw_video):
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.use_precomputed_text_embeddings:
            if self.config.i2v:
                raise ValueError(
                    "Precomputed text-only embeddings do not support i2v training"
                )
            if int(config.batch_size) != 1:
                raise ValueError(
                    "Variable-length embeddings require per-rank batch_size=1"
                )
            dataset = PrecomputedTextEmbeddingDataset(
                config.precomputed_text_embeddings
            )
            negative_path = os.path.join(
                config.precomputed_text_embeddings,
                "negative_prompt.safetensors"
            )
            negative_prompt_embeds = load_safetensors(negative_path)["prompt_embeds"]
            if negative_prompt_embeds.dtype != torch.bfloat16:
                raise TypeError(
                    "Negative prompt embedding must be bfloat16, "
                    f"got {negative_prompt_embeds.dtype}"
                )
            self.unconditional_dict = {
                "prompt_embeds": negative_prompt_embeds.unsqueeze(0).to(
                    device=self.device, dtype=self.dtype
                )
            }
        elif self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            # Prompt-only Prospective Forcing initializes CUDA before constructing this
            # loader. Allow it to disable worker fork without changing the
            # sampler, sample order, batch, or optimization objective.
            num_workers=getattr(config, "dataloader_num_workers", 8),
            pin_memory=self.use_precomputed_text_embeddings)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            checkpoint = torch.load(config.generator_ckpt, map_location="cpu")
            requested_field = getattr(config, "generator_ckpt_field", None)
            if requested_field is not None:
                if requested_field not in checkpoint:
                    raise RuntimeError(
                        f"requested checkpoint field is absent: {requested_field}"
                    )
                state_dict = checkpoint[requested_field]
                print(f"Loading checkpoint field: {requested_field}")
            elif "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif "generator_ema" in checkpoint:
                state_dict = checkpoint["generator_ema"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Released Causal Consistency Distillation checkpoints store the
            # EMA generator under FSDP wrapper-qualified parameter names.
            # Normalize serialization-only wrappers without changing weights.
            state_dict = normalize_checkpoint_keys(state_dict)

            incompatible = self.model.generator.load_state_dict(
                state_dict, strict=False
            )
            require_complete_prospective = bool(
                getattr(config, "require_complete_prospective_checkpoint", False)
            )
            adaptive_pdd2 = bool(
                getattr(
                    getattr(config, "model_kwargs", {}),
                    "adaptive_pdd2_enabled",
                    False,
                )
            )
            m1_method = getattr(
                getattr(config, "model_kwargs", {}),
                "m1_method",
                None,
            )
            if require_complete_prospective:
                allowed_missing_prefixes = []
                if adaptive_pdd2:
                    allowed_missing_prefixes.append(
                        "prospective_draft.pdd_controller."
                    )
                if m1_method not in (None, "", "none"):
                    allowed_missing_prefixes.append("m1_controller.")
                allowed_missing_prefixes = tuple(allowed_missing_prefixes)
                invalid_missing = [
                    key for key in incompatible.missing_keys
                    if not key.startswith(allowed_missing_prefixes)
                ]
                if m1_method not in (None, "", "none"):
                    m1_missing = [
                        key for key in incompatible.missing_keys
                        if key.startswith("m1_controller.")
                    ]
                    if not m1_missing:
                        raise RuntimeError(
                            "M1 source load did not expose declared new keys"
                        )
                if adaptive_pdd2 and not incompatible.missing_keys:
                    print("Adaptive-PDD-2 parameters restored from checkpoint")
                elif adaptive_pdd2 and self.is_main_process:
                    print(
                        "Initialized only approved Adaptive-PDD-2 keys: "
                        f"{incompatible.missing_keys}"
                    )
            else:
                invalid_missing = [
                    key for key in incompatible.missing_keys
                    if "prospective_draft." not in key
                ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"checkpoint mismatch: missing={invalid_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            if (
                incompatible.missing_keys
                and self.is_main_process
                and not require_complete_prospective
            ):
                print(f"Initialized Prospective Forcing keys: {incompatible.missing_keys}")

        ##############################################################################################################

        # Initialize EMA after loading the requested generator checkpoint. This
        # matters for Stage 2: its EMA must start from the Stage 1 EMA weights,
        # not from the model parameters that existed before checkpoint loading.
        if (
            ema_weight is not None
            and ema_weight > 0.0
            and self.step >= config.ema_start_step
        ):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(
                self.model.generator, decay=ema_weight
            )

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        if self.use_precomputed_text_embeddings:
            prompt_embeds = batch["prompt_embeds"]
            if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != batch_size:
                raise ValueError(
                    f"Unexpected prompt embedding batch shape: {tuple(prompt_embeds.shape)}"
                )
            conditional_dict = {
                "prompt_embeds": prompt_embeds.to(
                    device=self.device, dtype=self.dtype, non_blocking=True
                )
            }
            negative_prompt_embeds = self.unconditional_dict["prompt_embeds"]
            if batch_size != negative_prompt_embeds.shape[0]:
                negative_prompt_embeds = negative_prompt_embeds.expand(
                    batch_size, -1, -1
                )
            unconditional_dict = {"prompt_embeds": negative_prompt_embeds}
        else:
            with torch.no_grad():
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)

                if not getattr(self, "unconditional_dict", None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {k: v.detach()
                                          for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict
                else:
                    unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss_kwargs = dict(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
            )
            m1_method = getattr(
                getattr(self.config, "model_kwargs", {}),
                "m1_method",
                None,
            )
            if m1_method not in (None, "", "none"):
                generator_loss_kwargs["training_step"] = self.step
            generator_loss, generator_log_dict = self.model.generator_loss(
                **generator_loss_kwargs
            )

            if "prospective_loss" in generator_log_dict:
                if self.is_main_process:
                    prospective_metrics = {
                        name: float(generator_log_dict[name].mean())
                        for name in ("prospective_feature_loss", "prospective_flow_loss", "prospective_loss")
                    }
                    print(f"Prospective Forcing pre-backward metrics: {prospective_metrics}")
                # Finish Prospective Forcing feature forwards before releasing cached blocks.
                torch.cuda.synchronize(device=self.device)
                torch.distributed.barrier()
                if self.step == 0:
                    torch.cuda.empty_cache()
                torch.cuda.synchronize(device=self.device)
                torch.distributed.barrier()

            generator_loss.backward()

            if hasattr(self.model.generator, "m1_controller"):
                # FSDP intentionally ignores this replicated controller so it
                # can be called after generator.forward(). Match DDP semantics
                # by averaging every controller gradient across all ranks.
                controller = self.model.generator.m1_controller
                world_size = dist.get_world_size()
                for parameter in controller.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad.div_(world_size)

            if "prospective_loss" in generator_log_dict:
                prospective_grad_sq = torch.zeros((), device=self.device, dtype=torch.float32)
                m1_grad_sq = torch.zeros((), device=self.device, dtype=torch.float32)
                matched_param_count = 0
                grad_param_count = 0
                for name, parameter in self.model.generator.named_parameters():
                    if "prospective_draft." in name:
                        matched_param_count += 1
                        if parameter.grad is not None:
                            grad_param_count += 1
                            prospective_grad_sq.add_(parameter.grad.detach().float().square().sum())
                    if "m1_controller." in name and parameter.grad is not None:
                        m1_grad_sq.add_(
                            parameter.grad.detach().float().square().sum()
                        )
                dist.all_reduce(prospective_grad_sq, op=dist.ReduceOp.SUM)
                dist.all_reduce(m1_grad_sq, op=dist.ReduceOp.SUM)
                prospective_grad_norm = prospective_grad_sq.sqrt()
                if not torch.isfinite(prospective_grad_norm) or prospective_grad_norm <= 0:
                    raise RuntimeError(
                        "Prospective Forcing draft gradient check failed: "
                        f"norm={prospective_grad_norm.item()}, matched={matched_param_count}, "
                        f"with_grad={grad_param_count}"
                    )
                generator_log_dict["prospective_draft_grad_norm"] = prospective_grad_norm
                if hasattr(self.model.generator, "m1_controller"):
                    m1_grad_norm = m1_grad_sq.sqrt()
                    if (
                        not torch.isfinite(m1_grad_norm)
                        or m1_grad_norm <= 0
                    ):
                        raise RuntimeError(
                            f"M1 controller gradient check failed: {m1_grad_norm.item()}"
                        )
                    generator_log_dict["m1_controller_grad_norm"] = m1_grad_norm

            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                # The optimizer has consumed these gradients. Keeping the FSDP
                # gradient shards alive through the following critic-side VAE
                # decode only increases its transient CUDA memory pressure.
                self.generator_optimizer.zero_grad(set_to_none=True)
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                # Logging happens after the critic update, so retain detached
                # values rather than the completed generator autograd graph.
                generator_log_dict = {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in generator_log_dict.items()
                }
                del batch, extra, extras_list
                torch.cuda.empty_cache()

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_log_dict = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in critic_log_dict.items()
            }
            del batch, extra, extras_list

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:

                if TRAIN_GENERATOR:
                    self.writer.add_scalar(
                        "generator_loss",
                        generator_log_dict["generator_loss"].mean().item(),
                        self.step
                    )
                    self.writer.add_scalar(
                        "generator_grad_norm",
                        generator_log_dict["generator_grad_norm"].mean().item(),
                        self.step
                    )
                    self.writer.add_scalar(
                        "dmdtrain_gradient_norm",
                        generator_log_dict["dmdtrain_gradient_norm"].mean().item(),
                        self.step
                    )
                    for metric_name in (
                        "dmd_loss", "prospective_feature_loss", "prospective_flow_loss", "prospective_loss",
                        "prospective_feature_loss_h1", "prospective_feature_loss_h2",
                        "prospective_flow_loss_h1", "prospective_flow_loss_h2",
                        "prospective_draft_grad_norm", "native_prospective_loss",
                        "weighted_m1_loss", "m1_auxiliary_loss",
                        "m1_controller_grad_norm", "candidate_alignment_loss",
                        "oracle_best_error", "oracle_accept_rate",
                        "g1_pre_bce", "g1_pre_brier", "g1_pre_probability",
                        "router_cost_regularization", "c1_post_bce",
                        "c1_post_brier", "c1_post_probability",
                        "verifier_cross_entropy", "verifier_reject_brier",
                        "candidate_cosine", "candidate_distance",
                        "verifier_top1_accuracy",
                        "verifier_reject_rate"
                    ):
                        if metric_name in generator_log_dict:
                            value = generator_log_dict[metric_name].mean().item()
                            self.writer.add_scalar(metric_name, value, self.step)

                self.writer.add_scalar(
                    "critic_loss",
                    critic_log_dict["critic_loss"].mean().item(),
                    self.step
                )
                self.writer.add_scalar(
                    "critic_grad_norm",
                    critic_log_dict["critic_grad_norm"].mean().item(),
                    self.step
                )

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    self.writer.add_scalar(
                        "per iteration time",
                        current_time - self.previous_time,
                        self.step
                    )
                    print(
                        f"Step {self.step} | "
                        f"Iteration time: {current_time - self.previous_time:.2f} seconds | "
                    )
                    self.previous_time = current_time

            max_steps = getattr(self.config, "max_steps", None)
            if max_steps is not None and self.step - start_step >= max_steps:
                if self.is_main_process:
                    self.writer.flush()
                    self.writer.close()
                break

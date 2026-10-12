import argparse
import gc
import json
import torch
import os
import time
from pathlib import Path
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
import imageio
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from utils.prospective_eval_contract import assert_prospective_frame_contract
from utils.m1_checkpoint_load import load_m1_checkpoint
from utils.checkpoint_compat import normalize_checkpoint_keys

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--num_save_frames", type=int, default=None,
                    help="If set, save exactly this many decoded RGB frames")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--deterministic_by_prompt", action="store_true",
                    help="Derive all sample randomness from the base seed and prompt index")
parser.add_argument("--require_prospective", action="store_true",
                    help="Require the PROSPECTIVE-trained architecture and checkpoint weights")
parser.add_argument("--strict_prospective_short_smoke", action="store_true",
                    help="Use the strict 21-latent/80-frame one-prompt PROSPECTIVE smoke contract; "
                         "formal evaluation remains fixed at 126/480")
parser.add_argument("--allow_untrained_m1_controller", action="store_true",
                    help="Allow only missing m1_controller keys for the explicitly "
                         "precalibration source-checkpoint smoke")
parser.add_argument("--m1_force_execute", action="store_true",
                    help="Force the M1 q1/K=2 mechanics during the explicitly "
                         "precalibration source-checkpoint smoke only")
parser.add_argument("--m1_calibration_shadow", action="store_true",
                    help="Force candidates, keep the B0 rollout, collect exact next-window "
                         "oracle errors, and skip RGB decode")
parser.add_argument("--independent_sharding", action="store_true",
                    help="Shard prompts across independent ranks without NCCL communication")
parser.add_argument("--skip_existing", action="store_true",
                    help="Skip an index when all of its expected non-empty output files already exist")
parser.add_argument("--auxiliary_window_heads", type=int, choices=[0, 1, 2],
                    default=0, help="Number of future-window heads to commit")
parser.add_argument(
    "--draft_commit_policy",
    choices=["last_chunk_only_rolling_stitch_v1"],
    default=None,
)
parser.add_argument("--expected_draft_output_num_chunks", type=int, default=None)
parser.add_argument("--instrumentation_folder", type=str, default=None,
                    help="Directory for per-rank inference JSONL")
parser.add_argument("--prompt_indices", type=int, nargs="+", default=None,
                    help="Optional exact prompt indices for process-isolated evaluation")
args = parser.parse_args()
assert_prospective_frame_contract(
    require_prospective=args.require_prospective,
    short_smoke=args.strict_prospective_short_smoke,
    num_output_frames=args.num_output_frames,
    num_save_frames=args.num_save_frames,
)
process_start_time = time.perf_counter()

# Initialize distributed inference
if args.independent_sharding:
    global_rank = int(os.environ["INFERENCE_RANK"])
    world_size = int(os.environ["INFERENCE_WORLD_SIZE"])
    assert 0 <= global_rank < world_size
    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:0")
    set_seed(args.seed)
elif "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    set_seed(args.seed)
else:
    device = torch.device("cuda")
    local_rank = 0
    global_rank = 0
    world_size = 1
    set_seed(args.seed)

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
code_root = Path(__file__).resolve().parent
default_config = OmegaConf.load(code_root / "configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

m1_method = OmegaConf.select(config, "model_kwargs.m1_method")
m1_method = None if m1_method in (None, "", "none") else str(m1_method).upper()
m1_threshold_status = OmegaConf.select(config, "m1_threshold_status")
m1_threshold_status = (
    None if m1_threshold_status in (None, "", "none")
    else str(m1_threshold_status)
)
if args.allow_untrained_m1_controller:
    assert m1_method in {"R1", "M1", "RM1"}
    assert m1_threshold_status == "precalibration_smoke_only"
    assert args.strict_prospective_short_smoke
    assert args.auxiliary_window_heads == 1
if args.m1_force_execute:
    assert args.allow_untrained_m1_controller or args.m1_calibration_shadow
    assert m1_method in {"R1", "RM1"}
if args.m1_calibration_shadow:
    assert not args.allow_untrained_m1_controller
    assert args.auxiliary_window_heads == 1
    assert args.m1_force_execute or m1_method == "M1"
if m1_threshold_status == "calibrated_validation_only":
    assert m1_method in {"R1", "M1", "RM1"}
    assert not args.allow_untrained_m1_controller
    assert not args.m1_force_execute
    assert not args.m1_calibration_shadow
    assert args.auxiliary_window_heads == 1

if args.require_prospective:
    assert args.use_ema, "Strict PROSPECTIVE inference requires --use_ema"
    assert bool(OmegaConf.select(config, "model_kwargs.prospective_enabled")), \
        "model_kwargs.prospective_enabled must be true"
    assert list(config.denoising_step_list) == [1000, 800, 600, 400, 200]
    assert bool(config.warp_denoising_step) is True
    assert float(config.timestep_shift) == 5.0
    assert float(config.guidance_scale) == 3.0
    assert int(config.num_frame_per_block) == 3
    assert int(config.height) == 480 and int(config.width) == 832
    assert args.num_samples == 1
    assert args.save_with_index
    assert args.deterministic_by_prompt
    assert args.independent_sharding
    if args.prompt_indices is None:
        assert world_size >= 1
    else:
        assert min(args.prompt_indices) >= 0
        assert max(args.prompt_indices) < 200
        assert len(set(args.prompt_indices)) == len(args.prompt_indices)
        assert world_size == 1
        assert len(args.prompt_indices) == 1
    assert args.instrumentation_folder is not None
    prediction_scope = str(
        OmegaConf.select(config, "prospective_prediction_scope") or "chunk"
    )
    output_num_chunks = (
        len(config.denoising_step_list)
        if prediction_scope == "window"
        else int(
            OmegaConf.select(config, "model_kwargs.prospective_output_num_chunks")
            or 1
        )
        if prediction_scope == "suffix"
        else 1
    )
    assert 1 <= output_num_chunks <= len(config.denoising_step_list)
    if args.auxiliary_window_heads:
        assert args.draft_commit_policy == (
            "last_chunk_only_rolling_stitch_v1"
        )
        assert args.expected_draft_output_num_chunks == output_num_chunks
        print(
            "Strict Prospective Forcing last-chunk Rolling-stitch evaluation enabled "
            f"with prediction_scope={prediction_scope}, "
            f"computed_output_chunks={output_num_chunks}, "
            f"auxiliary_window_heads={args.auxiliary_window_heads}; "
            "only the final entering chunk from each q head is committed."
        )
    else:
        assert prediction_scope in {"chunk", "suffix", "window"}
        print(
            "Strict Prospective Forcing B0 standard evaluation enabled; "
            "auxiliary draft heads are disabled."
        )

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

checkpoint_load_audit = None
if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if args.require_prospective:
        assert 'generator_ema' in state_dict, "Checkpoint has no generator_ema"
        assert isinstance(state_dict['generator_ema'], dict)
    if args.use_ema:
        state_dict_to_load = state_dict['generator_ema']
        def remove_fsdp_prefix(state_dict):
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if "_fsdp_wrapped_module." in key:
                    new_key = key.replace("_fsdp_wrapped_module.", "")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict
        state_dict_to_load = normalize_checkpoint_keys(
            remove_fsdp_prefix(state_dict_to_load)
        )
    else:
        state_dict_to_load = state_dict['generator']
    if args.require_prospective:
        assert any(key.startswith('prospective_draft.') for key in state_dict_to_load), \
            "generator_ema does not contain PROSPECTIVE draft weights"
    if m1_method is None:
        pipeline.generator.load_state_dict(state_dict_to_load)
    else:
        checkpoint_load_audit = load_m1_checkpoint(
            module=pipeline.generator,
            state_dict=state_dict_to_load,
            method=m1_method,
            allow_untrained_controller=args.allow_untrained_m1_controller,
        )
    del state_dict_to_load, state_dict
    gc.collect()

if args.require_prospective:
    assert hasattr(pipeline.generator, 'prospective_draft'), \
        "Inference architecture has no PROSPECTIVE draft module"
    assert pipeline.generator.prospective_draft is not None, \
        "PROSPECTIVE draft module was not instantiated"
    print("Verified PROSPECTIVE architecture and generator_ema draft weights.")

pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
torch.cuda.synchronize(device)
model_cold_start_seconds = time.perf_counter() - process_start_time

# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if args.independent_sharding:
    if args.prompt_indices is None:
        sampler = list(range(global_rank, num_prompts, world_size))
    else:
        requested_indices = sorted(args.prompt_indices)
        assert requested_indices[-1] < num_prompts
        sampler = requested_indices[global_rank::world_size]
elif dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
os.makedirs(args.output_folder, exist_ok=True)
instrumentation_path = None
if args.instrumentation_folder is not None:
    os.makedirs(args.instrumentation_folder, exist_ok=True)
    instrumentation_rank = int(
        os.environ.get("INSTRUMENTATION_RANK", global_rank)
    )
    instrumentation_path = os.path.join(
        args.instrumentation_folder, f"rank_{instrumentation_rank:03d}.jsonl"
    )

if dist.is_initialized() and not args.independent_sharding:
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(global_rank != 0)):
    idx = batch_data['idx'].item()

    if args.skip_existing:
        variant = "ema" if args.use_ema else "regular"
        expected_paths = [
            os.path.join(args.output_folder, f'{idx}-{sample_idx}_{variant}.mp4')
            for sample_idx in range(args.num_samples)
        ]
        if all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in expected_paths):
            print(f"Skipping existing prompt index {idx}")
            continue
    video_wall_start = time.perf_counter()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.deterministic_by_prompt:
        effective_seed = args.seed + idx
        set_seed(effective_seed)
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(effective_seed)
    else:
        noise_generator = None

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device,
            dtype=torch.bfloat16, generator=noise_generator
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device,
            dtype=torch.bfloat16, generator=noise_generator
        )

    # Generate 81 frames
    video, latents = pipeline.inference_rolling_forcing(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        auxiliary_window_heads=args.auxiliary_window_heads,
        m1_force_execute=(args.m1_force_execute or args.m1_calibration_shadow),
        m1_calibration_shadow=args.m1_calibration_shadow,
        decode_output=not args.m1_calibration_shadow,
    )
    inference_stats = dict(pipeline.last_inference_stats)
    if not args.m1_calibration_shadow:
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    if not args.m1_calibration_shadow:
        video = torch.cat(all_video, dim=1)
        if args.num_save_frames is not None:
            assert video.shape[1] >= args.num_save_frames, \
                f"Decoded only {video.shape[1]} frames; cannot save {args.num_save_frames}"
            video = video[:, :args.num_save_frames]
        if args.require_prospective:
            assert tuple(video.shape[1:4]) == (args.num_save_frames, 480, 832), \
                f"Unexpected decoded video shape: {tuple(video.shape)}"
        video = (255.0 * video).clamp(0, 255).to(torch.uint8)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts and not args.m1_calibration_shadow:
        model = "regular" if not args.use_ema else "ema"
        write_wall_start = time.perf_counter()
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
            # imageio.mimwrite(output_path, video[seed_idx], fps=16, quality=8, output_params=["-loglevel", "error"])
        video_write_seconds = time.perf_counter() - write_wall_start
    else:
        video_write_seconds = 0.0
    saved_rgb_frames = 0 if args.m1_calibration_shadow else int(video.shape[1])

    if instrumentation_path is not None:
        record = {
            **inference_stats,
            "index": idx,
            "effective_seed": (
                args.seed + idx if args.deterministic_by_prompt else None
            ),
            "rank": global_rank,
            "world_size": world_size,
            "model_cold_start_seconds": model_cold_start_seconds,
            "video_write_seconds": video_write_seconds,
            "end_to_end_seconds": time.perf_counter() - video_wall_start,
            "generated_latent_frames": int(latents.shape[1]),
            "saved_rgb_frames": saved_rgb_frames,
            "fps": 16,
            "inference_mode": (
                f"M1_{m1_method}_CALIBRATION64_B0_SHADOW"
                if args.m1_calibration_shadow
                else f"M1_{m1_method}_COMMON200_CALIBRATED"
                if m1_threshold_status == "calibrated_validation_only"
                else f"M1_{m1_method}_A1_5MIN_PRECALIBRATION_SMOKE"
                if m1_method is not None
                else {
                    0: "B0_STANDARD",
                    1: "A1_ONE_WINDOW",
                    2: "A2_TWO_WINDOWS",
                }[args.auxiliary_window_heads]
            ),
            "m1_method": m1_method,
            "m1_threshold_status": m1_threshold_status,
            "m1_checkpoint_load_audit": checkpoint_load_audit,
            "m1_force_execute": args.m1_force_execute,
            "m1_calibration_shadow": args.m1_calibration_shadow,
            "approximate": (
                bool(args.auxiliary_window_heads)
                and not args.m1_calibration_shadow
            ),
        }
        with open(instrumentation_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    if args.m1_calibration_shadow:
        del latents, sampled_noise, all_video
    else:
        del video, current_video, latents, sampled_noise, all_video
    gc.collect()
    torch.cuda.empty_cache()

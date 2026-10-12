# Inference

B0 and A1 load the Stage 1 EMA. M1 loads the Stage 2 EMA. All three use five
denoising levels, timestep shift 5.0, guidance scale 3.0, three latent frames per
chunk, and deterministic `base_seed + prompt_index` sampling.

## B0

B0 runs the standard rolling generator and commits no prospective draft. The
draft weights remain present in the checkpoint but do not affect the rollout.

## A1

A1 requests q1 at each eligible roll and applies
`last_chunk_only_rolling_stitch_v1`. Only the final entering latent chunk from
the predicted q1 output is committed; the main rolling context is retained.

## M1

M1 evaluates two q1 candidates through the trained controller. The verifier
selects candidate 1, candidate 2, or reject. Reject preserves the B0 path. The
formal configuration is `configs/inference/m1.yaml`.

## Output contract

The default launchers generate 126 latent frames, decode at least 480 RGB
frames, retain the first 480, and write 16-fps 832×480 MP4 files. Per-rank JSONL
instrumentation is written next to the videos. A production evaluation should
add full ffmpeg/PyAV decode validation and file hashes before accepting outputs.

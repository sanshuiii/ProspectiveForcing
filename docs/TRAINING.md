# Training

## Stage 1

`configs/training/stage1_cm_init.yaml` is the authoritative Stage 1 contract.
It initializes the causal student from `causal_cd.pt`, uses the Wan2.1 14B
real-score teacher, and trains for 3,000 optimizer-loop steps with eight ranks,
batch 1 per rank, full sharding, and seed 20260720. The denoising schedule is
`[1000, 800, 600, 400, 200]`.

An optimizer-loop step updates the score model once. The generator update ratio
is five, so the 3,000-loop Stage 1 run performs 600 generator updates and 3,000
score-model updates.

The final file is normally
`runs/stage1/logs/checkpoint_model_003000/model.pt`. B0/A1 read the
`generator_ema` field from this file.

## Stage 2

`configs/training/stage2_m1.yaml` loads the Stage 1 `generator_ema`, attaches the
M1 K=2 controller/verifier, and runs 1,000 additional optimizer-loop steps.
Stage 2 continues the registered DMD generator/critic schedule while learning
the controller/verifier; it is not a controller-only freeze of the backbone.
The 1,000-loop run performs 200 generator updates and 1,000 score-model
updates.
The data order, batch, sharding, seed, denoising schedule, and base losses remain
aligned with Stage 1. For the candidate feature-alignment loss, the best
candidate receives weight `1 - beta` and the remaining candidates share weight
`beta`, with `beta=0.05`. The verifier target accepts the lowest-error candidate
when its normalized feature error is at most `delta=0.12`; otherwise, it uses
the rejection class. Stage 2 uses candidate seed 20260816 and auxiliary loss
weight `lambda_aux=0.1`. Its verifier objective is cross-entropy plus the Brier
loss on the rejection probability; there is no separate candidate-diversity
loss.

The native future feature and velocity alignment terms use the lowest-error
candidate for each sample, as specified by the paper. Candidate ranking uses
the normalized feature error to the detached backbone target.

The final file is normally
`runs/stage2/logs/checkpoint_model_001000/model.pt`. M1 inference reads its
`generator_ema` field.

## Cached text conditions

Formal training expects precomputed BF16 UMT5 embeddings. The directory schema
is documented in `data/README.md`. `prepare_embeddings.py` creates exactly that
schema. Keep the prompt file, manifest, index, and all shards together.

## Distributed contract

The published launchers default to one node with eight visible GPUs. They also
support a two-node/four-GPU-per-node layout through the following variables:

```bash
export PF_NNODES=2
export PF_NPROC_PER_NODE=4
export PF_NODE_RANK=0              # use 1 on the second node
export PF_MASTER_ADDR=coordinator-host
export PF_MASTER_PORT=29500
bash scripts/training/train_stage1.sh
```

Both nodes run the same command with different node ranks. Preserve eight total
ranks, batch 1 per rank, global batch 8, and FSDP full sharding to match the
reference scientific configuration. Scheduler integration is intentionally
left to the user; no cluster-specific account, partition, or host is embedded.

# Prospective Forcing: full usage guide

Prospective Forcing is a two-stage long-video generation method built on a
causal Rolling Forcing backbone. It first trains lightweight future-feature
draft heads jointly with the DMD generator. Stage 2 resumes the Stage 1 EMA,
continues the registered generator/critic update schedule, and learns an M1
controller that proposes two q1 candidates and selects one with a learned
verifier. The public release exposes three inference modes:

- **B0** uses the Stage 1 EMA with the standard rolling path.
- **A1** uses the same Stage 1 EMA and commits one q1 draft with a
  last-chunk-only rolling stitch.
- **M1** uses the Stage 2 EMA, evaluates two learned candidates, and may reject
  both candidates back to the B0 path.

## Repository layout

```text
configs/training/       Stage 1 CM-initialized training and Stage 2 M1 training
configs/inference/      B0, A1, and M1 inference contracts
model/ trainer/         DMD model and distributed trainers
pipeline/               Rolling training and inference pipelines
wan/                    Modified Wan causal backbone
utils/                  Data, distributed, checkpoint, and model wrappers
scripts/training/       Executable two-stage training entry points
scripts/inference/      Executable B0/A1/M1 inference entry points
scripts/utilities/      Embedding, environment, and checkpoint validation tools
checkpoints/            Empty artifact locations; weights are not in this release
docs/                   Method, training, inference, and configuration details
tests/                  CPU-safe contract and checkpoint-compatibility tests
```

## Environment

The reference environment uses Linux, Python 3.10, PyTorch 2.5.1, CUDA 12.4,
and FlashAttention 2.8.3.post1. Stage 1 and Stage 2 use eight distributed ranks,
batch size 1 per rank, and FSDP full sharding.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements/training.txt
python scripts/utilities/verify_environment.py
```

This repository disables Weights & Biases in every published launcher. No API
key, host name, user name, cluster path, scheduler account, or telemetry
endpoint is embedded in the release.

## Required artifacts

| Artifact | Default location | Source |
|---|---|---|
| Wan2.1-T2V-1.3B | `wan_models/Wan2.1-T2V-1.3B` | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-T2V-14B | `wan_models/Wan2.1-T2V-14B` | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| causal-model initialization | `checkpoints/base/causal_cd.pt` | [Causal Forcing checkpoint](https://huggingface.co/zhuhz22/Causal-Forcing/blob/a6a8f0e3bbdea1044fc6fef09c9cb9f648bf1bc3/chunkwise/causal_cd.pt) |
| precomputed text embeddings | `data/embeddings` | Prepare with the included utility or provide an equivalent artifact |
| Stage 1 checkpoint | `checkpoints/stage1/prospective_forcing_stage1.pt` | Train locally or insert the separately released checkpoint |
| Stage 2 checkpoint | `checkpoints/stage2/prospective_forcing_stage2_m1.pt` | Train locally or insert the separately released checkpoint |

The machine-readable list is `checkpoints/artifact_manifest.json`. A helper is
also provided:

```bash
bash scripts/utilities/download_public_artifacts.sh
```

Weights are deliberately excluded from this source-only package.

## Prepare text embeddings

Store one training prompt per line in `data/train_prompts.txt`, then run:

```bash
PYTHONPATH=. python scripts/utilities/prepare_embeddings.py \
  --prompts data/train_prompts.txt \
  --output-dir data/embeddings \
  --negative-prompt "low quality, static, artifacts"
```

The resulting directory contains `manifest.json`, `index.jsonl`, BF16
SafeTensor shards, the selected prompt list, and the negative-prompt embedding.

## Two-stage training

Stage 1 starts from the causal-model initialization checkpoint and performs
3,000 optimizer-loop steps:

```bash
bash scripts/training/train_stage1.sh
```

Place or link its final checkpoint at
`checkpoints/stage1/prospective_forcing_stage1.pt`, then run Stage 2 for 1,000
M1 post-training steps:

```bash
bash scripts/training/train_stage2.sh
```

Each loop step updates the score model. With a generator update ratio of five,
the two stages perform 600 and 200 generator updates, respectively.

To run both stages consecutively using the default run locations:

```bash
bash scripts/training/train_all_stages.sh
```

The launchers default to one node with eight GPUs. For the reference two-node
layout, run the same command on both nodes with `PF_NNODES=2`,
`PF_NPROC_PER_NODE=4`, a distinct `PF_NODE_RANK`, and shared
`PF_MASTER_ADDR`/`PF_MASTER_PORT`. The scripts reject layouts that do not total
eight ranks.

## B0, A1, and M1 inference

Each prompt file contains one prompt per line. Eight GPUs are used by default;
set `PF_GPU_IDS=0` for one GPU or, for example, `PF_GPU_IDS=0,2,5,7` for four
independent workers. The seed is derived as `base_seed + prompt_index`, so
sharding does not change a sample's seed.

```bash
bash scripts/inference/infer_b0.sh examples/prompts.txt outputs/b0
bash scripts/inference/infer_a1.sh examples/prompts.txt outputs/a1
bash scripts/inference/infer_m1.sh examples/prompts.txt outputs/m1
```

The default formal generation contract is 126 latent frames, 480 saved RGB
frames, 16 fps, and 832×480. See [docs/INFERENCE.md](INFERENCE.md) for the
exact graph differences.

## Configuration

All paths can be overridden without editing tracked files:

| Variable | Meaning |
|---|---|
| `PF_BASE_CHECKPOINT` | causal-model initialization used by Stage 1 |
| `PF_STAGE1_CHECKPOINT` | Stage 1 EMA used by Stage 2, B0, and A1 |
| `PF_STAGE2_CHECKPOINT` | Stage 2 EMA used by M1 |
| `PF_TRAIN_PROMPTS` | one-prompt-per-line training file |
| `PF_EMBEDDINGS_DIR` | precomputed BF16 embedding directory |
| `PF_GPU_IDS` | comma-separated inference GPU IDs |
| `PF_SEED` | inference base seed |

Read [docs/CONFIGURATION.md](CONFIGURATION.md) before changing scientific
hyperparameters.

## Checkpoint compatibility

The public implementation uses `prospective_*` and `m1_*` module names. The
loader recognizes two corresponding legacy state-dict prefixes and translates
keys only; tensor values are unchanged. New checkpoints are saved with the
public names.

## Upstream code and license

This code derives from Rolling Forcing and Wan2.1. The original license and
required notices are retained. The inherited Rolling Forcing license restricts
use to academic purposes; read `LICENSE` before use. Additional provenance and
links are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

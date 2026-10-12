<div align="center">

<h1>Prospective Forcing:<br>Learning to Look Ahead for Autoregressive Long Video Diffusion</h1>

<p>
  <strong>Yifan Shen</strong><sup>1,3†</sup>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>Yewei Xia</strong><sup>1</sup>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>Gongxu Luo</strong><sup>1,3</sup>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>Peiyuan Zhu</strong><sup>1</sup>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>Boyang Sun</strong><sup>1</sup>
  <br>
  <strong>Zhenhao Chen</strong><sup>1</sup>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>Guangyi Liu</strong><sup>1,3</sup>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>Guangyi Chen</strong><sup>1,2*</sup>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>Kun Zhang</strong><sup>2,1</sup>
</p>

<p>
  <sup>1</sup> Mohamed bin Zayed University of AI
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sup>2</sup> Carnegie Mellon University
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sup>3</sup> Institute of Foundation Models
</p>

<p>
  <sup>†</sup> Work done during an internship at the Institute of Foundation Models.
  &nbsp;&nbsp;
  <sup>*</sup> Corresponding author.
</p>

<p>
  <a href="https://sanshuiii.github.io/ProspectiveForcing/"><strong>Project Page</strong></a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <strong>arXiv</strong> (coming soon)
</p>

</div>

---

<a href="https://sanshuiii.github.io/ProspectiveForcing/">
  <img src="assets/pv_poster.jpg" alt="Prospective Forcing video: watch on the project page" width="100%">
</a>

## Quick start

1. Create the reference environment (Linux, Python 3.10, PyTorch 2.5.1, CUDA 12.4, FlashAttention 2.8.3.post1):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements/training.txt
python scripts/utilities/verify_environment.py
```

2. Download the public base models (Wan2.1-T2V-1.3B / 14B and the causal-model initialization) to their default
   locations, then place the Prospective Forcing checkpoint in `checkpoints/stage2/`
   (see [`checkpoints/README.md`](checkpoints/README.md); release of our weights is coming soon, or train it with
   the two-stage recipe in [`docs/USAGE.md`](docs/USAGE.md)):

```bash
pip install -U "huggingface_hub[cli]"   # provides the `hf` command used by the script
bash scripts/utilities/download_public_artifacts.sh
```

3. Generate videos from a text file with one prompt per line. Eight GPUs are used by default; set `PF_GPU_IDS=0`
   for a single GPU.

```bash
bash scripts/inference/infer_m1.sh examples/prompts.txt outputs/
```

Training, configuration, and checkpoint details are in [`docs/USAGE.md`](docs/USAGE.md) and [`docs/`](docs/).
This code builds on Rolling Forcing and Wan2.1; the inherited license restricts use to academic purposes
(see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

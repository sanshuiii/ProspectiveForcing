# Checkpoint layout

Checkpoint files are intentionally excluded from the source release.

Place artifacts at these default locations:

- `base/causal_cd.pt`: public causal-model initialization checkpoint.
- `stage1/prospective_forcing_stage1.pt`: Stage 1 EMA checkpoint.
- `stage2/prospective_forcing_stage2_m1.pt`: Stage 2 M1 EMA checkpoint.

The training code saves full dictionaries whose inference weights are stored in
the `generator_ema` field. Pre-release checkpoints using the historical module
prefixes are translated at load time without altering tensor values.

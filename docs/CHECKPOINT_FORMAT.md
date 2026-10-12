# Checkpoint format

Training checkpoints are PyTorch dictionaries. The inference model is loaded
from `generator_ema`; `generator`, `critic`, optimizer, scheduler, and step
state may also be present for resumption.

A Stage 1 EMA must contain `prospective_draft.*`. A Stage 2 EMA must contain
both `prospective_draft.*` and `m1_controller.*`. Validate files with:

```bash
PYTHONPATH=. python scripts/utilities/validate_checkpoint.py \
  checkpoints/stage1/prospective_forcing_stage1.pt --stage stage1
PYTHONPATH=. python scripts/utilities/validate_checkpoint.py \
  checkpoints/stage2/prospective_forcing_stage2_m1.pt --stage stage2
```

The compatibility loader maps pre-release prefixes to public prefixes before
validation. It does not modify, cast, reshape, or average tensors.

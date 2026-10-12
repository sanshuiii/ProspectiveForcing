# Method

Prospective Forcing augments a causal long-video diffusion generator with
future-feature prediction. At a full rolling window, lightweight draft heads
predict representations for the next two latent chunks from the current hidden
state and persistent future noise. Training compares each prediction with the
detached backbone representation observed when that physical chunk later
enters the rolling window. The auxiliary objective combines normalized feature
matching and flow prediction while leaving the DMD and critic objectives intact.

Stage 1 jointly trains the generator, critic, and draft heads from a public
causal-model initialization. Rolling and self-forcing rollouts remain mixed by
the underlying training pipeline; the prospective loss is active only on
eligible rolling rollouts.

Stage 2 starts from the Stage 1 EMA and adds M1. M1 calls the q1 head with two
learned candidate embeddings. A learned verifier chooses candidate 1,
candidate 2, or reject. Reject follows the standard B0 rollout. The formal M1
inference path uses hard selection rather than candidate averaging.

The three public inference modes isolate the effect of using the trained draft:

| Mode | Checkpoint | Draft behavior |
|---|---|---|
| B0 | Stage 1 | draft disabled; standard rolling path |
| A1 | Stage 1 | one q1 draft, last entering chunk committed |
| M1 | Stage 2 | two q1 candidates, learned verifier, reject to B0 |

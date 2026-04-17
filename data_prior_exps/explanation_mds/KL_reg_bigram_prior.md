# train_gpt_exp2.py

## Main idea

This experiment tests **KL regularization toward a bigram prior**.

Instead of directly injecting prior logits into the model output, it uses the prior as a target distribution and penalizes the model when its next-token distribution drifts away from that prior.

## What the code is doing

- Adds two new controls: `PRIOR_KL_WEIGHT` and `PRIOR_KL_TEMP`.
- Builds a KL term between the model distribution and the bigram-prior distribution on positions `t > 0`.
- Uses a weighted loss of the form:
  - cross-entropy
  - plus a prior KL penalty scaled by `PRIOR_KL_WEIGHT * prior_alpha`
- Logs the extra loss pieces separately: `train_total`, `train_raw_kl`, `train_kl`, and `train_mtp`.

## Important nuance

In this file, the prior is **not** added directly into the logits the way `exp1` does it. The prior only affects training through the auxiliary KL term.

So the experiment is really:

- no direct prior injection
- yes prior-based regularization

## Why this experiment exists

The hypothesis is that it may be better to encourage the model toward a teacher-like or corpus-derived local distribution than to hard-bias the logits at every step.

That makes the prior softer:

- `exp1` says "use these logits"
- `exp2` says "stay near this distribution"

## Important knobs

- `BIGRAM_PRIOR_PATH`
- `PRIOR_ALPHA_START`
- `PRIOR_MAX_STEP`
- `PRIOR_KL_WEIGHT`
- `PRIOR_KL_TEMP`

## Representative logs in `outputs/`

- `outputs/exp2_kl_sp1024_w001_s500.txt`
- `outputs/exp2_kl_sp1024_w005_s400.txt`

## Note

This is the first file where the experiment-specific logging clearly exposes the prior term as its own training component.

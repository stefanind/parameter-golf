# train_gpt_exp1.py

## Main idea

This experiment tests **bigram prior injection into logit space**.

The goal is to give the model a frozen, hand-built next-token bias from token-to-token statistics, instead of asking the network to learn that short-range structure from scratch.

## What the code is doing

- Loads a bigram prior from `BIGRAM_PRIOR_PATH`.
- Computes a decaying prior strength with `get_alpha(step, PRIOR_MAX_STEP, PRIOR_ALPHA_START)`.
- During the forward pass, looks up the prior for the previous token and **adds it directly to the logits** for positions `t > 0`.
- Applies the model loss after that prior-adjusted logit computation.

In other words, the prior is not trainable here. It is a fixed external bias added to the model's logits during training.

## Why this experiment exists

The hypothesis is that a small model may benefit from being nudged toward strong local token-transition statistics early in training. If that works, the model could spend less capacity relearning common bigram structure.

The decay schedule is important: it tests whether the prior is most useful as an early scaffold rather than a permanent crutch.

## Important knobs

- `BIGRAM_PRIOR_PATH`
- `PRIOR_ALPHA_START`
- `PRIOR_MAX_STEP`

## Representative logs in `outputs/`

- `outputs/exp1_prior_inject_alpha_maxstep_500.txt`
- `outputs/exp1_prior_inject_alpha_maxstep_1000.txt`
- `outputs/exp1_prior_inject_alpha_maxstep_2000.txt`
- `outputs/exp1_prior_inject_alpha_maxstep_10000.txt`

## Note

This file contains a larger training stack than just the prior feature. The experiment-specific change is the **direct logit injection of the frozen bigram prior**.

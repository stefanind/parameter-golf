# train_gpt_exp3.py

## Main idea

This experiment tests a **unigram bias on the LM output head**.

The idea is to give the model a learned per-token bias term at the output layer, optionally initialized from corpus unigram probabilities, so common tokens start with a better base-rate prior.

## What the code is doing

- Adds a learnable parameter `lm_head_bias` with one bias value per vocabulary item.
- Optionally initializes that bias from `UNIGRAM_PROBS_PATH`.
- Converts unigram probabilities into centered log-probabilities and scales them by `UNIGRAM_BIAS_TAU`.
- Adds `lm_head_bias` to the output logits on every forward pass.

This means the output head can represent:

- content-dependent logits from the transformer
- plus a content-independent token popularity bias

## Important nuance

This file also removes the KL-regularization machinery from `exp2`.

It keeps the earlier **bigram prior injection** path from `exp1`, so the main new thing in `exp3` is the unigram output bias, not a completely isolated baseline.

## Why this experiment exists

The hypothesis is that a small model may waste capacity relearning simple token frequency effects. A unigram bias gives the model an explicit place to store that information, which may free the rest of the network to model context-dependent structure.

## Important knobs

- `USE_UNIGRAM_BIAS_INIT`
- `UNIGRAM_PROBS_PATH`
- `UNIGRAM_BIAS_TAU`
- `BIGRAM_PRIOR_PATH`
- `PRIOR_ALPHA_START`
- `PRIOR_MAX_STEP`

## Representative logs in `outputs/`

- `outputs/unigram_bias_exp3.txt`
- `outputs/prior_injection_exp3.txt`
- `outputs/baseline_exp3.txt`

## Note

If you want to think about `exp3` in one sentence: it is **"exp1-style prior injection, plus a learned unigram bias on the output logits."**

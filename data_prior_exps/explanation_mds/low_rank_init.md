# train_gpt_exp4.py

## Main idea

This experiment tests **low-rank co-occurrence initialization**.

The goal is to initialize the token embedding space from a low-rank factorization of a token-token bigram matrix, so the model starts with a geometry that already reflects corpus co-occurrence structure.

## What the code is doing

- Loads a dense token-token matrix from `LOWRANK_BIGRAM_PATH`.
- Supports either:
  - `logp`
  - `centered_logp`
- Runs an SVD on that matrix.
- Uses the low-rank factors to initialize:
  - `tok_emb.weight`
  - and `lm_head.weight` when embeddings are untied
- Optionally pads unused dimensions with small random noise.

This is an **initialization experiment**, not a persistent auxiliary loss or trainable adapter.

## Important nuance

The earlier `exp3` machinery is still present in the file, including unigram-bias support and bigram-prior support. But the defaults change here:

- `PRIOR_ALPHA_START` defaults to `0.0`
- `PRIOR_MAX_STEP` defaults to `2000`

So by default, this file is set up to isolate the low-rank initialization idea rather than actively use prior injection.

## Why this experiment exists

The hypothesis is that a bigram/co-occurrence matrix contains a useful low-dimensional structure that can be baked directly into the embedding space. If that structure is good, the model may train faster or reach better quality than random initialization alone.

## Important knobs

- `USE_LOWRANK_BIGRAM_INIT`
- `LOWRANK_BIGRAM_PATH`
- `LOWRANK_INIT_RANK`
- `LOWRANK_INIT_MATRIX`
- `LOWRANK_INIT_SCALE`
- `LOWRANK_INIT_NOISE_STD`

## Representative logs in `outputs/`

- `outputs/svd_exp4.txt`
- `outputs/baseline_exp4.txt`
- `outputs/prior_injection_exp4.txt`
- `outputs/all_exp4.txt`

## Note

The cleanest summary is: `exp4` is the **init-only** version of using bigram structure. It seeds the model from an SVD-derived co-occurrence space, then trains normally.

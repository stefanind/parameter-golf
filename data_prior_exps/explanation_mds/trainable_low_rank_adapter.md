# train_gpt_exp5.py

## Main idea

This experiment tests a **trainable low-rank bigram adapter**.

Instead of only using bigram structure at initialization time, this file adds a dedicated low-rank module that continues to contribute token-to-token logits throughout training.

## What the code is doing

- Adds two trainable matrices:
  - `bigram_adapter_A` with shape `[vocab_size, rank]`
  - `bigram_adapter_B` with shape `[vocab_size, rank]`
- Looks up the row of `A` for the previous token.
- Multiplies that low-rank representation by `B^T` to produce an additive vocabulary-sized logit correction.
- Scales that correction by a learned scalar `bigram_adapter_scale`.
- Adds the result to the main logits before the loss.

So the adapter behaves like a learned low-rank token-to-token transition table.

## Initialization behavior

The adapter can be initialized in two ways:

- random low-rank factors
- SVD-derived factors from `BIGRAM_ADAPTER_INIT_PATH`

That makes this file a natural bridge between:

- `exp4`: use bigram structure only at initialization
- `exp5`: keep a trainable low-rank bigram pathway alive during optimization

## Optimizer behavior

The adapter has its own optimizer path:

- adapter matrices use `BIGRAM_ADAPTER_LR`
- adapter scale is treated like a scalar parameter

This means the experiment can tune the adapter somewhat independently from the base model.

## Why this experiment exists

The hypothesis is that a compact low-rank transition module may be a better use of parameters than forcing the transformer to absorb all bigram structure into its general weights.

It is the persistent, trainable version of the same family of ideas explored more statically in `exp4`.

## Important knobs

- `BIGRAM_ADAPTER_RANK`
- `BIGRAM_ADAPTER_LR`
- `BIGRAM_ADAPTER_INIT_PATH`
- `BIGRAM_ADAPTER_INIT_MODE`
- `BIGRAM_ADAPTER_INIT_SCALE`
- `BIGRAM_ADAPTER_SCALE_INIT`

## Representative logs in `outputs/`

- `outputs/adapter_exp5.txt`
- `outputs/adapter_exp5_run2.txt`
- `outputs/adapter_exp5_run3.txt`
- `outputs/adapter_exp5_run4_optimizer_step.txt`
- `outputs/baseline_exp5.txt`

## Note

If `exp4` is "initialize from a low-rank bigram structure," `exp5` is "add a trainable low-rank bigram structure to the model itself."

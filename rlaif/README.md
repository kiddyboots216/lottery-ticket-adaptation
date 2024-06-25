# Aligning Language Models

## What is this repo?

This repo is the official implementation of Lottery Ticket Adaptation, described in the paper [Lottery Ticket Adaptation](https://arxiv.org/abs/2406.16797). This codebase builds on the official implementation of [A Critical Evaluation of AI Feedback for Aligning Language Models](https://arxiv.org/abs/2402.12366), which itself builds on the official [DPO](https://github.com/eric-mitchell/direct-preference-optimization) implementation.

## What does this repo implement?

We implement datasets for a range of tasks (see the `tasks` folder for a full overview) and a trainer that uses a single GPU to finetune models up to Llama-3-8B in size. This is made possible by fusing the optimizer into the backward pass in `apply_mask_in_backward`. 

### Customizing training
The options for training are in `config/config.yaml`, `config/model/blank_model.yaml`, and `config/loss/dpo.yaml`. See the comments in these files for more information on what they do.

You can use one of the pre-configured models by passing `model=some_model`, where `config/model/some_model.yaml` exists. We have a few examples already given.

If you want to use another model, just create a new config for that model (following our examples; it must be a `.yaml` file!), or use `model=blank_model` with `model.name_or_path=NAME_OR_PATH`, optionally `model.tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH` if it is different than the model's name/path, and `model.block_name=NAME_OF_TRANSFORMER_BLOCK` (if you are using FSDP). The only other options you might want to change are the dpo loss options, which are `loss.beta` and `loss.reference_free` (see `config/loss/dpo.yaml`).

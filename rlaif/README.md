# Aligning Language Models

## What is this repo?

This repo is the official implementation of Lottery Ticket Adaptation, described in the paper [Lottery Ticket Adaptation](https://arxiv.org/abs/2406.16797). This codebase builds on the official implementation of [A Critical Evaluation of AI Feedback for Aligning Language Models](https://arxiv.org/abs/2402.12366), which itself builds on the official [DPO](https://github.com/eric-mitchell/direct-preference-optimization) implementation.

## What does this repo implement?

We implement datasets for a range of tasks (see the `tasks` folder for a full overview) and a trainer that uses a single GPU to finetune models up to Llama-3-8B in size. This is made possible by fusing the optimizer into the backward pass in `apply_mask_in_backward`. 

### Setup

## Step 0: Create environment and set paths
    cd rlaif
    conda create -p $CONDA_ENV_PATH align
    conda activate $CONDA_ENV_PATH/align
    pip install -r requirements.txt
    cd ../mergekit
    pip install -e .
## Step 1: Download ShareGPT4 Data
    wget -P ${PROJECT_CACHE}/sharegpt_data https://huggingface.co/datasets/TRI-ML/dpo-rlaif-data/resolve/main/sharegpt1turn_df1.0_ff0_gpt4_completions.json

### Running training

```
python -u train_single_gpu.py do_first_eval=False \
        mask_path=../masks/$mask_dir/${sparsity_ratio}_mask.pt \
        loss=sft \
        model=${model} \
        model.archive=${model_archive} \
        datasets=[${dataset_name}] \
        exp_name=${exp_name} \
        eval_batch_size=16 \
        sample_during_eval=false \
        lr=$lr \
        trainer=$trainer_type \
        activation_checkpointing=True \
        data_fraction=$data_fraction \
        save_every=epoch_$n_epochs \
        eval_every=100000 \
        n_epochs=$n_epochs \
        batch_size=$batch_size \
        gradient_accumulation_steps=1 \
        model.fsdp_policy_mp=bfloat16 \
        fsdp_port=${MASTER_PORT} \
        optimizer=RMSprop \
        grad_norm_strategy=even \
        max_grad_norm=$grad_norm 
```

### Converting the saved model

The model is saved in a `.pt` format. You can convert this to HF. 

`python convert_policy_to_hf_resize.py --model_path ${model_archive} --policy_path ${model_save_path}/epoch-$n_epochs/policy.pt --save_path ${model_save_path}/epoch-$n_epochs/`

### Evaluating the saved model

For capabilities other than instruction following, we can evaluate the model with the `eval_all` script. This script also contains all details on how we extract answers, prompt the model, etc. for all datasets.

`python eval_model_all.py --model "${model_save_path}/epoch-$n_epochs/" --datasets "${dataset_name}" --sample`

### Generating samples for instruction following

In order to evaluate the winrate, we need to generate samples from the model, which we can do via

`python generate_samples.py --prompt_set alpaca_eval --temperatures 0.7 --model_name ${exp_name} --model_path ${model_save_path}/epoch-$n_epochs/`

We can then get the winrate with AlpacaEval. 

### Customizing training
The options for training are in `config/config.yaml`, `config/model/blank_model.yaml`, and `config/loss/dpo.yaml`. See the comments in these files for more information on what they do.

You can use one of the pre-configured models by passing `model=some_model`, where `config/model/some_model.yaml` exists. We have a few examples already given.

If you want to use another model, just create a new config for that model (following our examples; it must be a `.yaml` file!), or use `model=blank_model` with `model.name_or_path=NAME_OR_PATH`, optionally `model.tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH` if it is different than the model's name/path, and `model.block_name=NAME_OF_TRANSFORMER_BLOCK` (if you are using FSDP). The only other options you might want to change are the dpo loss options, which are `loss.beta` and `loss.reference_free` (see `config/loss/dpo.yaml`).

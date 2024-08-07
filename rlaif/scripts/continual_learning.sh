#!/bin/bash

### first let's do the fft baseline
# first train a model on gsm8k
archive="mistralai/Mistral-7B-v0.1"
lr="5e-7"
dataset_name="gsm8k"
data_fraction="1.0"
n_epochs="3"
mask_dir="none"
sparsity_ratio="0.0"
batch_size="32"
model="mistral7b"
grad_norm="1"
bash scripts/train_single_gpu.sh "$archive" "$lr" "$dataset_name" "$data_fraction" "$n_epochs" "$mask_dir" "$sparsity_ratio" "$batch_size" "$model" "$grad_norm"
# then train that same model on commonsense
# you would need to change the archive path to the path of the model you just trained
archive="/scratch/gpfs/ashwinee/rlaif-cache/gsm8k_mistralai/Mistral-7B-v0.1_1_5e-7_32_1_none_0.0/epoch-3/"
dataset_name="commonsense"
n_epochs="1"
bash scripts/train_single_gpu.sh "$archive" "$lr" "$dataset_name" "$data_fraction" "$n_epochs" "$mask_dir" "$sparsity_ratio" "$batch_size" "$model" "$grad_norm"
# now eval the model on commonsense and gsm8k
# similarly, set the model save path here to the path of the model you just trained
model_save_path="/scratch/gpfs/ashwinee/rlaif-cache//commonsense_/scratch/gpfs/ashwinee/rlaif-cache//gsm8k_mistralai/Mistral-7B-v0.1_1_5e-7_32_1_none_0.0/epoch-3/_1_5e-7_32_1_none_0.0/epoch-1/"
python eval_model_all.py --model "${model_save_path}" --datasets "commonsense,gsm8k" --sample

### now let's do the lora baseline
# resetting normal hparams
archive="mistralai/Mistral-7B-v0.1"
dataset_name="gsm8k"
n_epochs="3"
# now set the lora hparams
rank="64"
alpha="32" 
lr="1e-4"
# after tireless hparam search (512 grid dimension) these are the best values
bash scripts/train_single_gpu_lora.sh "$archive" "$lr" "$dataset_name" "$data_fraction" "$n_epochs" "$mask_dir" "$sparsity_ratio" "$batch_size" "$model" "$grad_norm" "$lora_rank" "$alpha"
# now set the archive to the model you just trained
archive="/scratch/gpfs/ashwinee/rlaif-cache/gsm8k_mistralai/Mistral-7B-v0.1_1_1e-4_32_1_none_0.0_LORA_64_32/epoch-1/"
dataset_name="commonsense"
n_epochs="1"
bash scripts/train_single_gpu.sh "$archive" "$lr" "$dataset_name" "$data_fraction" "$n_epochs" "$mask_dir" "$sparsity_ratio" "$batch_size" "$model" "$grad_norm" "$lora_rank" "$alpha"
python eval_model_all.py --model "${model_save_path}" --datasets "commonsense,gsm8k" --sample

### now let's do a competitive lora baseline; we'll lora half the layers on gsm8k, then the other half on commonsense
# resetting normal hparams
archive="mistralai/Mistral-7B-v0.1"
dataset_name="gsm8k"
n_epochs="3"
# now set the lora hparams
rank="64"
alpha="32" 
lr="1e-4"
freeze_odd_layers="true"
# after tireless hparam search (512 grid dimension) these are the best values
bash scripts/train_single_gpu_lora.sh "$archive" "$lr" "$dataset_name" "$data_fraction" "$n_epochs" "$mask_dir" "$sparsity_ratio" "$batch_size" "$model" "$grad_norm" "$lora_rank" "$alpha" "$freeze_odd_layers"
# now set the archive to the model you just trained
archive="/scratch/gpfs/ashwinee/rlaif-cache/gsm8k_mistralai/Mistral-7B-v0.1_1_1e-4_32_1_none_0.0_LORA_64_32/epoch-1/"
dataset_name="commonsense"
n_epochs="1"
freeze_odd_layers="false"
freeze_even_layers="true"
bash scripts/train_single_gpu.sh "$archive" "$lr" "$dataset_name" "$data_fraction" "$n_epochs" "$mask_dir" "$sparsity_ratio" "$batch_size" "$model" "$grad_norm" "$lora_rank" "$alpha" "$freeze_odd_layers" "$freeze_even_layers"
python eval_model_all.py --model "${model_save_path}" --datasets "commonsense,gsm8k" --sample

### now it's time for a lota baseline
# you can calibrate this easily yourself by just setting data_fraction=something small and n_epochs=1, but here we'll assume that you've followed the previous steps and we'll just reuse the existing model
archive_path="/scratch/gpfs/ashwinee/rlaif-cache/gsm8k_mistralai/Mistral-7B-v0.1_1_5e-7_32_1_none_0.0/epoch-3/"
base_path="mistral"
merge_name="gsm8k-mistral-1"
sparsities="99" # you can train with less sparsity, here we want to show an extreme example where we can fit both these tasks with 1% sparsity each without forgetting
bash scripts/make_diffs_and_masks.sh $archive_path $base_path $merge_name $sparsities
# now let's train the gsm8k model
archive="mistralai/Mistral-7B-v0.1"
lr="1e-5"
dataset_name="gsm8k"
data_fraction="1.0" # you can set this lower -we've found that even 0.001 works, but you will have to increase the ratio of (lr / batch size)
n_epochs="3"
mask_dir="gsm8k-mistral-1"
sparsity_ratio="0.9"
batch_size="32"
model="mistral7b"
grad_norm="1"
freeze_odd_layers="false"
freeze_even_layers="false"
bash scripts/train_single_gpu.sh "$archive" "$lr" "$dataset_name" "$data_fraction" "$n_epochs" "$mask_dir" "$sparsity_ratio" "$batch_size" "$model" "$grad_norm"
# then train that same model on commonsense
# you would need to change the archive path to the path of the model you just trained
archive="/scratch/gpfs/ashwinee/rlaif-cache//gsm8k_mistralai/Mistral-7B-v0.1_1_1e-5_32_1_gsm8k-mistral-1-mlpattn-only_0.99/epoch-3/"
dataset_name="commonsense"
n_epochs="1"
lr="1e-7" # you can make this smaller to make commonsense better, and make this smaller to make gsm8k better
flip_mask="true" # now we're going to train on everything *other* than what we just trained on
bash scripts/train_single_gpu.sh "$archive" "$lr" "$dataset_name" "$data_fraction" "$n_epochs" "$mask_dir" "$sparsity_ratio" "$batch_size" "$model" "$grad_norm" "$flip_mask"
# now eval the model on commonsense and gsm8k
# similarly, set the model save path here to the path of the model you just trained
model_save_path=""
python eval_model_all.py --model "${model_save_path}" --datasets "commonsense,gsm8k" --sample
# this model does very well! but this allows 99% of the parameters to be updated for commonsense. assuming that we have more sequential tasks, we'll want to also make sure that this mask is sparse.
# to do this, we'll once again create a mask.
# (and once again, if you're skipping to this step, feel free to just use a small fraction of data to calibrate this mask)
archive_path="/scratch/gpfs/ashwinee/rlaif-cache//commonsense_/scratch/gpfs/ashwinee/rlaif-cache//gsm8k_mistralai/Mistral-7B-v0.1_1_1e-5_32_1_gsm8k-mistral-1-mlpattn-only_0.99/epoch-3/_1_1e-7_32_1_gsm8k-mistral-1-mlpattn-only_0.99/epoch-1/"
base_path="/scratch/gpfs/ashwinee/rlaif-cache//gsm8k_mistralai/Mistral-7B-v0.1_1_1e-5_32_1_gsm8k-mistral-1-mlpattn-only_0.99/epoch-3/"
merge_name="gsm8k-commonsense-mistral-1"
sparsities="99"
bash scripts/make_diffs_and_masks.sh $archive_path $base_path $merge_name $sparsities
# now let's train a model that is 99% sparse on commonsense from the model that was 99% on gsm8k
mask_dir="gsm8k-commonsense-mistral-1"
bash scripts/train_single_gpu.sh "$archive" "$lr" "$dataset_name" "$data_fraction" "$n_epochs" "$mask_dir" "$sparsity_ratio" "$batch_size" "$model" "$grad_norm"
# similarly, set the model save path here to the path of the model you just trained
model_save_path=""
python eval_model_all.py --model "${model_save_path}" --datasets "commonsense,gsm8k" --sample
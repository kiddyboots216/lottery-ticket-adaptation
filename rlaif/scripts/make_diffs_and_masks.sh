#!/bin/bash

# replace these as necessary
CONDA_PATH=/scratch/gpfs/$USER/envs/align
module purge 
module load anaconda3/2022.10
conda activate $CONDA_PATH
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/scratch/gpfs/ashwinee/rlaif-cache/
export PROJECT_CACHE=/scratch/gpfs/ashwinee/rlaif-cache/
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

model_path="${1}"
base_path="${2}"
merge_name="${3}"
sparsity="${4}"
python modify_yaml.py $model_path $base_path
mergekit-yaml task_vector.yaml $merge_name --cuda
python save_mask.py --merge_path $merge_name --sparsities $sparsity
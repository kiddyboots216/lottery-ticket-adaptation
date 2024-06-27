#!/bin/bash

#SBATCH     --nodes=1               # node count
#SBATCH     --ntasks-per-node=1      # total number of tasks per node
#SBATCH     --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH     --mem=80G                # total memory per node (4 GB per cpu-core is default)
#SBATCH     --gres=gpu:1             # number of gpus per node
#SBATCH     --time=16:59:00          # total run time limit (HH:MM:SS)
##SBATCH    --partition=pli            # partition (queue)
##SBATCH --account=lotteryticket
#SBATCH    --constraint=gpu80         # constraint (e.g. gpu80)
#SBATCH     -o Report/%j.out            # STDOUT
#SBATCH     --mail-type=FAIL          # send email on job start, end and fail
#SBATCH     --mail-user=ashwinee@princeton.edu      # email address to send to

# replace these as necessary
CONDA_PATH=/scratch/gpfs/$USER/envs/align
module purge 
module load anaconda3/2022.10
conda activate $CONDA_PATH
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/scratch/gpfs/ashwinee/rlaif-cache/
export HF_HOME=/scratch/gpfs/ashwinee/rlaif-cache/
export PROJECT_CACHE=/scratch/gpfs/ashwinee/rlaif-cache/
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export TORCH_DISTRIBUTED_DEBUG=OFF

archive="${1:-mistralai/Mistral-7B-v0.1}"
lr="${2:-5e-7}"
dataset_name="${3:-sharegpt4}"
data_fraction="${4:-1.0}"
n_epochs="${5:-3}"
mask_dir="${6:-none}"
sparsity_ratio="${7:-0.0}"
batch_size="${8:-8}"
model="${9:-mistral7b}"
grad_norm="${10:-10}"
lora_rank="${11:-8}"
lora_alpha="${12:-32}"

model_archive=${archive}
if [[ $archive == "/scratch/gpfs/ashwinee/rlaif-cache/sharegpt4_"* ]]; then
    archive=${archive#"/scratch/gpfs/ashwinee/rlaif-cache/sharegpt4_"}
fi
# Replace commas with underscores to avoid issues with file paths or names
sanitized_dataset_name=$(echo $dataset_name | tr ',' '_')
exp_name="${sanitized_dataset_name}_${archive}_${grad_norm}_${lr}_${batch_size}_${data_fraction}_${mask_dir}_${sparsity_ratio}_LORA_${lora_rank}_${lora_alpha}"
# exp_name="${sanitized_dataset_name}_${archive}_${grad_norm}_${lr}_${batch_size}_${data_fraction}_${mask_dir}_${sparsity_ratio}_DORA"
model_save_path="/scratch/gpfs/ashwinee/rlaif-cache/${exp_name}/"
trainer_type="FSDPTrainer"

python -u train_single_gpu_lora.py do_first_eval=False \
        mask_path=/scratch/gpfs/ashwinee/alignment-durability/masks/$mask_dir/${sparsity_ratio}_mask.pt \
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
        max_grad_norm=$grad_norm \
        lora_rank=$lora_rank \
        lora_alpha=$lora_alpha 

python convert_policy_to_hf_lora.py --lora_rank ${lora_rank} --model_path ${model_archive} --policy_path ${model_save_path}/epoch-$n_epochs/policy.pt --save_path ${model_save_path}/epoch-$n_epochs/

python eval_model_all.py --model "${model_save_path}/epoch-$n_epochs/" --datasets "${dataset_name}"

python generate_samples.py --prompt_set alpaca_eval --temperatures 0.7 --model_name ${exp_name} --model_path ${model_save_path}/epoch-$n_epochs/

python cleanup.py --model ${model_save_path}/epoch-$n_epochs/
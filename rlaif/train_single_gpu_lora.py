import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed
import os
import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set

from huggingface_hub import login
dist.set_debug_level(dist.DebugLevel.OFF)

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    # trainer.save(run_alpaca_eval=config.trigger_alpaca_eval)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    
    load_path = config.model.archive if "null" not in config.model.archive else config.model.name_or_path
    print('building policy from path', load_path)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        load_path, low_cpu_mem_usage=True, use_cache=False, torch_dtype=policy_dtype, **model_kwargs)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # use_dora=True,
            target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )
    policy = get_peft_model(policy, peft_config)
    
    freeze_odd_layers = config.freeze_odd_layers
    freeze_even_layers = config.freeze_even_layers
    if freeze_odd_layers:
        for idx, (name, param) in enumerate(policy.named_parameters()):
            if idx % 2 == 1:
                param.requires_grad = False
    if freeze_even_layers:
        for idx, (name, param) in enumerate(policy.named_parameters()):
            if idx % 2 == 0:
                param.requires_grad = False
    disable_dropout(policy)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        policy.config.pad_token_id = tokenizer.pad_token_id
        policy.resize_token_embeddings(len(tokenizer))

    if config.loss.name in ['dpo', 'soft_sft']:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, use_cache=False, low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None
    import glob

    # if config.local_run_dir is not None:
    #     step_dirs = glob.glob(os.path.join(config.local_run_dir, 'step-*'))
    #     epoch_dirs = glob.glob(os.path.join(config.local_run_dir, 'epoch-*'))
    #     dirs_to_try = sorted(step_dirs + epoch_dirs, key=os.path.getctime, reverse=True)
        
    #     for dir in dirs_to_try:
    #         try:
    #             policy_path = os.path.join(dir, 'policy.pt')
    #             if os.path.isfile(policy_path):
    #                 state_dict = torch.load(policy_path, map_location='cpu')
    #                 policy.load_state_dict(state_dict['state'])
    #                 print(f'Loading pre-trained weights from {dir}')
    #                 config.fast_forward = int(state_dict["step_idx"] / config.batch_size)
    #                 config.ckpt_path = dir
    #                 break  # Exit the loop if successfully loaded
    #             else:
    #                 raise FileNotFoundError(f"No policy.pt file found in {dir}")
    #         except (FileNotFoundError, RuntimeError) as e:
    #             print(f"Failed to load from {dir}: {e}")
    #             continue  # Try the next directory
        
    #     if not config.ckpt_path:
    #         config.fast_forward = 0
    #         print("No valid checkpoint found, starting from scratch.")

    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
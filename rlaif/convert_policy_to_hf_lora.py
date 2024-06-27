import torch 
import os
import transformers 
import argparse
from transformers import LlamaTokenizer

# Parse arguments for policy path and save path
parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", type=str, required=True, help="Path to the policy file.")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the converted policy.")
parser.add_argument("--model_path", type=str, default='mistralai/Mistral-7B-v0.1', help="Path to the model.")
parser.add_argument("--lora_rank", type=int, default=256, help="Rank of the LoRA decomposition.")
args = parser.parse_args()
model_path = args.model_path

# Load the policy
state_dict = torch.load(args.policy_path, map_location='cpu')
policy = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda')
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
peft_config = LoraConfig(
        # r=16,
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)
policy = get_peft_model(policy, peft_config)
# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.resize_token_embeddings(len(tokenizer))# if getattr(tokenizer, "pad_token", None) is None:
# Load the state dict and save the pretrained model
policy.load_state_dict(state_dict['state'])
policy.merge_and_unload()
policy.base_model.save_pretrained(args.save_path)
tokenizer.save_pretrained(args.save_path)
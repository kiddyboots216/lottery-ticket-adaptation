import torch 
import os
import transformers 
import argparse
from transformers import LlamaTokenizer

# Parse arguments for policy path and save path
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='mistralai/Mistral-7B-v0.1', help="Path to the model.")
parser.add_argument("--policy_path", type=str, required=True, help="Path to the policy file.")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the converted policy.")
args = parser.parse_args()

model_path = args.model_path
# Load the tokenizer
# print(model_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

# Load the policy
state_dict = torch.load(args.policy_path, map_location='cpu')
policy = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda')



# tokenizer.pad_token_id = tokenizer.eos_token_id
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.resize_token_embeddings(len(tokenizer))
# Load the state dict and save the pretrained model
policy.load_state_dict(state_dict['state'])
policy.save_pretrained(args.save_path)
tokenizer.save_pretrained(args.save_path)


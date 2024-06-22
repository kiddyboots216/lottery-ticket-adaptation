import transformers
import torch
import os
from datasets import load_dataset
import torch.nn as nn

def get_random_sparsity_masks(model, sparsity_ratio):
    """
    This code randomly masks the weights instead of using a threshold.
    """    
    masks = {}
    dir_name = "random_masks_mlp"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for sparsity_ratio in sparsity_ratios:
        pruned_num = 0
        total_num = 0
        mask = {}
        for name, param in model.named_parameters():
            if "weight" in name and "mlp" in name:
                # Create a random mask with the same number of True values as specified by the sparsity_ratio
                total_elements = param.data.numel()
                num_true = int(total_elements * sparsity_ratio)
                mask_flat = torch.zeros(total_elements, dtype=torch.bool)
                mask_flat[:num_true] = True
                # Shuffle the mask to randomize which parameters are set to True
                mask_flat = mask_flat[torch.randperm(total_elements)]
                mask[name] = mask_flat.view_as(param.data).to("cpu").detach()
                # Update the count of masked and total parameters
                pruned_num += mask[name].int().sum()
                total_num += total_elements
        masks[sparsity_ratio] = mask
        print(f"Global sparsity enforced at {sparsity_ratio:.2f} level.")
        print(f"{(100 * pruned_num / total_num):.2f}% of parameters will be pruned.")
        print(f"{(100 * (total_num - pruned_num) / total_num):.2f}% of parameters will be retained.")
        save_mask(mask, f"{dir_name}/{sparsity_ratio}_mask.pt")
    return masks

def get_global_sparsity_masks(model, sparsity_ratios, save_path, pruning_fn, only_update_prune=False, bottom_k=False):
    """
    This function enforces different levels of global sparsity across all parameters in the model.
    It calculates global threshold values for sparsity using topk for efficiency.
    The resulting masks will be True for weights that are being pruned.
    
    Parameters:
    - model (nn.Module): The model to apply the sparsity masks to.
    - sparsity_ratios (list of float): The sparsity ratios to enforce.

    Returns:
    - dict of masks: A dictionary where keys are sparsity ratios and values are the corresponding masks.
    """
    # Concatenate all parameters into a single vector and sort the absolute values
    all_params_abs = []
    for name, p in model.named_parameters():
        if p.requires_grad and pruning_fn(name) and 'weight' in name:
            # Flatten and take absolute values of the parameters
            all_params_abs.append(torch.abs(p.data.view(-1)).cpu())

    # Concatenate the absolute values into a single tensor
    all_params_abs = torch.cat(all_params_abs)
    total_num = all_params_abs.numel()
    masks = {}
    if bottom_k:
        save_path = f"{save_path}_bottomk"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sparsity_ratio in sparsity_ratios:
        pruned_num = 0
        # Calculate the number of values to retain based on the sparsity ratio
        k = int((1 - sparsity_ratio) * total_num)
        # Use topk to find the k largest values, which are the ones we want to retain
        # The threshold is the smallest value among the ones we want to retain
        # if bottom_k:
        if False:
            threshold = torch.topk(all_params_abs, k, largest=False).values[-1]
        else:
            threshold = torch.topk(all_params_abs, k).values[-1]
        print(f"{threshold:.16f}")

        # Create a mask for each parameter based on the threshold
        mask = {}
        for name, param in model.named_parameters():
            if "weight" in name and pruning_fn(name):
                if bottom_k:   
                    # Parameters above the threshold are marked True (to be pruned)
                    param_mask = (torch.abs(param.data) > threshold).to(param.device)
                else:
                    # Parameters below the threshold are marked True (to be pruned)
                    param_mask = (torch.abs(param.data) < threshold).to(param.device)
                mask[name] = param_mask
                pruned_num += param_mask.sum().item()
            elif only_update_prune:
                # this means that we are going to set the entire param to True (meaning it will all get pruned)
                param_mask = torch.ones_like(param.data).to(param.device)
                mask[name] = param_mask

        masks[sparsity_ratio] = mask
        print(f"Global sparsity enforced at {sparsity_ratio:.2f} level.")
        print(f"{(100 * pruned_num / total_num):.2f}% of parameters will be pruned.")
        print(f"{(100 * (total_num - pruned_num) / total_num):.2f}% of parameters will be retained.")
        save_mask(mask, f"{save_path}/{sparsity_ratio}_mask.pt")
    return masks

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def save_mask(mask, file_path):
    """
    Save the mask to a file.

    Parameters:
    - mask (dict): The mask to save.
    - file_path (str): The file path where the mask will be saved.
    """
    torch.save(mask, file_path)
    print(f"Mask saved to {file_path}")

def load_mask(file_path):
    """
    Load the mask from a file.

    Parameters:
    - file_path (str): The file path from where to load the mask.

    Returns:
    - dict: The loaded mask.
    """
    mask = torch.load(file_path)
    print(f"Mask loaded from {file_path}")
    return mask
# Assuming `mask` is your mask dictionary and you want to save it
# save_mask_path = f"masks/{modeltag}_mask.pt"
# save_mask(mask, save_mask_path)

def compare_masks(mask_path_a, mask_path_b):
    """
    Load two masks and iterate over all the parameters and see how many parameters are "False" for both masks.
    """
    mask_a = torch.load(mask_path_a, map_location='cuda')
    mask_b = torch.load(mask_path_b, map_location='cuda')
    assert mask_a.keys() == mask_b.keys(), "Keys of the two masks do not match."
    total_params = 0
    common_false_params = 0
    mask_a_params = 0
    mask_b_params = 0
    for name in mask_a:
        # Count the parameters that are False in mask_a
        mask_a_params += (~mask_a[name]).sum().item()
        # Count the parameters that are False in mask_b
        mask_b_params += (~mask_b[name]).sum().item()
        # Count the parameters that are False in both masks
        common_false_params += (~(mask_a[name] | mask_b[name])).sum().item()
        total_params += mask_a[name].numel()
    print(f"Common False parameters: {common_false_params}, Total parameters: {total_params}")
    print(f"Common False ratio: {common_false_params / total_params:.4f}")
    print(f"Mask A False ratio: {mask_a_params / total_params:.4f}")
    print(f"Mask B False ratio: {mask_b_params / total_params:.4f}")
def get_constrained_mask(model, sparsity_ratio, output_dir, input_mask_path, pruning_fn, only_update_prune=False):
    """
    Create a mask that respects the constraints of an input_mask. Parameters set to "False" in the input_mask
    will be set to "True" in the new mask, ensuring they are pruned. Then, proceed to create a global sparsity
    mask as normal.

    Parameters:
    - model (nn.Module): The model to apply the sparsity masks to.
    - sparsity_ratio (float): The sparsity ratio to enforce.
    - output_dir (str): The directory where the mask will be saved.
    - input_mask_path (str): The file path of the input mask to respect.

    Returns:
    - dict: The constrained mask.
    """
    # Load the input mask
    input_mask = torch.load(input_mask_path, map_location='cuda')
    # Save the constrained mask
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Concatenate all parameters into a single vector and sort the absolute values
    all_params_abs = []
    for name, p in model.named_parameters():
        if p.requires_grad and pruning_fn(name) and 'weight' in name:
            # Flatten and take absolute values of the parameters
            param_values = p.data.view(-1)
            # Apply the constraint from the input mask
            constraint = input_mask[name].view(-1)
            param_values[~constraint] = 0  # The mask is 1 if the OG is pruning something, so it's 0 if the OG is not being pruned
            all_params_abs.append(torch.abs(param_values).cpu())

    # Concatenate the absolute values into a single tensor
    all_params_abs = torch.cat(all_params_abs)
    total_num = all_params_abs.numel()
    for sparsity_ratio in sparsity_ratios:
        pruned_num = 0
        # Calculate the number of values to retain based on the sparsity ratio
        k = int((1 - sparsity_ratio) * total_num)
        # Use topk to find the k largest values, which are the ones we want to retain
        # The threshold is the smallest value among the ones we want to retain
        threshold = torch.topk(all_params_abs, k).values[-1]

        # Create a mask for each parameter based on the threshold
        constrained_mask = {}
        for name, param in model.named_parameters():
            if "weight" in name and pruning_fn(name):
                # Parameters below the threshold are marked True (to be pruned)
                param_mask = (torch.abs(param.data) < threshold).to('cpu')
                # Apply the constraint from the input mask
                constrained_mask[name] = param_mask | ~input_mask[name].cpu()
                pruned_num += param_mask.sum().item()
            elif only_update_prune:
                # this means that we are going to set the entire param to True (meaning it will all get pruned)
                param_mask = torch.ones_like(param.data).to(param.device)
                constrained_mask[name] = param_mask
        total_params = 0
        common_false_params = 0
        # for name in constrained_mask:
        #     # Count the parameters that are False in both masks
        #     common_false_params += (~(constrained_mask[name] | input_mask[name].cpu())).sum().item()
        #     total_params += constrained_mask[name].numel()
        # print(f"Common False parameters: {common_false_params}, Total parameters: {total_params}")
        # print(f"Common False ratio: {common_false_params / total_params:.4f}")
        print(f"Global sparsity enforced at {sparsity_ratio:.2f} level.")
        print(f"{(100 * pruned_num / total_num):.2f}% of parameters will be pruned.")
        print(f"{(100 * (total_num - pruned_num) / total_num):.2f}% of parameters will be retained.")
        save_mask(constrained_mask, f"{output_dir}/{sparsity_ratio}_mask.pt")

    return constrained_mask

def invert_mask(path):
    """
    Invert the mask so that the True values become False and vice versa.

    Parameters:
    - mask (dict): The mask to invert.

    Returns:
    - dict: The inverted mask.
    """
    mask = torch.load(path, map_location='cuda')
    inverted_mask = {}
    for name in mask:
        inverted_mask[name] = ~(mask[name].bool())  # Ensure the mask is of Boolean type before inverting
    torch.save(inverted_mask, path.replace(".pt", "_inverted.pt"))

def get_task_vector(model, base_model):
    """
    The task vector is the difference between model and base_model for all parameters.
    Returns a dictionary mapping parameter name to the task vector.
    """
    task_vector = {}
    base_model_params = base_model.state_dict()
    for name, param in model.named_parameters():
        if 'weight' in name and ('mlp' in name or 'attn' in name):
            if param.requires_grad:
                base_param = base_model_params.get(name, torch.zeros_like(param.data))
                task_vector[name] = param.data - base_param
    return task_vector

def get_task_vector_sparsity(task_vector):
    # returns the percentage of entries in the task vector that are nonzero
    total_params = 0
    nonzero_params = 0
    for name, vector in task_vector.items():
        total_params += vector.numel()
        nonzero_params += (vector != 0).sum().item()
    print(f"Total params: {total_params}, Nonzero params: {nonzero_params}")
    return nonzero_params / total_params if total_params > 0 else 0

def compare_task_vectors(model_a_path, model_b_path, base_model_path):
    from transformers import AutoModelForCausalLM
    # compute the task vector for each model
    # compute the sparsity of those task vectors
    # finally, compute the sparsity of the intersection of the task vectors
    model_a = AutoModelForCausalLM.from_pretrained(model_a_path)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_model.resize_token_embeddings(32001)
    task_vector_a = get_task_vector(model_a, base_model)
    task_vector_b = get_task_vector(model_b, base_model)
    task_vector_intersection = {}
    for name in task_vector_a:
        if name in task_vector_b:
            task_vector_intersection[name] = task_vector_a[name] * (task_vector_b[name] != 0)
    sparsity_a = get_task_vector_sparsity(task_vector_a)
    sparsity_b = get_task_vector_sparsity(task_vector_b)
    sparsity_intersection = get_task_vector_sparsity(task_vector_intersection)
    print(f"Sparsity of task vector A: {sparsity_a:.4f}")
    print(f"Sparsity of task vector B: {sparsity_b:.4f}")
    print(f"Sparsity of intersection of task vectors: {sparsity_intersection:.4f}")


if __name__ == "__main__":
    if False:
        compare_task_vectors(
            "/scratch/gpfs/ashwinee/rlaif-cache/sharegpt4_mistralai/Mistral-7B-v0.1_1_5e-7_8_1.0_sharegpt4-mistral_0.9/epoch-3",
            "/scratch/gpfs/ashwinee/rlaif-cache/gsm8k_mistralai/Mistral-7B-v0.1_10_5e-7_32_1.0_gsm8k-mistral-constrained_0.9/epoch-4",
            "/scratch/gpfs/ashwinee/rlaif-cache/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24"
        )
        # compare_masks("/scratch/gpfs/ashwinee/alignment-durability/gsm8k-mistral-constrained/0.9_mask.pt",
                    #   "/scratch/gpfs/ashwinee/alignment-durability/sharegpt4-mistral-0.9/0.9_mask.pt")
        # from argparse import ArgumentParser
        # parser = ArgumentParser()
        # parser.add_argument("--merge_path", type=str, default="sharegpt4")
        # args = parser.parse_args()
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        # f"/scratch/gpfs/ashwinee/alignment-durability/output-merges/{args.merge_path}",
        #     torch_dtype=torch.float32,
        #     device_map='cpu'
        #     )
        # check_sparsity(model)
    else:
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument("--merge_path", type=str, default="sharegpt4")
        parser.add_argument("--constraint_path", type=str)
        args = parser.parse_args()
        model = transformers.AutoModelForCausalLM.from_pretrained(
        f"/scratch/gpfs/ashwinee/alignment-durability/output-merges/{args.merge_path}",
            torch_dtype=torch.float32,
            device_map='cuda'
            )
        sparsity_ratios = [0.9]
        if True:
            pruning_fn = lambda name: 'mlp' in name or 'attn' in name
            masks = get_global_sparsity_masks(model, sparsity_ratios, f"masks/{args.merge_path}-mlpattn-only", pruning_fn, only_update_prune=True)
        if False:
            pruning_fn = lambda name: 'mlp' in name or 'attn' in name
            masks = get_global_sparsity_masks(model, [0.90], f"masks/{args.merge_path}-mlpattn-only", pruning_fn, only_update_prune=True, bottom_k=True)
        if False:
            pruning_fn = lambda name: 'mlp' in name
            masks = get_global_sparsity_masks(model, sparsity_ratios, f"{args.merge_path}-mlp-only", pruning_fn, only_update_prune=True)
            pruning_fn = lambda name: 'mlp' in name or 'attn' in name
            masks = get_global_sparsity_masks(model, sparsity_ratios, f"{args.merge_path}-mlpattn-only", pruning_fn, only_update_prune=True)
            pruning_fn = lambda name: 'mlp' in name
            masks = get_global_sparsity_masks(model, sparsity_ratios, f"{args.merge_path}-mlp", pruning_fn, only_update_prune=False)
            pruning_fn = lambda name: 'mlp' in name or 'attn' in name
            masks = get_global_sparsity_masks(model, sparsity_ratios, f"{args.merge_path}-mlpattn", pruning_fn, only_update_prune=False)
            # invert_mask(f"{args.merge_path}/0.99_mask.pt")
        if False:
            constraint_mask_path = args.constraint_path
            pruning_fn = lambda name: 'mlp' in name
            masks = get_constrained_mask(model, sparsity_ratios, f"{args.merge_path}-mlp-only-constrained", constraint_mask_path, pruning_fn, only_update_prune=True)
            pruning_fn = lambda name: 'mlp' in name or 'attn' in name
            masks = get_constrained_mask(model, sparsity_ratios, f"{args.merge_path}-mlpattn-only-constrained", constraint_mask_path, pruning_fn, only_update_prune=True)
            pruning_fn = lambda name: 'mlp' in name
            masks = get_constrained_mask(model, sparsity_ratios, f"{args.merge_path}-mlp-constrained", constraint_mask_path, pruning_fn, only_update_prune=False)
            pruning_fn = lambda name: 'mlp' in name or 'attn' in name
            masks = get_constrained_mask(model, sparsity_ratios, f"{args.merge_path}-mlpattn-constrained", constraint_mask_path, pruning_fn, only_update_prune=False)
        # masks = get_random_sparsity_masks(model, sparsity_ratios)
    
import yaml

def modify_yaml(file_path, new_model_path, new_base_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update the model path
    data['models'][0]['model'] = new_model_path
    data['base_model'] = new_base_path

    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

if __name__ == '__main__':
    import sys
    yaml_file_path = 'task_vector.yaml'
    new_model_path = sys.argv[1]
    new_base_name = sys.argv[2]
    if new_base_name == "mistral":
        new_base_path = "/scratch/gpfs/ashwinee/rlaif-cache/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24"
    elif new_base_name == "llama3":
        new_base_path = "/scratch/gpfs/ashwinee/rlaif-cache/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"
    elif new_base_name == "llama3-instruct":
        new_base_path = "/scratch/gpfs/ashwinee/rlaif-cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
    else:
        new_base_path = new_base_name
        # new_base_path = "/scratch/gpfs/ashwinee/rlaif-cache/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24"
    modify_yaml(yaml_file_path, new_model_path, new_base_path)
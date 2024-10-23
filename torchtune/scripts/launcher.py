import argparse
import json
import os
import socket
import subprocess as sb
import sys
from typing import Dict, Optional, Tuple
import time
import traceback

def download_model(model_output_folder: str) -> None:
    """
    Download a model if necessary.

    Args:
        model_output_folder (str): The folder to store the downloaded model.
        args (argparse.Namespace): Command-line arguments.
    """
    download_folder = os.path.join(args.modeldir, model_output_folder)
    full_command = f'tune download {args.model_id} --output-dir {download_folder} --hf-token {args.hf_token}'
    
    folder_exists = os.path.isdir(model_output_folder)

    if not args.use_downloaded_model:
        print("Downloading model...")
        delete_command=f'rm -rf {download_folder}'
        run_command(delete_command)
        
        delete_model_artifacts=f'rm -rf {args.modeldir}/*'
        run_command(delete_model_artifacts)
        
        list_models=f'ls -ltr {args.modeldir}'
        run_command(list_models)

        run_command(full_command)
    else:
        print("Using existing downloaded model.")

    
def set_custom_env(env_vars: Dict[str, str]) -> None:
    """
    Set custom environment variables.

    Args:
        env_vars (Dict[str, str]): A dictionary of environment variables to set.
                                   Keys are variable names, values are their corresponding values.

    Returns:
        None

    Raises:
        TypeError: If env_vars is not a dictionary.
        ValueError: If any key or value in env_vars is not a string.
    """
    if not isinstance(env_vars, dict):
        raise TypeError("env_vars must be a dictionary")

    for key, value in env_vars.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("All keys and values in env_vars must be strings")

    os.environ.update(env_vars)

    # Optionally, print the updated environment variables
    print("Updated environment variables:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")
        
def finetune_model() -> None:
    """
    Fine-tune a model using distributed training.

    Args:
        args: An object containing command-line arguments.
        num_of_hosts (int): Number of hosts for distributed training.
        leader (str): Address of the leader node.
        default_node_rank (int): Rank of the current node.

    Returns:
        None
    """
    print("***** Starting model fine-tuning *****")
    
    # Set custom environment variables
    custom_env: Dict[str, str] = {"HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
                                 }
    set_custom_env(custom_env)
    os.makedirs('/opt/ml/output', exist_ok=True)
    
    # Download the model
    download_model(args.model_output_dir)
    
    # Construct the configuration file path
    config_loc = os.path.join(args.configdir, args.tune_config_name)
    
    #run_command(del_model_dir_artifacts)
    print("***** PRINTING ls -ltr /opt/ml/code/ *****")
    run_command("ls -ltr /opt/ml/code/")
    
    # Construct the fine-tuning command
    full_command = (
        f'PYTHONPATH=$PYTHONPATH:{args.templatedir} '
        f'tune run '
        f'--master-addr {leader} '
        f'--master-port 7777 '
        f'--nnodes {num_of_hosts} '
        f'--node-rank {default_node_rank} '
        f'--nproc_per_node {args.gpus} '
        f'{args.tune_recipe} '
        f'--config {config_loc}'
    )
    
    # Run the fine-tuning command
    run_command(full_command)
    
def run_inference_original_model() -> None:
    """
    Run inference using the original model.

    This function downloads the model, prepares the configuration,
    and runs the inference command if it's on the primary node.

    Args:
        args: An object containing command-line arguments.

    Returns:
        None
    """
    print("***** Running inference on original model *****")

    # Download the model
    download_model(args.model_output_dir)
    
    # Construct the configuration file path
    config_loc = os.path.join(args.configdir, args.tune_config_name)
    
    # try:
    #     # Parse the prompt from JSON
    #     prompt = json.loads(args.prompt)
    # except json.JSONDecodeError:
    #     print("Error: Invalid JSON in prompt. Using prompt as-is.")
    #     prompt = args.prompt
    #     raise

    # Construct the inference command
    # full_command = (
    #     f'PYTHONPATH=$PYTHONPATH:{args.templatedir} '
    #     f'tune run generate '
    #     f'--config {config_loc} '
    #     f'prompt="{args.prompt}"'
    # )
    
    
    try:
        # Parse the prompt from JSON
        prompt = json.loads(args.prompt)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in prompt. Using prompt as-is.")
        prompt = args.prompt
        raise

    # Construct the inference command with updated PYTHONPATH
    full_command = (
        f'PYTHONPATH=$PYTHONPATH:{args.templatedir} '
        f'tune run generate '
        f'--config {config_loc} '
        f'prompt="{prompt}"'
    )
    
    
    if is_primary_node():
        print("Running inference on primary node...")
        run_command(full_command)
    else:
        print("Not on primary node. Skipping inference.")
        
        
def run_inference_trained_model() -> None:
    """
    Run inference using the trained model.

    This function downloads the model, prepares the configuration,
    sets up the Python path, and runs the inference command if it's on the primary node.

    Args:
        args: An object containing command-line arguments.

    Returns:
        None
    """
    print("***** Running inference on trained model *****")

    # Download the model
    download_model(args.model_output_dir)
    
    # Construct the configuration file path
    config_loc = os.path.join(args.configdir, args.tune_config_name)
    
#     try:
#         print(f"***args.prompt:{args.prompt}")
#         print(f"***args.prompt1:{args.prompt1}")
        
    
#         print("done")
#         # Parse the prompt from JSON
#         prompt = json.loads(args.prompt)
#     except json.JSONDecodeError:
#         print("Error: Invalid JSON in prompt. Using prompt as-is.")
#         prompt = args.prompt
#         raise

#    # print(f"prompt**:{prompt}")
    
#    # print(f"prompt[prompt]]**:{prompt['prompt']}")

#     # Construct the inference command with updated PYTHONPATH
#     full_command = (
#         f'PYTHONPATH=$PYTHONPATH:{args.templatedir} '
#         f'tune run generate '
#         f'--config {config_loc} '
#         f'prompt="{str(prompt["prompt"])}"'
#     )
    
    try:
        # Parse the prompt from JSON
        prompt = json.loads(args.prompt)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in prompt. Using prompt as-is.")
        prompt = args.prompt
        raise

    # Construct the inference command with updated PYTHONPATH
    full_command = (
        f'PYTHONPATH=$PYTHONPATH:{args.templatedir} '
        f'tune run generate '
        f'--config {config_loc} '
        f'prompt="{prompt}"'
    )
    
    if is_primary_node():
        print("Running inference on primary node...")
        run_command(full_command)
    else:
        print("Not on primary node. Skipping inference.")


def run_eval() -> None:
    """
    Run evaluation on the model.

    This function sets up the environment, downloads the model,
    and runs the evaluation command.

    Args:
        args: An object containing command-line arguments.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If any subprocess command fails.
    """
    print("***** Starting model evaluation *****")

    # Set custom environment variables
    custom_env: Dict[str, str] = {
        "HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
        "HF_TOKEN": args.hf_token
    }
    set_custom_env(custom_env)
    
    # Download the model
    download_model(args.model_output_dir)
    
    # Construct the configuration file path
    config_loc = os.path.join(args.configdir, args.tune_config_name)
    
    print("Listing configuration directory contents:")
    run_command(f"ls -ltr {args.configdir}")

    # Construct the evaluation command
    full_command = f'tune run eleuther_eval --config {config_loc}'
    
    print("Running evaluation command...")
    run_command(full_command)

    
def run_quant() -> None:
    """
    Run quantization on the model.

    This function sets up the environment, displays the configuration,
    and runs the quantization command if it's on the primary node.

    Args:
        args: An object containing command-line arguments.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If any subprocess command fails.
    """
    print("***** Starting model quantization *****")

    # Construct the configuration file path
    config_loc = os.path.join(args.configdir, args.tune_config_name)
    
    print("Listing configuration directory contents:")
    run_command(f"ls -ltr {args.configdir}")

    # Construct the quantization command
    full_command = f'PYTHONPATH=$PYTHONPATH:{args.configdir} tune run quantize --config {config_loc}'
    
    if is_primary_node():
        print("Running quantization on primary node...")
        run_command(full_command)
    else:
        print("Not on primary node. Skipping quantization.")
        
def run_command(command: str) -> None:
    """
    Run a shell command and handle potential errors.

    Args:
        command (str): The command to run.

    Raises:
        subprocess.CalledProcessError: If the command fails.
        ValueError: If the command string is empty.
        
    """

    print(f'\n\n ***** Executing command: {command} \n\n')

    try:
        # Start the timer
        start_time = time.time()
        
        result = sb.run(
            command,
            shell=True,
            capture_output=False,
            text=True,
            check=True
        )
        # End the timer
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        print(f"\n\n ***** Execution time for command: {command} is : {elapsed_time:.4f} seconds \n\n")

    except sb.CalledProcessError as e:
        report_error=1
        print(f"**** Command failed with error code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        raise
    except Exception as e:
        report_error=1
        print(f"****An unexpected error occurred: {e}")
        raise
        
        
    
def check_pytorch_version() -> Optional[str]:
    """
    Check and return the installed PyTorch version.

    This function runs a Python command to import torch and print its version.

    Returns:
        Optional[str]: The PyTorch version as a string, or None if an error occurred.

    Raises:
        subprocess.CalledProcessError: If the subprocess command fails.
    """
    try:
        # Run the command to get the PyTorch version
        result = sb.run(
            ['python', '-c', 'import torch; print(torch.__version__)'],
            capture_output=True,
            text=True,
            check=True
        )

        # Extract and strip the version string
        version = result.stdout.strip()

        print(f"Installed PyTorch version: {version}")
        return version

    except sb.CalledProcessError as e:
        print(f"Error occurred while checking PyTorch version: {e}")
        print(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return None
    
    
def is_primary_node():
    """
    Check if the current instance is the primary node in a SageMaker training job.
    Returns True if it is the primary node, otherwise False.
    """
    # SageMaker sets the SM_CURRENT_HOST environment variable
    current_host = os.getenv('SM_CURRENT_HOST')

    # SageMaker sets the SM_HOSTS environment variable as a JSON list of all host names
    hosts = os.getenv('SM_HOSTS')

    # Parse the hosts list
    if hosts:
        hosts = json.loads(hosts)

        # The primary node is typically the first in the list
        if current_host == hosts[0]:
            return True

    return False


def get_host_details() -> Tuple[int, int, str, str]:
    """
    Retrieve details about the current instance in a SageMaker training job.

    Returns:
        Tuple[int, int, str, str]: A tuple containing:
            - Number of hosts
            - Current node rank
            - IP address of the leader node
            - Current host name
    """
    current_host = os.getenv('SM_CURRENT_HOST', '')
    hosts_json = os.getenv('SM_HOSTS', '[]')

    if hosts_json:
        hosts = json.loads(hosts_json)
        num_of_hosts = len(hosts)
        leader = socket.gethostbyname(hosts[0])
        node_rank = hosts.index(current_host)
    else:
        num_of_hosts = 1
        node_rank = 0
        leader = '127.0.0.1'

    return num_of_hosts, node_rank, leader, current_host
    

def parse_arge():

    parser = argparse.ArgumentParser()

    # infra configuration
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS")))
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--configdir", type=str, default=os.environ["SM_CHANNEL_CONFIG"])
    parser.add_argument("--modeldir", type=str, default=os.environ["SM_CHANNEL_MODEL"])
    parser.add_argument("--templatedir", type=str, default=os.environ["SM_CHANNEL_TEMPLATES"])

    parser.add_argument("--tune_config_name", type=str, default=os.environ["SM_HP_TUNE_CONFIG_NAME"])
    parser.add_argument("--model_output_dir", type=str, default="SM_HP_MODEL_DIR_NAME") 
    parser.add_argument("--prompt", type=str, default="SM_HP_PROMPT") 
    
    parser.add_argument("--prompt1", type=str, default="SM_HP_PROMPT1") 

    parser.add_argument("--hf_token", type=str, default="SM_HP_HF_TOKEN") 
    parser.add_argument("--tune_recipe", type=str, default="lora_finetune_distributed")     
    parser.add_argument("--tune_action", type=str, default="SM_HP_TUNE_ACTION")
    parser.add_argument("--model_id", type=str, default="SM_HP_MODEL_ID")
    parser.add_argument('--use_downloaded_model', type=lambda x: str(x).lower() in ['true', '1', 't', 'y', 'yes'], help="A boolean flag")

    args = parser.parse_known_args()
    
    return args

def print_env_vars():
    print("***** Printing enviroment variables *****")
    
    print("Number of GPU's in the cluster {}".format(args.gpus))
    print("Local node rank is {}".format(args.node_rank))
    print("Master address is {}".format(leader))
    print(f'number of hosts:{num_of_hosts}')
    print(f'torchtune_config_8b_lora:{args.configdir}/{args.tune_config_name}')    
    print(f'use_downloaded_model:{args.use_downloaded_model}')
    print(f'type of use_downloaded_model:{type(args.use_downloaded_model)}')
    
    print(f'Action:{args.tune_action}')
    
    check_pytorch_version()
    
def completion_status():
    print("***** Finished my Task *****")
        
    list_model_dir=f'ls -ltr {args.modeldir}'
    run_command(list_model_dir)
        
    list_quantized_model_dir = f'ls -ltr {args.modeldir}/quantized'
    run_command(list_quantized_model_dir)
    
def training_function():
    
    print_env_vars()

    # Step 1: Map values to functions
    function_map = {
        "generate-trained": run_inference_trained_model,
        "generate-original": run_inference_original_model,
        "fine-tune": finetune_model,
        "run-eval":run_eval,
        "run-quant":run_quant
    }
    
    # Step 2: Iterate through the array and call the corresponding functions
    for value in args.tune_action.split(","):
        if value in function_map:
            print(f'value:{value}')
            try:
                function_map[value]()
            except Exception as e:
                print(f"An error occurred in function {value}: {e}")
                raise e
        else:
            print(f"No function defined for value {value}")


if __name__ == "__main__":
    
    report_error=0

    args, _ = parse_arge()
    
    num_of_hosts,default_node_rank,leader, current_host = get_host_details()

    try:
        print("Starting training...")
        training_function()
        
        if(report_error==1):
            sys.exit(1)
            
        print(f"Training completed with code: {report_error}")

    except Exception as e:
        # Log the error
        print(f"Error occurred during training: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())

        # Exit with a non-zero status code
        sys.exit(1)
    
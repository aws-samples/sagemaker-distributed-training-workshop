import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import re
import os
import random
import torch
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,  set_seed
from transformers.trainer_utils import get_last_checkpoint
#from trl.commands.cli_utils import TrlParser
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
import wandb
os.environ["NCCL_DEBUG"] = "INFO"


logger = logging.getLogger(__name__)


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

@dataclass
class ScriptArguments:
    """
    Arguments for the script execution.
    """

    train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training dataset, e.g., /opt/ml/input/data/train/"}
    )

    test_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test dataset, e.g., /opt/ml/input/data/test/"}
    )

    model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Model ID to use for SFT training"}
    )

    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum sequence length for SFT Trainer"}
    )

    hf_token: str = field(
        default="",
        metadata={"help": "Hugging Face API token"}
    )

# generate r1 prompt with a prefix for the model to already start with the thinking process
def generate_r1_prompt(tokenizer, numbers, target):
    r1_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
      },
      { 
        "role": "user",
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
      },
      {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
      }]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}

def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):

      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            rewards.append(0.0)
            continue
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           rewards.append(0.0)
           continue
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builti'ns__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
      except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint
def training_function(script_args, training_args):
    """
    Function to train a model with GRPO using specified script and training arguments.
    """
    
    ################
    # Dataset
    ################
    
    # Load datasets
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.train_dataset_path, "dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.test_dataset_path, "dataset.json"),
        split="train",
    )

    ####################
    # Model & Tokenizer
    ####################
    # Initialize a tokenizer by loading a pre-trained tokenizer configuration, using the fast tokenizer implementation if available.
    # our model we are going to use as policy 
    tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_id,
            use_fast=True
        )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    model_config = ModelConfig(
        model_name_or_path=script_args.model_id,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        use_peft=True,
        load_in_4bit=True,
    )

    train_dataset = train_dataset.map(lambda x: generate_r1_prompt(tokenizer, x["nums"], x["target"]))
    test_dataset = test_dataset.map(lambda x: generate_r1_prompt(tokenizer, x["nums"], x["target"]))

    

    # print random sample on rank 0
    if training_args.distributed_state.is_main_process:
        for index in random.sample(range(len(train_dataset)), 2):
            print("train_dataset")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print

    #if training_args.gradient_checkpointing:
    #    model_config.model_name_or_path.gradient_checkpointing_enable()

    
    ################
    # Training
    ################
    # Hyperparameters
    grpo_training_args = GRPOConfig(
        output_dir=training_args.output_dir,
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        logging_steps=10,
        max_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        # GRPO specific parameters
        max_prompt_length=256,
        max_completion_length=1024, # max length of the generated output for our solution
        num_generations=2,
        beta=0.001,
        
    )
    trainer = GRPOTrainer(
        model=model_config.model_name_or_path,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=grpo_training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_config),
    )
    # if trainer.accelerator.is_main_process:
    #     trainer.model.print_trainable_parameters()
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    #########################################
    # SAVE ADAPTER AND CONFIG FOR SAGEMAKER
    #########################################
    # save model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
    print("*** Model Saved")

    ##################
    # CLEAN UP 
    ##################
    del trainer
    torch.cuda.empty_cache()  # Clears the cache
    
    # load and merge
    if training_args.distributed_state.is_main_process:
        tokenizer.save_pretrained("/opt/ml/model/tokenizer")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print

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

if __name__ == "__main__":    
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()
    
    if(is_primary_node):
        print(f"training_args:{training_args}")
    print(f"script_args:{script_args}")
    print(f"Number of visible GPUs: {torch.cuda.device_count()}")
    
    custom_env: Dict[str, str] = {"HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
                                  "HF_TOKEN": script_args.hf_token,
                                  "FSDP_CPU_RAM_EFFICIENT_LOADING": "1",
                                   "ACCELERATE_USE_FSDP": "1",
                                   "WANDB_API_KEY": script_args.wandb_token,
                                   "WANDB_DIR" : "/opt/ml/output",
                                   "CUDA_VISIBLE_DEVICES": str(torch.cuda.device_count())
                                  }
    set_custom_env(custom_env)
    
    #########################
    # Init Weights & Biases metrics 
    #########################
    wandb.init(project="GRPO_metric")
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)


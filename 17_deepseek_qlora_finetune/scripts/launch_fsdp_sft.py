import logging
from dataclasses import dataclass, field
from typing import Optional

import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from typing import Dict, Optional, Tuple

from trl import SFTTrainer

import wandb
from transformers.integrations import WandbCallback

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
    
    wandb_token: str = field(
        default="",
        metadata={"help": "Wandb API token"}
    )


def generate_prompt_eval(data_point):
    """
    Generates a medical analysis prompt based on patient information.
    
    Args:
        data_point (dict): Dictionary containing target and meaning_representation keys
        
    Returns:
        dict: Dictionary containing the formatted prompt
    """
    full_prompt = f"""
    Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request. 
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

    ### Instruction:
    You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
    Please answer the following medical question. 

    ### Question:
    {data_point["input"]}

    ### Response:
    

    """
    analysis = f"""
    {data_point["output"]}

    """
    return {"prompt": prompt.strip(),"human_baseline": analysis.strip()}


def model_eval(model, tokenizer):
    # Load dataset from the hub
    test_set = load_dataset(dataset_name, name="all-processed", split="train[5%:]")
    
    # Add system message to each conversation
    columns_to_remove = list(test_set.features)
    
    test_dataset = test_set.map(
        generate_and_tokenize_prompt_inference,
        remove_columns=columns_to_remove,
        batched=False
    )

    dialogue = test_dataset['prompt']
    human_baseline_summaries = test_dataset['human_baseline']
    
    model_summaries = []
    instruct_model_summaries = []
    
    for _, dialogue in enumerate(test_dataset_1):
        
        #print(dialogue['prompt'])
        input_ids = tokenizer(dialogue['prompt'], return_tensors='pt').input_ids
    
        original_model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
        model_summaries.append(original_model_text_output)

    import evaluate
    
    rouge = evaluate.load('rouge')
    
    original_model_results = rouge.compute(predictions=model_summaries, 
                                       references=human_baseline_summaries[0: len(model_summaries)],
                                      use_aggregator=True)

    print(f'Original Model: \n{original_model_results}\n')  
    



def training_function(script_args, training_args):
    """
    Function to train a model with QLoRA using specified script and training arguments.
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
    tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_id,
            use_fast=True
        )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(text):
        result = tokenizer(
            text['prompt'],
            max_length=script_args.max_seq_length,
            padding="max_length",
            truncation=True
        )
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset = train_dataset.map(tokenize, remove_columns=["prompt"])
    test_dataset = test_dataset.map(tokenize, remove_columns=["prompt"])

    # print random sample on rank 0
    if training_args.distributed_state.is_main_process:
        for index in random.sample(range(len(train_dataset)), 2):
            print("train_dataset")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print

    ##############################
    # Model with nf4 quantization
    ##############################
    # Configure model quantization
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    # Configures 4-bit quantization settings for the model
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch_dtype,
    #     bnb_4bit_quant_storage=quant_storage_dtype,
    # )
    
    model_loading_params = {
        #"quantization_config": quantization_config,
        "torch_dtype": quant_storage_dtype,
        "use_cache": False if training_args.gradient_checkpointing else True
    }
    
    model = AutoModelForCausalLM.from_pretrained(
             script_args.model_id,
             cache_dir="/opt/ml/sagemaker/warmpoolcache",
             **model_loading_params
        )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    # peft_config = LoraConfig(
    #     lora_alpha=8,
    #     lora_dropout=0.05,
    #     r=16,
    #     bias="none",
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    #     # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    # )
    
    
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        #peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
        #callbacks=[CustomWandbCallback()]
    )
    # if trainer.accelerator.is_main_process:
    #     trainer.model.print_trainable_parameters()
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    #########################################
    # SAVE ADAPTER AND CONFIG FOR SAGEMAKER
    #########################################
    # save adapter
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
    print("*** Model Saved")

    ##################
    # CLEAN UP 
    ##################
    del model
    del trainer
    torch.cuda.empty_cache()  # Clears the cache
    
    # load and merge
    if training_args.distributed_state.is_main_process:
        tokenizer.save_pretrained("/opt/ml/model/llama3.1/tokenizer")
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
    
    # if(is_primary_node):
    #     print(f"training_args:{training_args}")
    #print(f"script_args:{script_args}")
    #print(f"Number of visible GPUs: {torch.cuda.device_count()}")
    
    custom_env: Dict[str, str] = {"HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
                                  "HF_TOKEN": script_args.hf_token,
                                  "FSDP_CPU_RAM_EFFICIENT_LOADING": "1",
                                   "ACCELERATE_USE_FSDP": "1",
                                 #  "WANDB_API_KEY": script_args.wandb_token,
                                 #  "WANDB_DIR" : "/opt/ml/output",
                                   "CUDA_VISIBLE_DEVICES": str(torch.cuda.device_count())
                                  }
    set_custom_env(custom_env)
    
    #########################
    # Init Weights & Biases metrics 
    #########################
   # wandb.init(project="mixtral_qlora_metric")
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)

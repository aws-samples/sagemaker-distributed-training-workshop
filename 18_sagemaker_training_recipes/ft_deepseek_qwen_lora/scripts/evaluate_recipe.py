import logging
import os
import random
import torch
from transformers import AutoTokenizer, TrainingArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from transformers import GenerationConfig
from typing import Dict, Optional, Tuple
import argparse
from datasets import load_dataset
import xtarfile as tarfile
from datasets import load_from_disk

import sys
sys.path.append("rouge")

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

def generate_and_tokenize_prompt(data_point):
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
    {data_point["Question"]}

    ### Response:

    """
    analysis = f"""
    {data_point["Complex_CoT"]}

    """
    return {"prompt": full_prompt.strip(),"human_baseline": analysis.strip()}

def get_summaries(model, test_dataset):        
    model_summaries = []

    tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            use_fast=True
        )
    
    for _, dialogue in enumerate(test_dataset):
        
        #print(dialogue['prompt'])
        input_ids = tokenizer(dialogue['prompt'], return_tensors='pt').input_ids
        
        model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=300, num_beams=1))
        original_model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        model_summaries.append(original_model_text_output)
    
    return model_summaries


def calculate_metrics(model, desc):

    print(f'Printing datasetname:{args.dataset_name}')

    test_dataset = load_dataset(args.dataset_name, 'en', split="train[:1%]")

    
    # Add system message to each conversation
    columns_to_remove = list(test_dataset.features)
    
    test_dataset = test_dataset.map(
        generate_and_tokenize_prompt,
        remove_columns=columns_to_remove,
        batched=False
    )
    
    test_dataset = test_dataset.select(range(10))

    model_summaries=get_summaries(model, test_dataset)
    human_baseline_summaries = test_dataset['human_baseline']

    import evaluate
    #import pandas as pd
    rouge = evaluate.load('rouge')
    
    model_results = rouge.compute(predictions=model_summaries, 
                        references=human_baseline_summaries[0: len(model_summaries)],
                                          use_aggregator=True)
    
    print(f'{desc}: \n{model_results}\n')     


def merge_and_save_model(model_id, adapter_dir, output_dir):
    #from peft import PeftModel

    print("Trying to load the base model. It might take a while without feedback")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        #torch_dtype=torch.float16,
        device_map="auto",
        #offload_folder="/opt/ml/model/"
    )
    
    print("Loaded base model")

    print(f"\n\n\n*** Generating Metrics on Base Model: {calculate_metrics(base_model,'Base Model')}\n\n\n")
    
    base_model.config.use_cache = False
    
    trained_model = AutoModelForCausalLM.from_pretrained(adapter_dir)

    print(f"\n\n\n*** Generating Metrics on Trained Model: {calculate_metrics(trained_model,'Trained Model')}\n\n\n")


def parse_arge():

    parser = argparse.ArgumentParser()

    # infra configuration
    parser.add_argument("--adapterdir", type=str, default=os.environ["SM_CHANNEL_ADAPTERDIR"])
    parser.add_argument("--testdata", type=str, default=os.environ["SM_CHANNEL_TESTDATA"])

    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="FreedomIntelligence/medical-o1-reasoning-SFT")
    
    args = parser.parse_known_args()
    
    return args

if __name__ == "__main__":    
    
    args, _ = parse_arge()
    
    custom_env: Dict[str, str] = {"HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
                                  "HF_TOKEN": args.hf_token
                                  }
    set_custom_env(custom_env)

    # launch training
    merge_and_save_model(args.model_id, args.adapterdir,"/opt/ml/model/merged/")

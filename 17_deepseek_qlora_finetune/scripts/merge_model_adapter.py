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
        

def create_test_prompt():
    dataset = load_dataset(
        "json",
        data_files=os.path.join(args.testdata, "dataset.json"),
        split="train"
    )
    
    # Shuffle the dataset and select the first row
    random_row = dataset.shuffle().select(range(1))[0]
    
    return random_row




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


    

def generate_text(model, prompt, max_length=500, num_return_sequences=1):
    # Encode the input prompt     
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            use_fast=True
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt_input=prompt['prompt'].split("### Response:")[0] + "### Response:"

    print(prompt_input)

    input_ids = tokenizer.encode(prompt_input, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    # Decode and return the generated text
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
        
    return generated_texts

def merge_and_save_model(model_id, adapter_dir, output_dir):
    from peft import PeftModel

    print("Trying to load a Peft model. It might take a while without feedback")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        #torch_dtype=torch.float16,
        device_map="auto",
        #offload_folder="/opt/ml/model/"
    )
    
    print("Loaded base model")
    
    prompt=create_test_prompt()
    
    print(f"\n\n\n*** Generating Inference on Base Model: {generate_text(base_model,prompt)}\n\n\n")

    print(f"\n\n\n*** Generating Metrics on Base Model: {calculate_metrics(base_model,'Base Model')}\n\n\n")
    
    base_model.config.use_cache = False
    
    # Load the adapter
    peft_model = PeftModel.from_pretrained(
        base_model, 
        adapter_dir, 
        #torch_dtype=torch.float16,  # Set dtype to float16
        #offload_folder="/opt/ml/model/"
    )
    
    print("Loaded peft model")
    model = peft_model.merge_and_unload()
    print("Merge done")
    
    print(f"***\n\n\n Generating Inference on Trained Model: {generate_text(model,prompt)}\n\n\n")

    print(f"\n\n\n*** Generating Metrics on Trained Model: {calculate_metrics(model,'Trained Model')}\n\n\n")

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving the newly created merged model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    base_model.config.save_pretrained(output_dir)


def parse_arge():

    parser = argparse.ArgumentParser()

    # infra configuration
    parser.add_argument("--adapterdir", type=str, default=os.environ["SM_CHANNEL_ADAPTER"])
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

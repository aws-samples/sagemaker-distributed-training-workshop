import random
import numpy as np
from tqdm.notebook import tqdm
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from typing import Iterable, Sequence, List

from torchtyping import TensorType

import transformers
from transformers import DataCollatorWithPadding
from transformers import pipeline, AutoTokenizer

from datasets import load_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR


class PromptPipeline():
    def __init__(self, prompts: List[str], max_prompt_length: int, tokenizer):
        super().__init__()

        prompts = tokenizer(prompts).input_ids

        self.tokenizer = tokenizer
        self.prompts = [prompt[-max_prompt_length:] for prompt in prompts]
        self.prompts = [{"input_ids": prompt, "attention_mask": [1] * len(prompt)} for prompt in self.prompts]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

@dataclass
class PPORLElement:
    query_tensor: TensorType["query_size"]
    response_tensor: TensorType["response_size"]
    logprobs: TensorType["response_size", "vocab_size"]
    values: TensorType["response_size"]
    rewards: TensorType["response_size"]


@dataclass
class PPORLBatch:
    query_tensors: TensorType["batch_size", "query_size"]
    response_tensors: TensorType["batch_size", "response_size"]
    logprobs: TensorType["batch_size", "response_size", "vocab_size"]
    values: TensorType["batch_size", "response_size"]
    rewards: TensorType["batch_size", "response_size"]


class PPORolloutStorage():
    def __init__(self, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(self, batch_size: int, shuffle: bool) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]):
            return PPORLBatch(
                pad_sequence(
                    [elem.query_tensor.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1),
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.values for elem in elems],
                    padding_value=0.0,
                    batch_first=True
                ),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)

class Actor():

    def __init__(
            self,
            prompt_pipeline,
            tokenizer,
            chunk_size = 128):
        
        self.prompt_pipeline = prompt_pipeline
        self.chunk_size = chunk_size

        self.prompt_pipeline_loader = self.prompt_pipeline.create_loader(self.chunk_size, shuffle=True)
        self.prompt_pipeline_iterator = iter(self.prompt_pipeline_loader)

        self.ref_model = Agent(config.model.model_path)
        self.ref_model_device = config.train.ref_model_device
        self.ref_model = self.ref_model.to(self.ref_model_device)
        
        self.tokenizer = tokenizer        
    

    def make_experience(self, model, num_rollouts = 128):
        model_device = next(model.parameters()).device
        
        ppo_rl_elements = []
        while len(ppo_rl_elements) < num_rollouts:
            try:
                batch = next(self.prompt_pipeline_iterator)
            except StopIteration:
                self.pipeline_iterator = iter(self.prompt_pipeline_loader)
                batch = next(self.prompt_pipeline_iterator)
            
            trajectories = generate(model, self.tokenizer, **batch.to(model_device))

            query_tensors = batch.input_ids
            response_tensors = trajectories[:, query_tensors.shape[1] :]

            all_tokens, attention_mask, position_ids = get_model_inputs(
                query_tensors.to(response_tensors.device), response_tensors, self.tokenizer.pad_token_id)
            with torch.no_grad():
                logits, values = model(
                    all_tokens, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids)
                ref_logits, _ = self.ref_model(
                    all_tokens.to(self.ref_model_device),
                    attention_mask=attention_mask.to(self.ref_model_device),
                    position_ids=position_ids.to(self.ref_model_device))
            
            all_tokens = all_tokens.cpu()
            logits = logits.cpu()
            ref_logits = ref_logits.cpu()

            logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], all_tokens[:, 1:])
            
            n = trajectories.shape[0]
            values = values.cpu()[:, :-1]
            query_tensors = query_tensors.cpu()
            response_tensors = response_tensors.cpu()
            
            start = query_tensors.shape[1] - 1
            ends = start + attention_mask[:, start:].sum(1)
            all_values = [values[i, start : ends[i]] for i in range(n)]
            all_logprobs = [logprobs[i, start : ends[i]] for i in range(n)]
            
            texts = self.tokenizer.batch_decode(trajectories, skip_special_tokens=True)
            scores = torch.tensor(reward_fn(texts), device='cpu', dtype=torch.float)

            rewards = -config.method.kl_coef * (logprobs - ref_logprobs)
            all_rewards = [None] * n
            for i in range(n):
                rs = rewards[i][start : ends[i]]
                rs[-1] = scores[i]
                all_rewards[i] = rs
            
            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_tensors[i],
                    response_tensor=response_tensors[i],
                    logprobs=all_logprobs[i],
                    values=all_values[i],
                    rewards=all_rewards[i],
                )
                for i in range(n)
            ]

            ppo_rl_elements += new_ppo_rl_elements

        return ppo_rl_elements, scores.mean().item()

class Agent(nn.Module):
    def __init__(self, model_path, num_layers_unfrozen=0):
        super().__init__()

        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(model_path, cache_dir="./models")

        self.logit_head = self.base_model.get_output_embeddings()
        
        n_embd = self.base_model.lm_head.in_features
        self.value_head = nn.Sequential(
            nn.Linear(n_embd, n_embd*2),
            nn.ReLU(),
            nn.Linear(n_embd*2, 1))
        
        freeze_bottom_causal_layers(self.base_model, num_layers_unfrozen)
        
    
    def generate(self, input_ids, **x):
        return self.base_model.generate(input_ids, **x)

    def forward(self, input_ids, attention_mask, position_ids):

        transformer_outputs = self.base_model.transformer(input_ids=input_ids,
                                                          attention_mask=attention_mask,
                                                          position_ids=position_ids)
    
        last_hidden_state = transformer_outputs.last_hidden_state
        lm_logits = self.logit_head(last_hidden_state)
        value = self.value_head(last_hidden_state).squeeze(-1)
        
        return lm_logits, value

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

from utils import PromptPipeline, PPORLElement, PPORLBatch, PPORolloutStorage, Actor, Agent

def generate(model, tokenizer, input_ids, attention_mask=None, **kwargs):
    
    generate_kwargs = dict(
        config.method.gen_kwargs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id)

    kwargs = dict(generate_kwargs, **kwargs)

    with torch.no_grad():
        generated_results = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    return generated_results


def get_model_inputs(query_tensors, response_tensors, pad_token_id):
    tokens = torch.cat((query_tensors, response_tensors), dim=1)[:, -config.train.seq_length :]
    attention_mask = (tokens.not_equal(pad_token_id).long().to(tokens.device))
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    return tokens, attention_mask, position_ids


def logprobs_from_logits(logits, labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    hidden_layers = model.transformer.h
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)

sentiment_fn = pipeline(
    model = "lvwerra/distilbert-imdb",
    top_k=2,
    batch_size=config.method.num_rollouts,
    device=config.train.reward_model_device,
)

def get_positive_score(scores):
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

def reward_fn(samples: List[str]) -> List[float]:
    sentiments = list(map(get_positive_score, sentiment_fn(samples)))
    return sentiments

imdb = load_dataset("imdb", split="train+test")

prompts = [" ".join(review.split()[:config.method.prompt_size]) for review in imdb["text"]]

tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_ida = tokenizer.eos_token_id
tokenizer.padding_side = "left"
pad_token_id = 50256

max_prompt_length = (config.train.seq_length - config.method.gen_kwargs["max_new_tokens"])
test_prompt_pipeline = PromptPipeline(prompts, max_prompt_length, tokenizer)

model = Agent(config.model.model_path, config.model.num_layers_unfrozen).to(config.train.model_device)

input_ids = tokenizer.batch_encode_plus(
    ["my feeling about the movie", "this is", "I can tell with certainty"],
    return_tensors='pt',
    padding=True)['input_ids']

print (input_ids)

model_device = next(model.parameters()).device
output_ids = generate(model, tokenizer, input_ids.to(model_device), max_new_tokens=config.method.gen_kwargs["max_new_tokens"])

generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

print (generated_text)

reward_fn(generated_text)

prompt_pipeline = PromptPipeline(prompts, config.train.seq_length, tokenizer)

actor = Actor(prompt_pipeline, tokenizer, chunk_size=config.method.chunk_size)

store = PPORolloutStorage(tokenizer.pad_token_id)

opt = torch.optim.Adam(model.parameters(), **config.optimizer.kwargs)
scheduler = CosineAnnealingLR(opt, **config.scheduler.kwargs)

n_updates_per_batch = config.method.ppo_epochs
total_steps = 400 # TODO: fix this

tbar = tqdm(initial=0, total=total_steps)

for _ in range(config.train.epochs):
    
    store.clear_history()
    rollouts, score = actor.make_experience(model, config.method.num_rollouts)
    store.push(rollouts)
    train_dataloader = store.create_loader(config.train.batch_size, shuffle=True)
    
    for batch in train_dataloader:
        for _ in range(n_updates_per_batch):

            loss, reward = loss_fn(batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()
            tbar.update()
    
    tbar.set_description(f"| score: {score:.3f} |")

input_ids = tokenizer.batch_encode_plus(
    ["my feeling about the movie", "this is", "I can tell with certainty"],
    return_tensors='pt',
    padding=True)['input_ids']
input_ids

model_device = next(model.parameters()).device
output_ids = generate(
    model,       
    tokenizer,
    input_ids.to(model_device),
#     min_length=20,
#     max_new_tokens=100,
#     do_sample=True,
#     top_p=0.92, 
#     top_k=0
)
generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
rewards = reward_fn(generated_text)
print(generated_text[0].replace('\n', ' ') + '\n', rewards[0])
print(generated_text[1].replace('\n', ' ') + '\n', rewards[1])
print(generated_text[2].replace('\n', ' ') + '\n', rewards[2])
print('all rewards mean:',np.mean(rewards))

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


#####################
## PyTorch Objects ## 
#####################

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

#####################
## Util Functions  ## 
#####################

def whiten(x):
    var, mean = torch.var_mean(x)
    return (x - mean) * torch.rsqrt(var + 1e-8)


def gae(
    values,
    rewards,
):
    advantages = torch.zeros_like(rewards, device=rewards.device)
    last_advantage = 0
    last_value = 0
    
    with torch.no_grad():
        for t in reversed(range(rewards.shape[1])):
            delta = rewards[:, t] + config.method.gamma * last_value - values[:, t]
            last_advantage = delta + config.method.gamma * config.method.lam * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[:, t]

        returns = advantages + values
    
    if config.method.use_whitening:
        advantages = whiten(advantages)
    
    return advantages, returns

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

def get_positive_score(scores):
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

def reward_fn(samples: List[str]) -> List[float]:
    sentiments = list(map(get_positive_score, sentiment_fn(samples)))
    return sentiments

def ppo_loss(
    logprobs,     
    values,       
    old_logprobs, 
    old_values,   
    advantages,   
    returns,      
    mask,         
):

    values_clipped = torch.clamp(
        values,
        old_values - config.method.cliprange_value,
        old_values + config.method.cliprange_value,
    )
    
    n = mask.sum()
    
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n

    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - config.method.cliprange, 1.0 + config.method.cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
    pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

    loss = pg_loss + config.method.vf_coef * vf_loss
    
    return loss

def loss_fn(batch):
    model_device = next(model.parameters()).device
    query_tensors = batch.query_tensors.to(model_device)
    response_tensors = batch.response_tensors.to(model_device)
    old_logprobs = batch.logprobs.to(model_device)
    old_values = batch.values.to(model_device)
    old_rewards = batch.rewards.to(model_device)
    
    response_length = old_rewards.shape[1]

    advantages, returns = gae(old_values, old_rewards)

    tokens, attention_mask, position_ids = get_model_inputs(query_tensors, response_tensors, tokenizer.pad_token_id)

    logits, values_pred = model(tokens,
                                attention_mask=attention_mask,
                                position_ids=position_ids)
    values_pred = values_pred[:, :-1]
    logprobs = logprobs_from_logits(logits[:, :-1, :], tokens[:, 1:])
    attention_mask = attention_mask[:, :-1]

    start = query_tensors.shape[1] - 1
    end = start + response_length
    logprobs, values_pred, mask = (
        logprobs[:, start:end],
        values_pred[:, start:end],
        attention_mask[:, start:end],
    )

    loss = ppo_loss(
        logprobs=logprobs,
        values=values_pred,
        old_logprobs=old_logprobs,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        mask=mask,
    )

    return loss, old_rewards[:,-1].mean().item()

############################
## Model and Data Configs ## 
############################

config = {
    'train': {
        'seed': 2023,
        'seq_length': 1024,
        'epochs': 50,
        'total_steps': 5000,
        'batch_size': 64,
        'eval_interval': 100,
        'model_device':'cuda:0',
        'ref_model_device':'cpu',
        'reward_model_device':'cpu'},
    'model': {
        'model_path': 'lvwerra/gpt2-imdb', #'edbeeching/gpt-neo-1.3B-imdb',
        'tokenizer_path': 'lvwerra/gpt2-imdb', #'edbeeching/gpt-neo-1.3B-imdb',
        'num_layers_unfrozen': 1},
    'optimizer': {
        'name': 'adamw',
        'kwargs': {'lr': 0.0001,
        'betas': [0.9, 0.95],
        'eps': 1e-08,
        'weight_decay': 1e-06}},
    'scheduler': {
        'name': 'cosine_annealing',
        'kwargs': {
            'T_max': 10000, 'eta_min': 0.0001}},
    'method': {
        'use_whitening': True,
        'prompt_size': 10,
        'num_rollouts': 128,
        'chunk_size': 128,
        'ppo_epochs': 4,
        'kl_coef': 0.05,
        'horizon': 10000,
        'gamma': 1,
        'lam': 0.95,
        'cliprange': 0.2,
        'cliprange_value': 0.2,
        'vf_coef': 1,
        'scale_reward': False,
        'ref_mean': None,
        'ref_std': None,
        'cliprange_reward': 10,
        'gen_kwargs': {
            'max_new_tokens': 60,
            'top_k': 0,
            'top_p': 1.0,
            'do_sample': True}}}

config = DictConfig(config)

random.seed(config.train.seed)
np.random.seed(config.train.seed)
torch.manual_seed(config.train.seed)
torch.cuda.manual_seed(config.train.seed)

sentiment_fn = pipeline(
    model = "lvwerra/distilbert-imdb",
    top_k=2,
    batch_size=config.method.num_rollouts,
    device=config.train.reward_model_device,
)

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


###############
## Main loop ## 
###############

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


################
## Eval steps ## 
################

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


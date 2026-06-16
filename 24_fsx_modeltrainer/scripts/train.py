import os
import argparse
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    get_scheduler,
    SchedulerType,
    FalconConfig,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from datasets import load_from_disk
import torch
import torch.distributed as dist
from utils import create_dataloaders, save_model
from tqdm import tqdm

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="tiiuae/falcon-7b")
    parser.add_argument("--dataset_path", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] >= 8 else False,
    )
    parser.add_argument("--fsdp", type=str, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--limit_all_gathers", type=bool, default=False)
    parser.add_argument("--forward_prefetch", type=bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--optimizer", type=str, default="adamw_torch")
    parser.add_argument("--cache_dir", type=str, default=None)

    args, _ = parser.parse_known_args()
    return args


def training_function(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    set_seed(args.seed)

    dataset_path = args.dataset_path
    if os.path.isdir(os.path.join(dataset_path, "processed", "data")):
        dataset_path = os.path.join(dataset_path, "processed", "data")
    elif os.path.isdir(os.path.join(dataset_path, "allenai", "processed", "data")):
        dataset_path = os.path.join(dataset_path, "allenai", "processed", "data")

    if rank == 0:
        print(f"Loading dataset from: {dataset_path}")

    dataset = load_from_disk(dataset_path)

    config = FalconConfig(
        vocab_size=65024,
        use_cache=True,
        parallel_attn=True,
        num_hidden_layers=32,
        num_attention_heads=71,
        new_decoder_architecture=False,
        multi_query=True,
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
        hidden_size=4544,
        hidden_dropout=0.0,
        eos_token_id=11,
        bos_token_id=11,
        bias=False,
    )

    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, rank, world_size,
        args.seed, args.per_device_train_batch_size, args.per_device_train_batch_size,
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={FalconDecoderLayer},
    )

    torch.cuda.set_device(local_rank)
    dtype = torch.bfloat16

    mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=args.forward_prefetch,
        limit_all_gathers=args.limit_all_gathers,
        device_id=torch.cuda.current_device(),
    )

    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper, offload_to_cpu=True, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    check_fn = lambda submodule: isinstance(submodule, FalconDecoderLayer)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if rank == 0:
        print(f"Number of update steps per epoch: {num_update_steps_per_epoch}")

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    device = torch.device(f"cuda:{local_rank}")

    for epoch in range(args.num_train_epochs):
        model.train()
        total_steps = 0
        fsdp_loss = torch.zeros(2).to(local_rank)

        for _, batch in enumerate(tqdm(train_dataloader, disable=not (rank == 0))):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output["loss"]
            loss.backward()
            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(batch["input_ids"])

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_steps += 1
            if args.max_steps is not None and total_steps > args.max_steps:
                break

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        train_loss = fsdp_loss[0] / fsdp_loss[1]
        train_ppl = torch.exp(train_loss)

        if rank == 0:
            print(f"******{epoch=}: {train_ppl=} {train_loss=}******")

        model.eval()
        fsdp_eval_loss = torch.zeros(2).to(local_rank)
        for steps, batch in enumerate(tqdm(eval_dataloader, disable=not (rank == 0))):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs["loss"]
            fsdp_eval_loss[0] += loss.item()
            fsdp_eval_loss[1] += len(batch["input_ids"])
            if args.max_steps is not None and steps > args.max_steps:
                break

        dist.all_reduce(fsdp_eval_loss, op=dist.ReduceOp.SUM)
        eval_loss = fsdp_eval_loss[0] / fsdp_eval_loss[1]
        eval_ppl = torch.exp(eval_loss)

        if rank == 0:
            print(f"*******{epoch=}: {eval_ppl=} {eval_loss=}*******")

        if args.max_steps is not None and total_steps > args.max_steps:
            break

    save_model(model, tokenizer, args.model_dir, rank)
    if rank == 0:
        print("Training done!")
    dist.barrier()


def main():
    dist.init_process_group("nccl")
    args = parse_args()
    training_function(args)


if __name__ == "__main__":
    main()

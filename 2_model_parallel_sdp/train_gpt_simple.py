import argparse
import collections
import logging
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import smdistributed.modelparallel
import smdistributed.modelparallel.torch as smp
import torch
import torch.nn as nn
import torch.utils.data
import transformers
from data_pipeline import create_pretraining_dataloader
from learning_rates import AnnealingLR, get_learning_rate_scheduler,get_param_groups_by_weight_decay
from memory_tracker import memory_status, memory_status_cpu
from sharded_data_parallel_checkpoint import get_buffer_names, get_param_shapes
from smdistributed.modelparallel.torch.nn.huggingface.gpt2 import (
    translate_hf_state_dict_to_smdistributed_gpt2,
    translate_state_dict_to_hf_gpt2,
)
from torch import optim
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process

logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@smp.step
def train_step(model, optimizer, input_ids, attention_mask, args):
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)["loss"]
    model.backward(loss)

    return loss

@smp.step
def test_step(model, input_ids, attention_mask):
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)["loss"]
    return loss


def eval_model(model, dataloader, num_batches):
    model = model.eval()
    n_batches = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader):
            input_ids, _, attention_mask, _, _ = input_data
            if batch_idx >= num_batches:
                break

            loss += test_step(model, input_ids, attention_mask).reduce_mean()
            n_batches += 1

    if n_batches > 0:
        torch.distributed.all_reduce(loss, group=smp.get_dp_process_group())
        loss /= smp.dp_size()
        loss /= n_batches
        loss = loss.item()
        ppl = math.exp(loss)
    else:
        loss = -1.0
        ppl = -1.0

    return loss, ppl


def train(
    model,
    optimizer,
    lr_scheduler,
    model_config,
    start_train_path_index,
    start_batch_index,
    num_params,
    total_steps,
    args,
):
    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="before train step")
    model.train()
    
    train_paths = sorted(
        [
            os.path.join(args.training_dir, p)
            for p in os.listdir(args.training_dir)
        ]
    )

    train_dataloader = create_pretraining_dataloader(
        [train_paths[start_train_path_index]],
        args.train_batch_size,
        args.max_context_width,
        seed=args.seed,
        shuffle=True,
    )

    if args.validation_freq is not None:
        # load all validation examples
        if smp.rank() == 0:
            print("Creating val dataloader")
        val_paths = sorted(
            [
                os.path.join(args.test_dir, p)
                for p in os.listdir(args.test_dir)
            ]
        )
        val_dataloader = create_pretraining_dataloader(
            val_paths,
            args.val_batch_size,
            args.max_context_width,
            seed=args.seed,
            shuffle=True,
        )
        if smp.rank() == 0:
            print("Created val dataloader")

    start = time.time()
    throughput = None
    to_save = {"loss": [], "val_loss": []}
    loss_metric = 0

    def grad_accumulation_boundary(batch_idx):
        return batch_idx % args.gradient_accumulation == args.gradient_accumulation - 1

    # Set the same seed for computation
    set_seed(args.seed)

    for index in range(start_train_path_index, args.epochs * len(train_paths)):
        next_train_path_index = (index + 1) % len(train_paths)
        curr_train_path_index = index % len(train_paths)

        if total_steps >= args.max_steps:
            break

        for batch_idx, input_data in enumerate(train_dataloader):
            if batch_idx < start_batch_index:
                if smp.rank() == 0:
                    print(
                        f"Resuming from saved batch index {start_batch_index}, skipping batch {batch_idx}..."
                    )
                if start_batch_index == len(train_dataloader):
                    # If saving at the last batch of the file, read from the next file
                    start_batch_index = 0
                    break
                continue
            else:
                start_batch_index = 0

            input_ids, _, attention_mask, _, _ = input_data
            if total_steps >= args.max_steps:
                break

            step_start = time.time()

            if grad_accumulation_boundary(batch_idx - 1):
                optimizer.zero_grad(set_to_none=True)

            loss = train_step(model, optimizer, input_ids, attention_mask, args).reduce_mean()

            if not args.validation_freq:
                loss_metric = loss.item()
            
            if args.enable_memory_profiling > 0:
                memory_status_cpu("After_train_step_cpu")
                memory_status(msg="After_train_step")

            if grad_accumulation_boundary(batch_idx):
                if args.fp16:
                    optimizer.clip_master_grads(args.grad_clip)
                    
                optimizer.step()
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()

                if args.enable_memory_profiling > 0:
                    memory_status(msg="After_opt_step")

            total_steps += 1
            time_elapsed = time.time() - start
            step_time = time.time() - step_start
            sample_processed = input_ids.shape[0] * smp.dp_size()
            throughput = sample_processed / step_time
            tokens_per_gpu = input_ids.shape[0] * input_ids.shape[1]

            # Based on the formula in https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/
            tflops_per_gpu = 8 * num_params * tokens_per_gpu / step_time / 1e12
            if smp.rank() == 0 and not total_steps % args.logging_freq:
                print(
                    f"({int(time_elapsed)}s), Batch {total_steps - 1} Loss: {loss.item()}, Speed: {throughput} samples/sec, TFLOPS/GPU: {tflops_per_gpu}"
                )

            # evaluate on validation
            if args.validation_freq and not (total_steps % args.validation_freq):
                model = model.eval()
                val_loss, val_ppl = eval_model(
                    model, val_dataloader, args.validation_batches
                )
                if is_main_process(smp.rank()):
                    print(
                        f"({int(time.time()-start)}s) Batch {total_steps - 1} Validation loss: {val_loss}"
                    )
                    print(
                        f"({int(time.time()-start)}s) Batch {total_steps - 1} Validation perplexity: {val_ppl}"
                    )
                loss_metric = val_loss
                model = model.train()

            # checkpoint
            if not (total_steps % args.checkpoint_freq):
                user_content = {
                    "cli_args": args.__dict__,
                    "num_params": num_params,
                    "total_steps": total_steps,
                    "start_train_path_index": curr_train_path_index,
                    "model_config": model_config,
                    "start_batch_index": batch_idx+1,
                }
                # to reconstruct the full model
                if args.sharded_data_parallel_degree > 1:
                    user_content["buffer_names"] = get_buffer_names(model)
                    user_content["param_shapes"] = get_param_shapes(model, optimizer)
                user_content["lr_scheduler"] = lr_scheduler.state_dict()
                smp.save_checkpoint(args.checkpoint_dir,
                    tag=f"total_steps{total_steps}",
                    partial=True,
                    model=model,
                    optimizer=optimizer,
                    user_content=user_content,
                    num_kept_partial_checkpoints=args.num_kept_checkpoints)


        if total_steps >= args.max_steps:
            break

        del train_dataloader

        train_dataloader = create_pretraining_dataloader(
            [train_paths[next_train_path_index]],
            args.train_batch_size,
            args.max_context_width,
            seed=args.seed,
            shuffle=True,
        )

    return total_steps, throughput, loss_metric


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.

    opt_grp = parser.add_argument_group(
        title="optimization", description="arguments for optimization"
    )
    opt_grp.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="batch size per dp rank, for tensor parallelism degree 8 with pipeline parallel degree 1 this means 8*this batch size per node",
    )
    opt_grp.add_argument("--val_batch_size", type=int, default=4)
    opt_grp.add_argument("--max_steps", type=int, default=5000)
    opt_grp.add_argument("--seed", type=int, default=12345)
    opt_grp.add_argument("--fp16", default=0, type=int, help="automatic mixed precision training")
    opt_grp.add_argument("--bf16", default=0, type=int, help="automatic mixed precision training")
    opt_grp.add_argument("--sharded_data_parallel_degree", default=1, type=int)
    opt_grp.add_argument("--grad_clip", default=1.0, type=float, help="gradient clipping")
    opt_grp.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    opt_grp.add_argument(
        "--beta1", default=0.9, type=float, help="beta1 parameter for Adam optimizer"
    )
    opt_grp.add_argument(
        "--beta2", default=0.95, type=float, help="beta2 parameter for Adam optimizer"
    )
    opt_grp.add_argument(
        "--activation_checkpointing",
        type=int,
        default=1,
        help="enable gradient checkpointing to reduce memory consumption",
    )
    parser.add_argument(
        "--logging_freq", type=int, default=1, help="number of iterations between logging"
    )

    # I/O
    io_grp = parser.add_argument_group(title="io", description="location for input and output")
    io_grp.add_argument(
        "--epochs", type=int, default=3, help="times of iterating over the training dataset"
    )
    io_grp.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    io_grp.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/opt/ml/checkpoints",
        help="Saves partial checkpoints (model, optimizer) to this dir, and loads latest checkpoint from this if load_partial is specified.",
    )
    io_grp.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="Saves full model for inference to this dir. Also used if load_full is given to load the model. Note the lack of optimizer state here.",
    )
    io_grp.add_argument("--training-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    io_grp.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    io_grp.add_argument(
        "--save_final_full_model",
        type=int,
        default=0,
        help="Enabling this will save a combined model only at the end",
    )
    io_grp.add_argument("--load_partial", type=int, default=0, help="Load from partial checkpoints")
    io_grp.add_argument("--load_full", type=int, default=0, help="Load from full checkpoints")

    # configure model size
    model_grp = parser.add_argument_group(
        title="model", description="arguments to describe model configuration"
    )
    model_grp.add_argument("--max_context_width", type=int, default=1024)
    model_grp.add_argument("--vocab_size", type=int, default=50264)
    model_grp.add_argument("--hidden_width", type=int, default=768)
    model_grp.add_argument("--num_layers", type=int, default=12)
    model_grp.add_argument("--num_heads", type=int, default=12)
    model_grp.add_argument("--resid_pdrop", type=float, default=0.1)
    model_grp.add_argument("--embd_pdrop", type=float, default=0.1)
    model_grp.add_argument("--attn_pdrop", type=float, default=0.1)
    model_grp.add_argument("--summary_first_pdrop", type=float, default=0.1)
    model_grp.add_argument("--use_distributed_transformer", type=int, default=0, help="Use distributed transformer")

    smp_grp = parser.add_argument_group(title="smp", description="smp")
    smp_grp.add_argument("--activation_strategy", type=str, default="each")
    smp_grp.add_argument("--offload_activations", type=int, default=0)
    smp_grp.add_argument("--delayed_param", type=int, default=0)
    smp_grp.add_argument("--attention_in_fp32", type=int, default=0)
    smp_grp.add_argument("--activation_loading_horizon", type=int, default=4)
    smp_grp.add_argument("--skip_tracing", type=int, default=1)
    smp_grp.add_argument("--query_key_layer_scaling", type=int, default=1)
    smp_grp.add_argument("--fused_softmax", type=int, default=1)
    smp_grp.add_argument("--fused_bias_gelu", type=int, default=1)
    smp_grp.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument(
        "--num_kept_checkpoints",
        type=int,
        default=5,
        help="how many checkpoints to keep before deleting",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10000,
        help="number of iterations between checkpointing",
    )
    parser.add_argument(
        "--validation_freq",
        type=int,
        default=None,
        help="number of iterations to print validation loss",
    )
    parser.add_argument(
        "--validation_batches",
        type=int,
        default=10,
        help="number of batches to estimate validation loss",
    )
    parser.add_argument("--use_fsx", type=int, default=0, help="Using FSx for checkpointing")
    parser.add_argument(
        "--enable_memory_profiling", type=int, default=0, help="Enable memory profile"
    )

    # learning rate
    lr_grp = parser.add_argument_group(
        title="lr", description="arguments for learning rate schedule"
    )
    lr_grp.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    lr_grp.add_argument(
        "--lr_decay_style",
        type=str,
        default="linear",
        choices=["constant", "linear", "cosine", "exponential", "plateau"],
        help="Learning rate decay function.",
    )
    lr_grp.add_argument(
        "--lr_decay_iters",
        type=int,
        default=None,
        help="number of iterations to decay learning rate over," " If None defaults to train iters",
    )
    lr_grp.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="Minumum value for learning rate. The scheduler" "clip values below this threshold.",
    )
    lr_grp.add_argument(
        "--warmup",
        type=float,
        default=0.01,
        help="Percentage of total iterations to warmup on "
        "(.01 = 1 percent of all training iters).",
    )
    lr_grp.add_argument(
        "--plateau",
        type=float,
        default=0.4,
        help="Percentage of total iterations to keep at max if using plateau lr",
    )

    args, _ = parser.parse_known_args()
    return args

def compute_num_params(model):
    num_params = 0
    seen = set()
    for p in model.parameters():
        if p not in seen:
            seen.add(p)
            if hasattr(p, "ds_shape"):
                num_params += np.prod(p.ds_shape) 
            else:
                num_params += np.prod(p.size())
    
    return num_params 

def main():
    args = parse_args()

    # any value here is overriden by the config set in notebook when launching the sagemaker job
    smp_config = {
        "ddp": True,
        "fp16": args.fp16 > 0,
        "bf16": args.bf16 > 0,
        "offload_activations": args.offload_activations > 0,
        "delayed_parameter_initialization": args.delayed_param > 0,
        "activation_loading_horizon": args.activation_loading_horizon,
        "skip_tracing": args.skip_tracing > 0,
        "sharded_data_parallel_degree": args.sharded_data_parallel_degree,
    }

    smp.init(smp_config)

    if smp.rank() == 0:
        print("Arguments:", args.__dict__)
        print(f"Transformers version: {transformers.__version__}")
        print(f"smdistributed.modelparallel version: {smdistributed.modelparallel.__version__}")
        print(f"smdistributed config: {smp_config}")

    if args.save_final_full_model and smp.rank() == 0:
        print(
            f"[Warning] Note that save_final_full_model only saves the final model at the end of all steps. It does not save optimizer state. Optimizer state is only saved with partial models which are saved at checkpointing_freq during training. If you want to restart training you need partial checkpoints."
        )

    model_config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.max_context_width,
        n_embd=args.hidden_width,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=args.resid_pdrop,
        embd_pdrop=args.embd_pdrop,
        attn_pdrop=args.attn_pdrop,
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=args.summary_first_pdrop,
        # gradient_checkpointing=args.gradient_checkpointing > 0,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
        return_dict=True,
    )

    set_seed(args.seed)

    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="before model creation")

    if args.fp16 and args.bf16:
        raise ValueError("FP16 and BF16 cannot be simultaneously enabled.")
    elif args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.get_default_dtype()

    with smp.model_creation(
        tensor_parallelism=smp.tp_size() > 1 or args.use_distributed_transformer > 0,
        dtype=dtype,
        attention_in_fp32=args.attention_in_fp32 > 0,
        query_key_layer_scaling=args.query_key_layer_scaling > 0 and args.bf16 < 1,
        fused_softmax=args.fused_softmax > 0,
        fused_bias_gelu=args.fused_bias_gelu > 0,
        ):
            model = AutoModelForCausalLM.from_config(model_config)
    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="after model creation")

    num_params = compute_num_params(model)
    if smp.rank() == 0:
        print(f"# total parameters: {num_params}")

    # smdistributed: Set the device to the GPU ID used by the current process.
    # Input tensors should be transferred to this device.
    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda")

    # smdistributed: Use the DistributedModel container to provide the model
    # to be partitioned across different ranks. For the rest of the script,
    # the returned DistributedModel object should be used in place of
    # the model provided for DistributedModel class instantiation.
    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="before dist model creation")
    model = smp.DistributedModel(model, trace_device="gpu", backward_passes_per_step=args.gradient_accumulation)
    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="after dist model creation")

    m = model.get_module()
    if args.use_distributed_transformer > 0:
        transformer_layers = m.transformer.seq_layers
    else:
        transformer_layers = m.transformer.h

    param_groups = get_param_groups_by_weight_decay(m)
    optimizer = optim.AdamW(
        param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.activation_checkpointing:
        if args.use_distributed_transformer or smp.tp_size() > 1:
            smp.set_activation_checkpointing(transformer_layers, strategy=args.activation_strategy)
        else:
            for c in transformer_layers.children():
                smp.set_activation_checkpointing(c)

    optimizer = smp.DistributedOptimizer(
        optimizer, 
        static_loss_scale=None, 
        dynamic_loss_scale=True,
        dynamic_loss_args={"scale_window": 1000, "min_scale": 1, "delayed_shift": 2},
        )
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.enable_memory_profiling > 0:
        model.register_post_partition_hook(
            lambda model, optimizer: memory_status(msg="After_partition")
        )

    # load after wrapping model and optimizer with smp Distributed...
    if args.load_full or args.load_partial:
        if args.load_partial and args.load_full:
            print(
                "Since both --load_partial and --load_full set, will try to load from full checkpoint."
                "If the intention is to load from partial checkpoint, please don't set --load_full"
            )
        partial = not args.load_full
        path = args.checkpoint_dir if partial else args.model_dir
        tag = None if partial else "fullmodel.pt"
        user_content = smp.resume_from_checkpoint(path, tag=tag, partial=partial)
        total_steps = user_content["total_steps"] if partial else 0
        start_train_path_index = user_content.get("start_train_path_index", 0)
        start_batch_index = user_content.get("start_batch_index", 0)
        if "lr_scheduler" in user_content:
            lr_scheduler.load_state_dict(user_content["lr_scheduler"])
    else:
        total_steps = 0
        start_train_path_index = 0
        start_batch_index = 0

    start = time.time()
    total_steps, throughput, loss = train(
        model,
        optimizer,
        lr_scheduler,
        model_config,
        start_train_path_index,
        start_batch_index,
        num_params,
        total_steps,
        args,
    )
    time_to_train = time.time() - start

    if args.save_final_full_model:
        # saves full model at the end
        user_content = {
            "cli_args": args.__dict__,
            "num_params": num_params,
            "total_steps": total_steps,
            "model_config": model_config,
        }
        if args.sharded_data_parallel_degree > 1:
            # When sharded_data_parallel_degree > 1, saving full model is not supported, saving partial instead
            # To get the full model, one can use the following API
            # > from sharded_data_parallel_checkpoint import get_full_state_dict_from_sharded_data_parallel_checkpoint
            # > full_model = get_full_state_dict_from_sharded_data_parallel_checkpoint(args.model_dir, tag=f"sharded_data_parallel_final_full_{num_params}", dtype=torch.float32)
            # > if args.use_distributed_transformer > 0: # translate the state_dict to hf format if distributed transformer is used
            # >     full_model = smp.nn.huggingface.gpt2.translate_state_dict_to_hf_gpt2(full_model, max_seq_len=args.max_context_width)
            # Note: the shared parameter will not be reflected so during loading you might need to load with strict=False
            user_content["buffer_names"] = get_buffer_names(model)
            user_content["param_shapes"] = get_param_shapes(model, optimizer)
            smp.save_checkpoint(args.model_dir,
                tag=f"sharded_data_parallel_final_full_{num_params}",
                partial=True,
                model=model,
                optimizer=optimizer,
                user_content=user_content)
        else:
            smp.save_checkpoint(args.model_dir, tag="fullmodel.pt", partial=False, model=model, user_content=user_content)

    smp.barrier()
    if smp.rank() == 0:
        print("SMP training finished successfully")


if __name__ == "__main__":
    main()

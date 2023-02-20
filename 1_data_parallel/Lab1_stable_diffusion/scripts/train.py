
import math
import os
import random


import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.utils.data.distributed



from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from torch.distributed.distributed_c10d import ReduceOp

from utils import parse_args,is_main_process,main_process_first,is_local_main_process,wait_for_everyone

import urllib.request
from PIL import Image


def freeze_params(params):
    for param in params:
        param.requires_grad = False


dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}


    
def main():
    args = parse_args()
    
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.global_rank = int(os.environ["RANK"])
    
    print(f"local rank:{args.local_rank} global rank:{args.global_rank} world size:{args.world_size}")

    torch.cuda.set_device(args.local_rank) 
    # initialize DDP with NCCL 
    dist.init_process_group(backend=args.backend) 
  
    # we will run the training with reduced precision to get better memory utilization.
    if args.datatype == "fp16":
        train_dtype = torch.float16
    elif args.datatype == "bf16":
        train_dtype = torch.bfloat16

    # If passed along, set the training seed now.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_auth_token=args.use_auth_token,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", use_auth_token=args.use_auth_token
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", use_auth_token=args.use_auth_token
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", use_auth_token=args.use_auth_token
    )

    # Freeze vae and text_encoder as we will be fine tuning only the unet model.
    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * args.world_size
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # TODO (patil-suraj): load scheduler using args
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt"
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if (args.dataset_name is not None) and ('.' not in args.dataset_name):
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            use_auth_token=True if args.use_auth_token else None,
        )
    elif (args.dataset_name is not None):
        dataset = load_dataset('parquet',data_files=args.dataset_name)
        
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        if args.validation_data_dir is not None:
            data_files["validation"] = os.path.join(args.validation_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_process#imagefolder.

    # If we don't have a validation split, split off a percentage of train as validation.
    args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        input_ids = tokenizer(captions, max_length=tokenizer.model_max_length, padding=True, truncation=True).input_ids
        return input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    def preprocess_val(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [val_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples, is_train=False)
        return examples

    with main_process_first(args.global_rank):
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
        if args.max_eval_samples is not None:
            dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(range(args.max_eval_samples))
        # Set the validation transforms
        eval_dataset = dataset["validation"].with_transform(preprocess_val)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
        }

            
    sampler = torch.utils.data.DistributedSampler(
                train_dataset,
                shuffle=True,
                seed=args.seed,
                rank=args.global_rank,
                num_replicas=args.world_size,
                drop_last=True,
            )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=0
    )
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.eval_batch_size, num_workers=0)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps))
    if args.max_train_steps <= 0:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    device = torch.device("cuda")
    unet.to(device)
    # Wrap the model with DDP
    unet = DDP(unet,device_ids=[args.local_rank])

    
    # Move vae and textencoder to device (gpu:rank)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda")
    vae.to(device,train_dtype)
    text_encoder.to(device,train_dtype)

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    text_encoder.eval()
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * args.world_size * args.gradient_accumulation_steps
    if is_main_process(args.global_rank):
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_local_main_process(args.local_rank))
    progress_bar.set_description("Steps")
    global_step = 0

    try:
        if is_main_process(args.global_rank):
            print("using local safety checker")
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(args.pretrained_model_name_or_path,subfolder='./safety_checker')
        feature_extractor=CLIPFeatureExtractor.from_pretrained(os.path.join(args.pretrained_model_name_or_path,'feature_extractor/preprocessor_config.json'))
    except Exception:
        if is_main_process(args.global_rank):
            print("using hf download for safety checkers")
            print(Exception)
        safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
    
    wait_for_everyone()
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with torch.autocast(device_type='cuda', dtype=train_dtype):

                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]
                            
                loss = F.mse_loss(noise_pred, noise, reduction="none")
                loss.backward(loss)
                loss = loss.mean([1, 2, 3]).mean()
                
                # do all reduce of loss across ranks
                dist.all_reduce(loss, ReduceOp.SUM)
                loss = loss / args.world_size

                optimizer.zero_grad()
                optimizer.step()
                lr_scheduler.step()
                
                progress_bar.update(1)
                global_step += 1 
                print(f"step {global_step} loss {loss}")

                if global_step >= args.max_train_steps:
                    break

        wait_for_everyone()

    # Create the pipeline using the trained modules and save it.

    if is_main_process(args.global_rank):
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet.module if args.world_size >1 else unet,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            ),
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()
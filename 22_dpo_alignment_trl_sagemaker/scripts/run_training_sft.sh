#!/bin/bash

huggingface-cli login --token $HF_token
aws s3 cp $data_location /opt/ml/input/data/dataset/

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${NUM_GPUS} GPUs on the machine"

accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes ${NUM_GPUS} run_sft.py --config receipes/sft-llama-3-2-3b-qlora.yaml
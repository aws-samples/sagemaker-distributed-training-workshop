#!/bin/bash

huggingface-cli login --token $HF_token
aws s3 cp $data_location /opt/ml/input/data/dataset/


accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes 8 run_sft.py --config receipes/sft-llama-3-2-3b-qlora.yaml
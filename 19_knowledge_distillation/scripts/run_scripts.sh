#!/bin/bash

huggingface-cli login --token $HF_token
aws s3 cp s3://sagemaker-us-east-1-783764584149/datasets/deepseek-r1-8b-1b-Medial/train/dataset.json /opt/ml/input/data/dataset/

python3 hf_download.py

torchrun --nproc_per_node 8 knowledge_distillation_distributed.py --config knowledge_distillation_distributed.yaml
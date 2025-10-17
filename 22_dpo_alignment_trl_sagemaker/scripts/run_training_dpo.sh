#!/bin/bash

huggingface-cli login --token $HF_token

echo "${model_location}"

aws s3 cp $model_location /opt/ml/input/model/

tar -xzvf /opt/ml/input/model/model.tar.gz -C /opt/ml/input/model/

aws s3 cp $data_location /opt/ml/input/data/dataset/

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${NUM_GPUS} GPUs on the machine"

# echo "Merging Lora Weights"
# python merge_adapter_weights.py --peft_model_id /opt/ml/input/model/Qwen3-0.6B-function-calling/ --output_dir /opt/ml/input/model/merged-weights --save_tokenizer True

echo "Start Training"
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes ${NUM_GPUS} run_dpo.py --config receipes/sft-dpo-qwen3-1.7b.yaml
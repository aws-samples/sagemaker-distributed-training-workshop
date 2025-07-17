#!/bin/bash

huggingface-cli login --token $HF_token

aws s3 cp $model_location /opt/ml/input/model/

tar -xzvf /opt/ml/input/model/model.tar.gz -C /opt/ml/input/model/

aws s3 cp $data_location /opt/ml/input/data/dataset/

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${NUM_GPUS} GPUs on the machine"


python merge_adapter_weights.py --peft_model_id /opt/ml/input/model/llama-3-2-3b-function-calling/ --output_dir /opt/ml/input/model/merged-weights --save_tokenizer True

accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes ${NUM_GPUS} run_dpo.py --config receipes/sft-dpo-llama-3-2-3b.yaml
#!/bin/bash

# Set the model weights path
FSX_PATH=$SM_CHANNEL_MODELWEIGHTS

echo "***** Current Dir "
pwd

git clone https://github.com/aws/sagemaker-hyperpod-training-adapter-for-nemo.git

cd sagemaker-hyperpod-training-adapter-for-nemo/

pip install .[all]

cd src/hyperpod_nemo_adapter/scripts/

echo "***** Listing after downloading /merge_peft_checkpoint.py "

ls -ltr merge_peft_checkpoint.py

echo "***** Merging the base model with the trained adapter "

mkdir "final_model"

python3 merge_peft_checkpoint.py --hf_model_name_or_path $FSX_PATH \
--peft_adapter_checkpoint_path $FSX_PATH"/output/checkpoints/peft_sharded/step_50/" \
 --output_model_path $FSX_PATH"/final_model/" --deepseek_v3 true
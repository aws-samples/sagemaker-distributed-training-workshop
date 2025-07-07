#!/bin/bash

# Set the FSX path from SageMaker environment variable
FSX_PATH=$SM_CHANNEL_MODELWEIGHTS
CONVERTED_DIR=$SM_HP_CONVERTED_FSX_DIR

# Print the FSx data channel path
echo "***** The FSx data channel path for the model is located at: $FSX_PATH"

pwd

sm_adapter_placeholder="sm_adapter_download"
deepseek_inference_placeholder="deepseek_inference"

mkdir $FSX_PATH/$sm_adapter_placeholder
mkdir $FSX_PATH/$deepseek_inference_placeholder

cd $FSX_PATH/$sm_adapter_placeholder
git clone https://github.com/aws/sagemaker-hyperpod-training-adapter-for-nemo.git

cd $FSX_PATH/$deepseek_inference_placeholder
# Clone the DeepSeek-V3 repository
git clone https://github.com/deepseek-ai/DeepSeek-V3.git

# Change to the inference directory and install requirements
cd DeepSeek-V3
find . -type d -exec touch {}/__init__.py \;
cd inference
pip install -r requirements.txt

cp -r $FSX_PATH/$sm_adapter_placeholder/sagemaker-hyperpod-training-adapter-for-nemo/src/hyperpod_nemo_adapter/scripts/* .

echo "***** Creating dir $FSX_PATH/$CONVERTED_DIR"
mkdir $FSX_PATH/$CONVERTED_DIR
ls -ltr $FSX_PATH/$CONVERTED_DIR

echo "***** Starting conversion from FP8 to Bf16 from $FSX_PATH to $FSX_PATH/$CONVERTED_DIR"

echo "Current Dir: $(pwd)"

current_inf_path=$(pwd)

export PYTHONPATH=${PYTHONPATH}:${current_inf_path}
export PYTHONPATH=${PYTHONPATH}:${current_inf_path}/casting_utils
echo "PYTHONPATH:"$PYTHONPATH

cd $FSX_PATH/$deepseek_inference_placeholder
# Run the conversion script
python -m DeepSeek-V3.inference.fp8_cast_bf16 --input-fp8-hf-path "$FSX_PATH" --output-bf16-hf-path "$FSX_PATH/$CONVERTED_DIR"

echo "Current working directory: $pwd"

# Copy specific files to $FSX_PATH/$CONVERTED_DIR/ directory
echo "***** Copying files to $FSX_PATH/$CONVERTED_DIR/ directory"
find $FSX_PATH -type f \
    ! -name "*.safetensors" \
    ! -name "*metadata*" \
    ! -name "*lock*" \
    -not -path "*/$sm_adapter_placeholder/*" \
    -not -path "*/$deepseek_inference_placeholder/*" \
    -not -path "*/$FSX_PATH/$CONVERTED_DIR//*" \
    -exec cp -v {} "$FSX_PATH/$CONVERTED_DIR/" \;

echo "***** Script execution completed"



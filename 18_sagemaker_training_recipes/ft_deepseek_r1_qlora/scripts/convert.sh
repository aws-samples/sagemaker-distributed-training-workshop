#!/bin/bash

# Set the FSX path from SageMaker environment variable
FSX_PATH=$SM_CHANNEL_MODELWEIGHTS
CONVERTED_DIR=$SM_HP_CONVERTED_FSX_DIR

# Print the FSx data channel path
echo "***** The FSx data channel path for the model is located at: $FSX_PATH"

pwd

# Clone the DeepSeek-V3 repository
git clone https://github.com/deepseek-ai/DeepSeek-V3.git

# Change to the inference directory and install requirements
cd DeepSeek-V3/inference
pip install -r requirements.txt

echo "***** Downloading the file now"

# Check if fp8_cast_bf16.py exists
if [ -f fp8_cast_bf16.py ]; then
    echo "***** Removing existing fp8_cast_bf16.py"
    rm fp8_cast_bf16.py
else
    echo "***** fp8_cast_bf16.py does not exist"
fi

# Download the new fp8_cast_bf16.py file
echo "***** Downloading new fp8_cast_bf16.py"
curl -O https://raw.githubusercontent.com/aws/sagemaker-hyperpod-training-adapter-for-nemo/main/src/hyperpod_nemo_adapter/scripts/fp8_cast_bf16.py

# Verify the download
if [ -f fp8_cast_bf16.py ]; then
    echo "***** fp8_cast_bf16.py successfully downloaded"
    ls -ltr fp8_cast_bf16.py
else
    echo "***** Error: fp8_cast_bf16.py download failed"
    exit 1
fi

echo "***** Starting conversion from FP8 to Bf16"

# Run the conversion script
python fp8_cast_bf16.py --input-fp8-hf-path "$FSX_PATH" --output-bf16-hf-path "$FSX_PATH/$CONVERTED_DIR"

echo "Current working directory: $pwd"

# Copy specific files to $FSX_PATH/$CONVERTED_DIR/ directory
echo "***** Copying files to $FSX_PATH/$CONVERTED_DIR/ directory"
find $FSX_PATH -type f \
    ! -name "*.safetensors" \
    ! -name "*metadata*" \
    ! -name "*lock*" \
    -exec cp -v {} "$FSX_PATH/$CONVERTED_DIR/" \;

echo "***** Script execution completed"



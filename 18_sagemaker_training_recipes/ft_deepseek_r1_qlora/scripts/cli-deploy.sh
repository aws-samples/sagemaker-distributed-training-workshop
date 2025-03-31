FSX_PATH=$SM_CHANNEL_MODELWEIGHTS

echo $base

# Use the variable as needed
echo "***** The FSx data channel path for the model is located at: $FSX_PATH"

#ls -ltr $FSX_PATH

git clone https://github.com/deepseek-ai/DeepSeek-V3.git

cd DeepSeek-V3/inference
pip install -r requirements.txt

echo "***** Starting conversion to DeepSeek"

python convert.py --hf-ckpt-path $FSX_PATH"/final_model" \
--save-path $FSX_PATH"/final_deepseek_model" --n-experts 256 \
 --model-parallel 32

echo "***** Completed Conversion"


 


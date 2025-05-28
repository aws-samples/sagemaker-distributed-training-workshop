from pathlib import Path
import os

# set HF_HUB_ENABLE_HF_TRANSFER env var to enable hf-transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

HF_TEACHER_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# create model dir
teacher_model_tar_dir = Path(HF_TEACHER_MODEL_ID.split("/")[-1])
teacher_model_tar_dir.mkdir(exist_ok=True)

# Download model from Hugging Face into model_dir
snapshot_download(
    HF_TEACHER_MODEL_ID,
    local_dir='/opt/ml/model/DeepSeek-R1-Distill-Llama-8B', # download to model dir
    revision="main", # use a specific revision, e.g. refs/pr/21
    #local_dir_use_symlinks=False, # use no symlinks to save disk space
    ignore_patterns=["*.msgpack*", "*.h5*", "*.bin*"], # to load safetensor weights
)

# check if safetensor weights are downloaded and available
#assert len(list(teacher_model_tar_dir.glob("*.safetensors"))) > 0, "Model download failed"

HF_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# create model dir
model_tar_dir = Path(HF_MODEL_ID.split("/")[-1])
model_tar_dir.mkdir(exist_ok=True)

# Download model from Hugging Face into model_dir
snapshot_download(
    HF_MODEL_ID,
    local_dir='/opt/ml/model/Llama-3.2-1B-Instruct', # download to model dir
    revision="main", # use a specific revision, e.g. refs/pr/21
    #local_dir_use_symlinks=False, # use no symlinks to save disk space
    ignore_patterns=["*.msgpack*", "*.h5*", "*.bin*"], # to load safetensor weights
)

# check if safetensor weights are downloaded and available
#assert len(list(model_tar_dir.glob("*.safetensors"))) > 0, "Model download failed"
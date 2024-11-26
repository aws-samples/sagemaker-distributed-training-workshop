import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from typing import List

SRC_DIR = "/fsx/datasets/c4/en/hf/"
OUT_DIR = "/fsx/datasets/c4/en/nmt-tokenized-2/llama"

if not Path(OUT_DIR).exists():
    os.makedirs(OUT_DIR)


# def process_file(idx):
#     file_idx_str = str(idx).zfill(5)
#     file_stem = f"c4-train.{file_idx_str}-of-01024"
#     file_name = f"{file_stem}.json.gz"
#     cmd = f"python data/_prepare_nemo_megatron_dataset.py \
#                 --input {os.path.join(SRC_DIR, file_name)} \
#                 --output-prefix {OUT_DIR}/{file_stem} \
#                 --tokenizer-library=huggingface \
#                 --tokenizer-type hf-internal-testing/llama-tokenizer \
#                 --dataset-impl mmap \
#                 --append-eod \
#                 --workers 32"
#     os.system(cmd)
#     output_partition_files = list(Path(OUT_DIR).glob(f"{file_stem}_[0-9]*"))
#     # Running with 2 partitions creates some extra files we don't need
#     for a_file in output_partition_files:
#         a_file.unlink()
#     input_partition_files = list(Path(SRC_DIR).glob(f"{file_stem}.json_[0-9].gz"))
#     for a_file in input_partition_files:
#         a_file.unlink()



def process_file(idx: int) -> None:
    """
    Process a file with the given index using secure subprocess execution.
    
    Args:
        idx: The index of the file to process
    """
    file_idx_str = str(idx).zfill(5)
    file_stem = f"c4-train.{file_idx_str}-of-01024"
    file_name = f"{file_stem}.json.gz"
    
    # Construct argument list instead of shell command string
    cmd_args = [
        "python",
        "data/_prepare_nemo_megatron_dataset.py",
        "--input", os.path.join(SRC_DIR, file_name),
        "--output-prefix", f"{OUT_DIR}/{file_stem}",
        "--tokenizer-library", "huggingface",
        "--tokenizer-type", "hf-internal-testing/llama-tokenizer",
        "--dataset-impl", "mmap",
        "--append-eod",
        "--workers", "32"
    ]
    
    try:
        # Run process without shell=True and with proper argument list
        result = subprocess.run(
            cmd_args,
            check=True,  # Raises CalledProcessError if return code != 0
            capture_output=True,  # Capture stdout/stderr
            text=True  # Return strings instead of bytes
        )
        
        # Handle output partition files
        output_partition_files = list(Path(OUT_DIR).glob(f"{file_stem}_[0-9]*"))
        for a_file in output_partition_files:
            try:
                a_file.unlink()
            except OSError as e:
                print(f"Error deleting output file {a_file}: {e}")
                
        # Handle input partition files
        input_partition_files = list(Path(SRC_DIR).glob(f"{file_stem}.json_[0-9].gz"))
        for a_file in input_partition_files:
            try:
                a_file.unlink()
            except OSError as e:
                print(f"Error deleting input file {a_file}: {e}")
                
    except subprocess.CalledProcessError as e:
        print(f"Process failed with return code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

pool = ThreadPoolExecutor(max_workers=32)

# import os
# node_id = int(os.getenv('SLURM_NODEID'))
# num_nodes = int(os.getenv('SLURM_NNODES'))
threads = [pool.submit(process_file, idx) for idx in range(95, 256)]

"""
Model Download Script using Hugging Face Hub.
This script downloads models from Hugging Face Hub with customizable parameters and environment settings.
"""

import argparse
import os
import multiprocessing
from typing import Dict
from huggingface_hub import snapshot_download
from utility import run_command

class ModelDownloader:
    """Handles model downloading operations with custom environment settings."""
    
    def __init__(self, args):
        """
        Initialize ModelDownloader with command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.setup_environment()
        
    def setup_environment(self) -> None:
        """Set up custom environment variables for the download process."""
        custom_env: Dict[str, str] = {
            "HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
            "HF_TOKEN": self.args.hf_token,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "1000"
        }
        self.set_custom_env(custom_env)
        
    @staticmethod
    def set_custom_env(env_vars: Dict[str, str]) -> None:
        """
        Set custom environment variables.

        Args:
            env_vars: Dictionary of environment variables to set

        Raises:
            TypeError: If env_vars is not a dictionary
            ValueError: If any key or value in env_vars is not a string
        """
        if not isinstance(env_vars, dict):
            raise TypeError("env_vars must be a dictionary")

        for key, value in env_vars.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("All keys and values in env_vars must be strings")

        os.environ.update(env_vars)
        print("Updated environment variables:")
        for key, value in env_vars.items():
            print(f"  {key}: {value}")
            
    def download_model(self) -> None:
        """Download the model using Hugging Face's snapshot_download."""
        local_dir = f"{self.args.modelweights}/{self.args.local_fsx_dir}"
        print(f"**** fsx dir: {local_dir}")

        # Change to modelweights directory
        create_model_dir = f'cd {self.args.modelweights}'
        run_command(create_model_dir)

        print(f"**** model_id: {self.args.model_id}")
        
        num_processes = multiprocessing.cpu_count()
        print(f"**** num_processes: {num_processes}")
        print(f"Starting download using {num_processes} processes...")

        try:
            snapshot_download(
                repo_id=self.args.model_id,
                local_dir=local_dir,
                resume_download=True,
                max_workers=num_processes
            )
            print("Download completed successfully!")
        except Exception as e:
            print(f"Error downloading: {e}")
            raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download models from Hugging Face Hub with custom parameters"
    )
    
    parser.add_argument(
        "--modelweights",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODELWEIGHTS"),
        help="Path to model weights directory"
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Hugging Face model ID"
    )
    
    parser.add_argument(
        "--local_fsx_dir",
        type=str,
        default="deepseek_r1_671b_tj",
        help="Local FSx directory name"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Hugging Face API token"
    )

    return parser.parse_args()

def main():
    """Main execution function."""
    print("Starting model download process...")
    args = parse_arguments()
    
    downloader = ModelDownloader(args)
    downloader.download_model()

if __name__ == "__main__":
    main()

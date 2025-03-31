import subprocess as sb
from typing import Dict, Optional, Tuple
import time

def run_command(command: str) -> None:
    """
    Run a shell command and handle potential errors.

    Args:
        command (str): The command to run.

    Raises:
        subprocess.CalledProcessError: If the command fails.
        ValueError: If the command string is empty.
        
    """

    print(f'\n\n ***** Executing command: {command} \n\n')

    try:
        # Start the timer
        start_time = time.time()
        
        result = sb.run(
            command,
            shell=True,
            capture_output=False,
            text=True,
            check=True
        )
        # End the timer
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        print(f"\n\n ***** Execution time for command: {command} is : {elapsed_time:.4f} seconds \n\n")

    except sb.CalledProcessError as e:
        report_error=1
        print(f"**** Command failed with error code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        raise
    except Exception as e:
        report_error=1
        print(f"****An unexpected error occurred: {e}")
        raise

import subprocess
import sys
import os

def install_packages_silently(packages: list):
    """
    Installs multiple Python packages silently (no output).
    
    Args:
        packages (list): A list of package names to install.
    """
    with open(os.devnull, 'w') as devnull:
        # Redirect stdout and stderr to devnull to suppress output
        subprocess.run([sys.executable, "-m", "pip", "install"] + packages, stdout=devnull, stderr=devnull)

import os
import subprocess
import sys
from pathlib import Path


def install_matlab_engine():
    # Path to the MATLAB installation directory
    matlab_path = Path(r'F:\Program Files\MATLAB\R2024a')

    # Path to the Python engine directory within MATLAB
    python_path = matlab_path / 'extern' / 'engines' / 'python'

    # Change the current working directory to the Python engine directory
    os.chdir(python_path)

    # Construct the command to install the MATLAB engine
    # It's better to use sys.executable to ensure the correct Python interpreter is used
    command = [sys.executable, 'setup.py', 'install']

    try:
        # Run the setup.py script to install the MATLAB engine
        print(f"Installing MATLAB Engine from {python_path}...")
        subprocess.check_call(command)
        print("MATLAB Engine installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing MATLAB Engine: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == '__main__':
    install_matlab_engine()

# Project for course DD2434: Advanced Machine Learning at KTH Royale Institute of Technology

## Description
This project implements GraphSage as described in the following papers:
- "A Comparative Study for Unsupervised Network Representation Learning" by Khosla, Setty, and Anand.
- "Inductive Representation Learning on Large Graphs" by Hamilton, Ying, and Leskovec.

## Report
todo: add link 

## Installing steps for requirements on a MACOS 14.1 - 14.2 
These steps are based on the solution provided in [this issue](https://github.com/rusty1s/pytorch_scatter/issues/241).

### Creating a Conda Environment
1. Create a new environment: `conda create -n graph_sage_env python=3.9`
2. Activate the environment: `conda activate graph_sage_env`

### Installing Packages
3. `conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64`
4. `python -m pip --no-cache-dir install torch torchvision torchaudio`
5. Verify Torch installation: `python -c "import torch; print(torch.__version__)"`
6. Install `torch-scatter`: `python -m pip --no-cache-dir install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html`
7. Install `torch-sparse`: `python -m pip --no-cache-dir install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html`
8. Install `torch-geometric`: `python -m pip --no-cache-dir install torch-geometric`
9. Install other requirements: `pip install -r requirements.txt`

### Additional Configuration
To ensure compatibility with macOS's GPU limitations, set the environment variable to fall back to CPU when GPU methods are not implemented. Also, ensure the `.env` file is added to your project root.
- Execute in the command line: `export PYTORCH_ENABLE_MPS_FALLBACK=1`

Note: These instructions are tailored for macOS 14.1 - 14.2. Adjustments might be needed for other versions or operating systems.


## Structure of the files



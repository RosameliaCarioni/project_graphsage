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


## File Structure Overview

This project's file structure is organized to facilitate understanding and interaction with the various components involved in the machine learning process. Here's the breakdown:

### Python Scripts
The core methods used throughout the project are encapsulated within `.py` files, each serving a specific purpose:

- `read_data.py`: Handles the information retrieval of files from the [Arizona State University data repository](http://datasets.syr.edu/pages/datasets.html) in order to create graphs. 

- `graph_information.py`: This script is a utility for graph analytics. It visualizes general information about a graph and it's loader (used to sample its neighbors), providing insights into the structure and composition of your networks.

- `test_embeddings.py`: Central to evaluating the performance of the models, this file contains functions for node classification and edge prediction, allowing for the assessment of the embeddings generated.

- `graphsage_calculate_embeddings.py`: This contains the model used to learn and derive the embedding matrix from the datasets. It offers flexibility by allowing the use of a local model (as applied in this study) or the inbuilt GraphSage from `torch_geometric.nn`.

### Jupyter Notebooks
For a more interactive and exploratory approach, `.ipynb` notebooks are used, particularly for experimenting with the datasets:

- Dataset Notebooks: Each of the datasets employed in this study has an associated notebook. These notebooks are where the data manipulation, experimentation, and initial analysis occur. 


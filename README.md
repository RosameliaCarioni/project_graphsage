# Project for course DD2434: Advanced Machine Learning at KTH Royale Institute of Technology
# Implementation of GraphSage

## Description
Implementation of GraphSage following the papers:
"A Comparative study for Unsupreevised Network Representation Learning" by Khosla, Setty and Anand
"Inductive Representation Learning on Large Graphs" by Hamilton, Ying and Leskovec 

## Report
todo: add link 

## Installing steps for requirements on a MACOS 14.1 - 14.2 
Following solution from: https://github.com/rusty1s/pytorch_scatter/issues/241

### Create environment 
1. conda create -n graph_sage_6 python=3.9  
2. conda activate graph_sage_6

### Install packages 
3. conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
4. python -m pip --no-cache-dir   install torch torchvision torchaudio
5. python -c "import torch; print(torch.__version__)"
6. python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html
7. python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html
8. python -m pip --no-cache-dir  install  torch-geometric
9. pip install -r requirements.txt

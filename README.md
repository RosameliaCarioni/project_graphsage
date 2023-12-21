# GraphSAGE implementation 

## Description


## Report


## Installing steps for a MACOS 14.2 
Following solution from: https://github.com/rusty1s/pytorch_scatter/issues/241

conda create -n graph_sage_6 python=3.9  
conda activate graph_sage_6
conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
python -m pip --no-cache-dir   install torch torchvision torchaudio
python -c "import torch; print(torch.__version__)"
python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html
python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html
python -m pip --no-cache-dir  install  torch-geometric
pip install -r requirements.txt

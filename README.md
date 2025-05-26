# CSGAV
Official PyTorch implementation of "Multispectral-Hyperspectral Image Fusion via Similarity-Guided Graph Attention and VAE-Transformer" (IEEE Transactions on Geoscience and Remote Sensing, 2025).

## Overview
CSGAV is a novel cross-modal fusion network for multispectral (MSI) and hyperspectral (HSI) image fusion. The model integrates similarity-guided graph attention (SGA) and a variational autoencoder-Transformer (VAET) to generate high-spatial-resolution hyperspectral images (HR-HSI).

### Key Features
- Similarity-Guided Graph Attention (SGA) : Computes similarity degrees between source images and constructs a graph attention module to mitigate modal differences
- Variational Autoencoder-Transformer (VAET) : Enhances generalization and robustness through adaptive weighted Transformer guided by a variational autoencoder
- Cross-modal Interactive Architecture : Integrates SGA and VAET to achieve high-quality fusion results
## Model Architecture
The CSGAV architecture consists of three main components:

1. SGA Module : Upsamples the low-resolution HSI with guidance from the high-resolution MSI using similarity measures and graph attention networks
2. VAET Module : Processes features through a variational autoencoder and adaptive weighted Transformer
3. Cross-modal Fusion : Integrates the outputs from both modules to generate the final HR-HSI

Requirements：
torch
torchvision
numpy
dgl
tensorboardX
tqdm
pandas
thop

## Dataset Preparation
The code supports three datasets: Houston, Botswana, and Pavia. Organize your data as follows:
dataset/
├── Houston/
│   ├── Train/
│   └── Test/
├── Botswana/
│   ├── Train/
│   └── Test/
└── Pavia/
    ├── Train/
    └── Test/

## Training
To train the CSGAV model on the Houston dataset:
python train.py --database Houston --num_epochs 150 --batch_size 4 --lr 0.0001

You can modify the following parameters:

- --database : Dataset name (Houston, Botswana, or Pavia)
- --num_epochs : Number of training epochs
- --batch_size : Batch size for training
- --lr : Learning rate
- --crop_size : Size of training patches

## Testing
To test the trained model:
python test.py --model CSGAV --dataset Houston
The fusion results will be saved in the fused directory.

## Citation
If you find this code useful for your research, please cite our paper:
@article{CSGAV2025,
  title={Multispectral-Hyperspectral Image Fusion via Similarity-Guided Graph Attention and VAE-Transformer},
  author={},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={},
  pages={1-1},
  year={2025},
  publisher={IEEE},
  doi={10.1109/TGRS.2025.3573047}
}
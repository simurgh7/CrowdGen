# CrowdGen
### Official Implementation of "Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting"

**CVPR 2026**

This repository implements generative adversarial perturbations for crowd counting networks. Representative methods for both density map (SASNet) and point regression (P2PNet) paradigms are provided.

## 📁 General Project Structure

├── Crowd Model/ # Density Map or Point Regression approach
│ ├── datasets/
│ │ └── dataset.py # Dataloader
| | └── make_npydata.py # Data Organizer
│ └── models/
| | └── model.py # Files for Crowd Localized Counting Model
| | └── unet.py # Perturbation Generator Network
│ └── util/ # Optional for specific methods. 
│ ├── attack_unet.py # Train perturbation generator
| ├── camgen.py # GradCAM extraction process for backbone
│ ├── config.py # Configuration with flag support
| ├── engine.py # evaluation functions
| ├── explain.py # cam ex
│ └── inference.py # Side-by-side comparison (clean vs. adversarial)

......
└── data/ # Data directory (create this for SHHA, UCF, JHU, NWPU)
  ├── image_data/ # Images (e.g., 1.jpg)
  └── gt_data/ # Ground truth (e.g., 1.h5)

## 🚀 Quick Start

### Data Preparation
Place images and ground truth in `data/image_data/` and `data/gt_data/` with matching names (e.g., `1.jpg` and `1.h5`). Run `datasets/make_npydata.py` to generate file lists that can be placed in `datasets/npydata/` folder. Run the same after adversarial image generation, gt is not needed this time, the list of images should be enough. 

### Training Perturbation Generator
# Density Map approach
cd 2021_AAAI_SASNet_DM
python attack_unet.py --epochs 50 --batch_size 8

# Point Regression approach
cd 2021_ICCV_P2PNet_PR
python attack_unet.py  --epochs 50 --batch_size 8

### Evaluation Side by Side
python models/inference.py --model_path ./weights/model.pth --adv_npy_path ./datasets/npydata/shha_adv.npy --clean_npy_path ./datasets/npydata/shha_clean.npy
config.py contains all configurable parameters. Override any attribute using command-line flags

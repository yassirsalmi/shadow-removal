# Shadow Removal via Generative Priors

## Overview
This project implements an advanced shadow removal technique using generative AI and deep learning. Inspired by the research in [Shadow Removal via Generative Priors](https://github.com/YingqingHe/Shadow-Removal-via-Generative-Priors), the model leverages StyleGAN and advanced image processing techniques to remove shadows from images.

## Key Features
- Advanced shadow removal using generative AI
- StyleGAN-based image generation
- Supports custom image processing
- Tensorflow implementation

## Requirements
- Python 3.10
- TensorFlow 2.15
- TensorFlow Hub
- NumPy
- Pillow
- tqdm

## Installation
1. Clone the repository
2. Create a virtual environment:
```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

## Download Checkpoints
Download the pretrained checkpoints from the [original project's Google Drive](https://drive.google.com/drive/folders/1Rg5He8XIY8qP4JYPFRRGUIvfZUcqm8zt?usp=sharing):

1. Create checkpoint directory:
```bash
mkdir -p checkpoint
```

2. Download required checkpoints:
- `550000.pt` (StyleGAN checkpoint)
- `face-seg-BiSeNet-79999_iter.pth` (Face segmentation model)

3. Convert Checkpoints:
```bash
python convert_checkpoint.py 
```

## Usage
```bash
python remove_shadow.py --input_image path/to/image.jpg --output_dir results/
```

## Folder structure

```
shadow-removal/
│
├── checkpoint/           # Saved model checkpoints
├── imgs/                 # Input and sample images
├── lpips/                # Perceptual loss implementation
├── models/               # Neural network model definitions
├── op/                   # Custom TensorFlow operations
├── results/              # Output images after shadow removal
├── utils/                # Utility functions and helpers
│
├── convert_checkpoint.py # Script to convert checkpoints
├── model.py              # Main model implementation
├── projector.py          # Image projection utilities
├── remove_shadow.py      # Main script for shadow removal
├── requirements.txt      # Project dependencies
├── run.sh                # Shell script for running the project
└── run_windows.sh        # Windows-specific run script
```

## Authors
- [Yassir Salmi](https://github.com/yassirsalmi)
- [Abdeljalil Ounaceur](https://github.com/Abdeljalil-Ounaceur)

# BEAM: Background Elimination with Advanced Machine learning

BEAM is a Python package for training and using conditional diffusion models to model and remove systematic effects from image data. 

This project models the shape and intensity of scattered light in TESS (Transiting Exoplanet Survey Satellite) full frame image (FFIs) conditioned on orbital parameters.

## Overview

BEAM uses conditional denoising diffusion probabilistic models (cDDPMs) to model the shape and intensity of scattered light in TESS full frame image (FFIs) conditioned on orbital parameters.

## Installation

### From Source

```bash
git clone https://github.com/yourusername/BEAM.git
cd BEAM
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.9+
- See requirements.txt for full dependencies

## Project Structure

```
BEAM/
├── beam/                           # Main package
│   ├── models/                     # Model implementations
│   ├── data/                       # Dataset and data loading
│   ├── training/                   # Training functionality
│   └── utils/                      # Utilities
├── configs/                        # Configuration files
├── scripts/                        # Command-line scripts
└── tests/                          # Unit tests
```

## Usage

### Training a Model

```bash
python scripts/train.py --config configs/default_config.yaml
```

### Generating Images

```bash
python scripts/generate.py --model_path model_outputs/TESS_diffusion/model_epoch100.pth --n_samples 10
```

### Using the API

```python
import torch
from beam.models.unet import ContextUnet
from beam.models.diffusion import DDPM

# Create model
unet = ContextUnet(in_channels=1, in_dim=12, n_feat=256)
model = DDPM(nn_model=unet, betas=(1e-4, 0.02), n_T=600, device=device)

# Load trained weights
model.load_state_dict(torch.load("model_path.pth"))

# Generate samples
params = torch.rand((5, 1, 12))  # 5 sets of orbital parameters
samples, _ = model.sample_c(params, n_sample=1, size=(1, 64, 64), device=device)
```

## Data

The model is trained on TESS Full Frame Images and the corresponding orbital parameters of the Earth and the Moon reletive to the satellite. The data should be organized as follows:

<!-- - Orbital parameters: Stored in pickle files in the format specified by the `TESSDataset` class
- TESS images: Processed and stored in pickle files in the specified directory

For more information on the data format, see the documentation in `beam/data/datasets.py`. -->

## Model Architecture

BEAM uses a conditional UNet architecture with the following key components:

- **ContextUnet**: A U-Net with context embeddings for conditioning on orbital parameters
- **DDPM**: Denoising Diffusion Probabilistic Model implementation
- **Classifier-Free Guidance**: To control the influence of the conditioning signal

The model generates images through an iterative denoising process, starting from random noise and gradually refining it based on the conditioning information.

## Configuration

BEAM uses YAML configuration files to manage training and generation parameters. See `configs/default_config.yaml` for an example configuration with documentation.

Key configuration sections include:

- **model**: Model architecture and diffusion process parameters
- **data**: Dataset specifications and preprocessing settings
- **training**: Training hyperparameters and optimization settings
- **checkpoint**: Saving and loading behavior
- **distributed**: Multi-GPU training configuration

## Distributed Training

BEAM supports distributed training across multiple GPUs:

```bash
# Automatically use all available GPUs
python scripts/train.py --config configs/default_config.yaml

# Resume training from a checkpoint
python scripts/train.py --config configs/default_config.yaml --resume model_outputs/TESS_diffusion/model_epoch50.pth
```

## Citation

If you use BEAM in your research, please cite it as follows:

```
@software{beam2023,
  author = {Muthukrishna, D., Lupo, J., Vanderspek, R.},
  title = {BEAM: Background Elimination with Advanced Machine Learning},
  year = {2025},
  url = {https://github.com/daniel-muthukrishna/BEAM}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The TESS mission for providing the data
- The diffusion model implementation is adapted from research from Ho et al. (DDPM) 
    - DDPM: https://arxiv.org/abs/2006.11239
    - Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598
- The diffusion model code is adapted from: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
which was originally modified from: https://github.com/cloneofsimo/minDiffusion


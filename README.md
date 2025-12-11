# Face Editor (GAN Inversion & Editing)

A modular pipeline for semantic face editing using GAN Inversion (e4e) and Latent Space Manipulation. 
This project allows for modifying facial attributes (Age, Smile, Pose) of any input image while preserving identity.

## 🏗 Structure
- `src/`: Core logic (Model loading, Latent Math, Image Processing).
- `config/`: Configuration for paths and hyperparameters.
- `main.py`: CLI entry point.

## 🚀 Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
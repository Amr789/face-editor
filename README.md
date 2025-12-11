[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v8Wd5JSk_UH8ht_96W7KlOKL6YsvBPL1)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License MIT](https://img.shields.io/badge/license-MIT-green)

# Face Editor (GAN Inversion & Editing)

A pipeline for semantic face editing using GAN Inversion (e4e) and Latent Space Manipulation. 
This project allows for modifying facial attributes (Age, Smile, Pose) of any input image while preserving identity.


## Overview

This project implements a modular **GAN Inversion and Latent Space Editing** pipeline. It leverages the **encoder4editing (e4e)** framework to invert real images into the StyleGAN2 latent space ($W+$), allowing for high-fidelity semantic editing without retraining the model.

The system is designed to perform disentangled edits—changing specific facial attributes (Age, Smile, Pose) while preserving the subject's identity. It features a production-ready file structure, automated asset management, and a robust inference engine.

## Key Features

* **Advanced Architecture:** Utilizes `pSp` / `e4e` encoders for accurate image inversion into $W+$ latent space.
* **Semantic Editing:** Performs linear vector arithmetic using pre-computed boundary vectors to modify attributes like **Age**, **Smile**, and **Head Pose**.
* **Automated Setup:** Includes a `setup.py` script that automatically handles heavy asset downloads (Model Weights, Dlib predictors) and external dependencies.
* **Modular Design:** Clean separation of configuration, data processing, model handling, and editing logic for maintainability and scalability.

## Project Structure

```text
face-editor/
├── config/
│   └── config.yaml       # Central configuration (URLs, file paths, hyperparameters)
├── src/
│   ├── __init__.py
│   ├── model_handler.py  # Wrapper for loading e4e/StyleGAN2 models
│   ├── image_processor.py# Handles face alignment (Dlib) and tensor transformations
│   ├── latent_editor.py  # Core logic for latent vector manipulation
│   └── utils.py          # Helper functions
├── .gitignore            # Git exclusion rules (prevents tracking large assets)
├── main.py               # CLI Entry point for running the pipeline
├── setup.py              # Script to download weights and clone sub-modules
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Installation

### Prerequisites
* Python 3.8+
* GPU recommended (CUDA)

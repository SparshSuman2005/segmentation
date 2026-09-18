# DeepCrack: Concrete Crack Segmentation using U-Net

## Overview

DeepCrack is a deep learning-based image segmentation project for detecting and segmenting cracks in concrete surfaces.

The project uses a **U-Net architecture** to perform pixel-level binary segmentation. Given a concrete surface image, the model predicts a binary mask identifying the pixels corresponding to cracks.

Unlike simple image classification, which only determines whether a crack is present, semantic segmentation provides the **location and shape of the crack at pixel level**.

---

## Objectives

The main objectives of this project are:

- Detect cracks in concrete surface images.
- Perform pixel-level crack segmentation.
- Generate binary crack masks.
- Evaluate segmentation performance using standard metrics.
- Save trained model checkpoints during training.
- Visualize the original image, ground-truth mask, and predicted mask.

---

## Model Architecture

The project uses **U-Net**, a convolutional neural network architecture designed for image segmentation.

U-Net consists of two major components:

### Encoder

The encoder extracts important visual features from the input image through convolution and downsampling operations.

### Decoder

The decoder gradually reconstructs the spatial resolution of the image and produces a pixel-level segmentation mask.

### Skip Connections

Skip connections transfer spatial information from the encoder to the decoder, helping the network preserve fine details such as thin cracks.

---

## Dataset

The dataset is organized into corresponding image and mask directories.

The project uses:

- Training images
- Training masks
- Testing images
- Testing masks

Each input image has a corresponding ground-truth segmentation mask.

The ground-truth mask represents:

- **Black pixels** → background
- **White pixels** → crack

---

## Project Structure

```text
DeepCrack/
│
├── dataset/
│
├── preprocessing/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── main.py
│
├── outputs/
│   ├── checkpoints/
│   │   ├── epoch_1.pth
│   │   ├── epoch_2.pth
│   │   ├── epoch_3.pth
│   │   ├── epoch_4.pth
│   │   ├── epoch_5.pth
│   │   ├── best_model.pth
│   │   ├── latest_model.pth
│   │   └── final_model.pth
│   │
│   └── ...
│
├── Figure_1.png
├── README.md
└── requirements.txt
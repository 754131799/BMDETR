# Transformer-Based Multi-Scale Feature Fusion for Real-Time CT Bone Metastasis Detection

BM-DETR is a deep learning-based framework for the automated detection of bone metastases in CT images. This repository provides an open-source Qt-based software application that integrates our trained BM-DETR model with a user-friendly interface for clinical and research use.

## 🧠 Overview

Bone metastasis detection is a challenging task due to the subtlety and variability of lesions. Our approach, BM-DETR, leverages a Transformer-based detection model trained specifically on annotated CT data to identify metastatic lesions with high accuracy and interpretability.

This repository includes:

- A pretrained BM-DETR model
- A Qt-based GUI application for automated lesion detection
- Demo CT images for quick testing
- Instructions for model training and deployment

## 🚀 Features

- 🖼️ Support for widely used medical image formats (e.g., `.png`, `.jpg`, `.bmp`, etc.)
- 🔍 Automatic lesion detection from CT images
- 🔥 Grad-CAM heatmaps for interpretability
- 🧩 Optimized for clinical workflows
- 🖥️ Runs on both CPU and NVIDIA GPU (recommended)

## 🛠️ Training Details

To ensure reproducibility and transparency, below are the core training parameters used to train BM-DETR:

- **Epochs**: 200  
- **Batch size**: 32  
- **Input size**: Images resized to 640 × 640 pixels  
- **Optimizer**: Adam  
- **Hardware**: Trained on an NVIDIA RTX 4090 GPU  
- **Frameworks**: PyTorch, OpenCV, Qt5

If you use this tool in your research, please cite our accompanying paper，《Transformer-Based Multi-Scale Feature Fusion for Real-Time CT Bone Metastasis Detection》

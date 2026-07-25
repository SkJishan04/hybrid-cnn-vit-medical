# Hybrid CNN + Vision Transformer for Dermoscopic Lesion Classification
 
**A comparative study of convolutional, transformer-based, and hybrid cross-attention architectures for skin lesion classification on HAM10000.**
 
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)]()
[![Status](https://img.shields.io/badge/status-in%20progress-yellow)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
 
---
## Abstract
 
Convolutional Neural Networks (CNNs) excel at capturing local texture and edge information but lack an inherent mechanism for modeling long-range spatial context. Vision Transformers (ViTs) address this via global self-attention, but typically require large-scale pretraining data to learn the inductive biases CNNs get for free. This project investigates a **hybrid CNN–ViT architecture with cross-attention fusion**, designed to combine the local feature sensitivity of CNNs with the global contextual reasoning of transformers, applied to the clinically-relevant task of dermoscopic skin lesion classification on the **HAM10000** dataset (7 diagnostic classes, 10,015 images).
 
This repository documents the full experimental pipeline — data preparation, baseline models, the hybrid architecture, and iterative refinement — in the style of a running lab notebook. Version 1 (V1) results below surfaced an important negative finding around class-imbalance handling, which directly motivated the corrected V2 pipeline.
 
---

## Table of Contents
 
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Experimental Setup](#experimental-setup)
- [Results — V1 (ResNet50 Baseline)](#results--v1-resnet50-baseline)
- [Findings & Discussion](#findings--discussion)
- [Roadmap](#roadmap)
- [Repository Structure](#repository-structure)
- [Setup & Reproduction](#setup--reproduction)
- [Acknowledgments](#acknowledgments)
---

## Motivation
 
Standard CNNs (ResNet, EfficientNet, etc.) remain the dominant architecture for dermoscopic image classification, but they process images through a fixed local receptive field that grows only gradually with depth. This makes it harder for a CNN to relate a lesion's border characteristics to its overall shape and surrounding skin context in a single representational step — information that's often clinically relevant (e.g. asymmetry, border irregularity, per the ABCDE rule used in dermatology).
 
Vision Transformers model global relationships from the first layer via self-attention, but are known to underperform CNNs on small-to-medium datasets without large-scale pretraining, since they lack CNNs' built-in translation-invariance and locality priors.
 
**This project's central hypothesis:** a hybrid architecture — CNN backbone for local feature extraction, feeding a lightweight transformer encoder with explicit cross-attention back to the CNN's feature maps — can outperform either architecture alone on a mid-sized medical imaging dataset, by combining local and global reasoning rather than relying on either in isolation.
 
---

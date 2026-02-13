# Siamese ResNet50 for Paired Rock–Paper–Scissors Image Classification

## Problem Overview

This project tackles a paired-image classification task inspired by the game rock–paper–scissors. Designed a Siamese ResNet50 in PyTorch to classify paired grayscale images for a rock–paper–scissors Kaggle challenge. Used shared encoders with feature concatenation and FC layers. Tuned warm restarts, dropout, balanced BCE, mixed precision, and F1-based thresholds. Achieved 0.9048 private LB accuracy, outperforming multiple baselines.

The dataset is not included in this repository due to size limitations.

Each training sample consists of:
- Two 24×24 grayscale images
- A binary label:
  - +1 if the first image beats the second
  - -1 otherwise

The goal is to learn a model that predicts the outcome (+1 / -1) for unseen image pairs.

## Solution Ranking: 4/91

## Approach

To capture relational patterns between two images, I implemented a Siamese architecture using a shared ResNet50 backbone. Each image is processed through the same encoder, and the resulting feature vectors are concatenated before being passed to a fully connected classifier.

### Key Design Choices

- Shared ResNet50 encoder (grayscale-modified)
- Feature concatenation for pair interaction modeling
- Differential learning rates for backbone and classifier
- Mixed precision training (AMP + GradScaler)
- CosineAnnealingWarmRestarts scheduler
- Balanced BCEWithLogitsLoss
- F1-based threshold tuning

## Results

- Public Leaderboard: 0.9050
- Private Leaderboard: 0.9048

The final model outperformed earlier CNN and ResNet18 baselines through systematic architectural improvements and hyperparameter tuning.

## Submission Format

Predictions were submitted as a CSV file with:

- `ID` column matching `test.pkl`
- Predicted binary label (+1 or -1)

## How to Train

```bash
python train.py

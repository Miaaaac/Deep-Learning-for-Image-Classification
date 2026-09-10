# Deep Learning for Image Classification

Corduneanu-Huci Mia & Haltrich Thanika

## About the project

This project compares traditional machine learning against a range of deep learning architectures for image classification, applied to two benchmark computer vision datasets: **CIFAR-10** and **Fashion MNIST**. The goal is to evaluate how model complexity, architectural choices, and regularization techniques affect both accuracy and training time

## Datasets

| Dataset | Description | Classes | Image Size |
|---|---|---|---|
| **CIFAR-10** | Color photographs of real-world objects (ship, dog, deer, bird, cat, etc.) | 10 | 32×32×3 |
| **Fashion MNIST** | Grayscale images of clothing items (t-shirt, trouser, sneaker, bag, etc.) | 10 | 28×28×1 |

## Models Compared

### Traditional ML
- **Random Forest** trained on hand-crafted features (mean/std RGB values, flattened grayscale pixel intensities) — used as a non-deep-learning baseline

### Deep Learning
- **CNNs** — 3 custom architectural variations (varying number/size of conv layers and filters), plus:
  - CNN with average pooling
  - CNN with batch normalization
  - CNN with extensive dropout
  - CNN with data augmentation
  - CNN with hyperparameter tuning
- **ResNet-18** (built from scratch in PyTorch)
- **ResNet-50** (Keras, trained from scratch — no pretrained weights)
- **Neural ODE (N-ODE)** — a continuous-depth model using an ODE solver in place of discrete residual blocks

## Results

### CIFAR-10 (Accuracy)

| Model | Accuracy |
|---|---|
| Random Forest | 0.343 |
| CNN 1 | 0.628 |
| CNN 2 | 0.663 |
| CNN 3 | 0.717 |
| CNN (Avg. Pooling) | 0.719 |
| CNN (Dropout) | 0.745 |
| CNN (Data Augmentation) | 0.692 |
| CNN (Batch Norm) | **0.819** |
| ResNet-50 (Keras) | 0.343 |
| ResNet-18 (PyTorch) | 0.754 |
| Neural ODE | 0.754 |

### Fashion MNIST (Accuracy)

| Model | Accuracy |
|---|---|
| Random Forest | 0.821 |
| CNN 1 | 0.884 |
| CNN 2 | 0.883 |
| CNN 3 | 0.907 |
| CNN (Avg. Pooling) | 0.881 |
| CNN (Batch Norm) | 0.897 |
| CNN (Dropout) | 0.897 |
| CNN (Data Augmentation) | 0.733 |
| ResNet-50 (Keras) | 0.870 |
| ResNet-18 (PyTorch) | 0.727 |
| Neural ODE | **0.916** |

Full per-class precision/recall/F1 breakdowns for the Random Forest baseline are included in the notebooks.

### Accuracy vs. Training Time (CIFAR-10)

The **CNN with batch normalization** achieved the best accuracy/time trade-off, reaching the highest accuracy (0.819) in roughly 130 seconds of training. **ResNet-50 and Neural ODE** were both far more expensive to train (~570–650s) without outperforming the lighter CNN variants on CIFAR-10, while **Random Forest** trained almost instantly but with much lower accuracy.

## Key Takeaways

- Simple architectural tweaks (batch normalization, dropout) often gave larger accuracy gains than switching to a much deeper or more exotic architecture.
- Random Forest with hand-crafted pixel/color features is a weak baseline for CIFAR-10 (photorealistic, more complex textures) but surprisingly competitive on Fashion MNIST (simpler, grayscale, more structured shapes).
- ResNet-50 trained from scratch (no pretrained weights) struggled on CIFAR-10, likely due to the small 32×32 input and lack of pretraining — deeper models need either more data, pretraining, or architectural adaptation to shine.
- Neural ODEs performed competitively and even best on Fashion MNIST, showing that continuous-depth models can match or exceed discrete CNN/ResNet architectures, at the cost of longer training time.
- There is a clear **accuracy vs. training-time trade-off** across model families — the "best" model depends on whether accuracy or compute budget is the priority.

## Repository Structure

```
.
├── Project2_Mia_Nika_CIFRA10.ipynb        # CIFAR-10: Random Forest, CNNs, ResNet-18/50, Neural ODE
├── Project2_Mia_Nika_fashionMNIST1.ipynb  # Fashion MNIST: same model suite
├── ADVML_MiaNika_Project2.pdf             # Project presentation slides
└── README.md
```

## Requirements

```
numpy
pandas
matplotlib
torch
torchvision
tensorflow
scikit-learn
Pillow
```

## How to Run

1. Install dependencies: `pip install -r requirements.txt` (or install the packages listed above).
2. CIFAR-10 and Fashion MNIST are loaded directly via `tensorflow.keras.datasets`, so no manual download is required.
3. Open and run `Project2_Mia_Nika_CIFRA10.ipynb` and `Project2_Mia_Nika_fashionMNIST1.ipynb` — each notebook is organized by model (Random Forest → CNN variations → ResNet-18 → ResNet-50 → Neural ODE).

**Note:** GPU acceleration (CUDA or Apple Metal/MPS) is recommended for training the CNN, ResNet, and Neural ODE models in reasonable time.

# GridBlur Data Augmentation (An Improved GridMask)

## Introduction

This project is an experimental implementation of a modified version of GridMask Augmentation by Chen et al.

## Motivation

GridMask is an excellent augmentation algorithm that introduces random dropouts in the dataset to promote generalization and robustness.

However, in the real world, the model never encounters "black square" dropouts, which often may lead to unexpected results.

Hence, this experiment explores a possible solution by applying a blurred grid of squares instead of black dropouts.


## GridMask Official Source

- Original GridMask Implementation: [dvlab-research/GridMask | GitHub](https://github.com/dvlab-research/GridMask)
- Original GridMask Paper: [GridMask | arXiv](https://arxiv.org/abs/2001.04086)


## Experimental Results (Ongoing)

### ImageNet

| Dataset                       | Architecture | Augmentation | Top 1 Accuracy | Top 5 Accuracy |
|-------------------------------|--------------|--------------|----------------|----------------|
| ImageNet Subset (50 classes) | ResNet50      | Baseline     | 77.72          | 94.40          |
| ImageNet Subset (50 classes) | ResNet50      | + GridMask   | 78.24          | 94.68          |
| ImageNet Subset (50 classes) | ResNet50      | + GridBlur   | Ongoing        | Ongoing        |

### COCO2017

TBD 


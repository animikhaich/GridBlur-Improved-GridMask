# GridBlur Data Augmentation

## Introduction

This project is an experimental implementation of a modified version of GridMask Augmentation by Chen et al.

## Motivation

GridMask is an excellent augmentation algorithm that introduces random dropouts in the dataset to promote generalization and robustness.

However, in the real world, the model never encounters "black square" dropouts, which often may lead to unexpected results.

Hence, this experiment explores a possible solution by applying a blurred grid of squares instead of black dropouts, which aim to maintain the statistics of dropouts while providing a more realistic nature of the corresponding dropout.

## Experimental Results (Ongoing)

### ImageNet

| Dataset                       | Architecture | Augmentation | Top 1 Accuracy | Top 5 Accuracy |
|-------------------------------|--------------|--------------|----------------|----------------|
| ImageNet Subset (50 classes)  | ResNet50     | Baseline     | 77.72          | 94.40          |
| ImageNet Subset (50 classes)  | ResNet50     | + GridMask   | 78.24          | 94.68          |
| ImageNet Subset (50 classes)  | ResNet50     | + GridBlur   | 78.40          | 94.84          |

### COCO2017

TBD 


## Acknowledgements

- Original GridMask Implementation: [dvlab-research/GridMask | GitHub](https://github.com/dvlab-research/GridMask)
- Original GridMask Paper: [GridMask Data Augmentation | arXiv](https://arxiv.org/abs/2001.04086)

# CPPNN: C++ Neural Network & Vulkan Compute Engine

A modular deep learning engine in C++17 featuring hardware-accelerated training using Vulkan Compute shaders.

## Overview
CPPNN implements deep learning primitives and backpropagation without high-level frameworks. Computation runs on CPU (OpenMP) or directly on GPU via Vulkan compute shaders.

## Features
- Full GPU training pipeline (forward, backward, optimizer on GPU)
- Conv2D, MaxPooling2D, Dense, Rescaling, Dropout
- Adam, SGDMomentum, SGD optimizers
- Verified on MNIST dataset (~95%+ accuracy)

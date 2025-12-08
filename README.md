# DCGAN for MNIST & CIFAR-10 — PyTorch Implementation

- This repository contains a complete and modular implementation of Deep Convolutional GAN (DCGAN) trained on:
    - MNIST (1-channel, 32×32)
    - CIFAR-10 (3-channel, 32×32)

- Features:
    - Custom Generator & Discriminator
    - Full DCGAN-style weights initialization
    - Label smoothing for stability
    - Learning-rate schedulers
    - Per-epoch metrics logging
    - Automatic checkpoint saving
    - Training stability analysis
    - Visualization:
        - Generated images every 5 epochs
        - Loss curves
        - delta-loss oscillation analysis
    - Inference scripts

## Training Stability Check

The project includes automatic detection of GAN instability:
    
    ```
    Mean |deltaD| < threshold → stable
    Mean |deltaG| < threshold → stable
    ```
Plots generated:
    - Loss curves
    - Oscillation curves
    - Image grids

## Architectures

- **Generator**:
    - ConvTranspose2d blocks (DCGAN style)

    - ReLU activations

    - Tanh output

- **Discriminator**:
    - Conv2d downsampling

    - LeakyReLU

    - Optional batch norm

    - BCEWithLogitsLoss

## Example MNIST Results

- fake epoch - 5:
    [](gan_outputs\fake_epoch_5.png)
# Visor

A convolutional autoencoder built in PyTorch, trained on MNIST.

## What it does

Visor takes a handwritten digit image, compresses it through two
convolutional layers into a much smaller representation (8 channels at
7x7, compared to the original 28x28 image). Then it reconstructs the image
back to its original size using another set of layers.

The network is trained by comparing its reconstruction to the original
image and minimizing the difference (Mean Squared Error loss). It only
seems images itself.

After training, the script pulls a test image and shows three things
side by side:
- the original image
- the compressed feature maps (what the network is actually keeping)
- the reconstructed image

## Why

I wanted to see what an autoencoder actually does, and see if I could build
one myself.

## Run it

    pip install torch torchvision matplotlib
    python visor.py

MNIST should download automatically on first run.

## Status

Early learning project. Revisiting/reviewing it now, so implementation
details may get cleaned up.

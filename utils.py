import random
import matplotlib.pyplot as plt

import torch
import torchvision


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def plot_image(original_image, reconstructed_image, random_image, cmap=None):
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original_image.permute(1, 2, 0).cpu(), cmap=cmap)
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.title("Reconstructed")
    plt.imshow(reconstructed_image.permute(1, 2, 0).cpu(), cmap=cmap)
    plt.axis(False)

    plt.subplot(1, 3, 3)
    plt.title("Random") 
    plt.imshow(random_image.permute(1, 2, 0).cpu(), cmap=cmap)
    plt.axis(False)

    plt.show()
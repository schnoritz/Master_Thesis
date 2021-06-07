import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == "__main__":

    trainings_masks = "/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/p_17/training_data.pt"
    target_masks = "/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/p_17/target_data.pt"

    training = torch.load(trainings_masks)
    target = torch.load(target_masks)

    fig, ax = plt.subplots(1, 6, figsize=(48, 8))
    ax[0].imshow(training[0, :, :, 37])
    ax[1].imshow(training[1, :, :, 37])
    ax[2].imshow(training[2, :, :, 37])
    ax[3].imshow(training[3, :, :, 37])
    ax[4].imshow(training[4, :, :, 37])
    ax[5].imshow(target[:, :, 37])
    plt.show()

    fig, ax = plt.subplots(1, 4, figsize=(32, 8))
    ax[0].imshow(training[0, :, :, 37])
    ax[0].imshow(target[:, :, 37], alpha=0.5)
    ax[1].imshow(training[1, :, :, 37])
    ax[1].imshow(training[2, :, :, 37], alpha=0.5)
    ax[2].imshow(training[1, :, :, 37])
    ax[2].imshow(target[:, :, 37], alpha=0.5)
    ax[3].imshow(training[0, :, :, 37])
    ax[3].imshow(training[3, :, :, 37], alpha=0.1)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from pprint import pprint
import os


def check_improvement(epochs, top_k=5):

    curr_epoch = epochs[-1]
    epochs = sorted(epochs, key=lambda k: k['test_loss'])
    if epochs.index(curr_epoch) < top_k:
        if len(epochs) > top_k:
            return epochs[top_k]["epoch"]
        else:
            return True
    else:
        return False


if __name__ == "__main__":

    segments = [f"p_{i}" for i in range(40)]

    training = torch.randn((5, 512, 512, 110))
    target = torch.randn((1, 512, 512, 110))

    for seg in segments:
        os.mkdir(
            f"/home/baumgartner/sgutwein84/container/trainings_data/{seg}"
        )

        torch.save(
            training,
            f'/home/baumgartner/sgutwein84/container/trainings_data/{seg}/training_data.pt'
        )

        torch.save(
            target,
            f'/home/baumgartner/sgutwein84/container/trainings_data/{seg}/target_data.pt'
        )

    # epochs = []
    # for epoch in range(100):

    #     train_loss = np.round(random.uniform(0.5, 10), 4)
    #     test_loss = np.round(train_loss + random.uniform(-0.5, 0.5), 4)

    #     epochs.append({
    #         "epoch": epoch+1,
    #         "train_loss": train_loss,
    #         "test_loss": test_loss
    #     })

    # pprint(sorted(epochs, key=lambda k: k['test_loss'])[:5])

    # trainings_masks = "/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/p_17/training_data.pt"
    # target_masks = "/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/p_17/target_data.pt"

    # training = torch.load(trainings_masks)
    # target = torch.load(target_masks)

    # fig, ax = plt.subplots(1, 6, figsize=(48, 8))
    # ax[0].imshow(training[0, :, :, 37])
    # ax[1].imshow(training[1, :, :, 37])
    # ax[2].imshow(training[2, :, :, 37])
    # ax[3].imshow(training[3, :, :, 37])
    # ax[4].imshow(training[4, :, :, 37])
    # ax[5].imshow(target[:, :, 37])
    # plt.show()

    # fig, ax = plt.subplots(1, 4, figsize=(32, 8))
    # ax[0].imshow(training[0, :, :, 37])
    # ax[0].imshow(target[:, :, 37], alpha=0.5)
    # ax[1].imshow(training[1, :, :, 37])
    # ax[1].imshow(training[2, :, :, 37], alpha=0.5)
    # ax[2].imshow(training[1, :, :, 37])
    # ax[2].imshow(target[:, :, 37], alpha=0.5)
    # ax[3].imshow(training[0, :, :, 37])
    # ax[3].imshow(training[3, :, :, 37], alpha=0.1)
    # plt.show()

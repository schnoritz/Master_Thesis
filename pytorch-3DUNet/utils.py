import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def plot_patches(patches, target, idx):

    fig, axs = plt.subplots(1, 6, figsize=(24, 4))
    img = 0
    for b in patches:
        i = 0
        img += 1
        for p in b:
            axs[i].imshow(p[:, :, p.shape[2]//2])
            i += 1

        axs[5].imshow(target[img-1, 0, :, :, target.shape[2]//2])
        axs[5].set_title("Target Patch")
        for i in range(5):
            axs[i].set_title("Training Patch")

        name = "/home/baumgartner/sgutwein84/training_data/output_images/image" + \
            str(img) + str(idx) + ".png"

        if os.path.isfile(name):
            os.remove(name)

        plt.savefig(name)

    plt.close()


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

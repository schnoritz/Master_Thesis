import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.pylab as pl


def latex(func):
    '''Decorator that sets latex parameters for plt'''

    def wrap(*args, **kwargs):

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 11
        result = func(*args, **kwargs)

        return result

    return wrap


def latex_plot(fig, width, height, path):

    cm = 1/2.54
    pl.figure(fig)
    fig.set_size_inches(width*cm, height*cm, forward=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')


@latex
def main():

    segment = "/mnt/qb/baumgartner/sgutwein84/training/training_prostate/p35_17"

    masks = torch.load(os.path.join(segment, "training_data.pt"))
    target = np.array(torch.load(os.path.join(segment, "target_data.pt")))
    target = target.squeeze()

    idx = np.argwhere(target == target.max())
    print(idx)

    fig, axs = plt.subplots(1, 5)

    for i in range(len(axs)):
        axs[i].imshow(masks[i, :, :, idx[0][2]], cmap="Oranges")
        axs[i].axis("off")

    latex_plot(fig, 14, 4, "/mnt/qb/baumgartner/sgutwein84/masks.pdf")


if __name__ == "__main__":
    main()

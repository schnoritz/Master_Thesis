import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.pylab as pl


def latex(width, height, path):
    '''Decorator that sets latex parameters for plt'''
    def do_latex(func):
        def wrap(*args, **kwargs):

            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Palatino"
            plt.rcParams["font.size"] = 9
            fig = func(*args, **kwargs)
            cm = 1/2.54
            fig.set_size_inches(width*cm, height*cm, forward=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
        return wrap
    return do_latex


@latex(width=14, height=5, path="/Users/simongutwein/Desktop/masks.pdf")
def main():

    segment = "/Users/simongutwein/Studium/Masterarbeit/p35_17"

    masks = torch.load(os.path.join(segment, "training_data.pt"))
    target = np.array(torch.load(os.path.join(segment, "target_data.pt")))
    target = target.squeeze()

    idx = np.argwhere(target == target.max())

    fig, axs = plt.subplots(1, 6)

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", ]
    mask = [0, 3, 4, 1, 2]

    for num, i in zip(mask, range(5)):
        axs[i].imshow(masks[num, :, :, idx[0][2]], cmap="Oranges")
        axs[i].axis("off")
        axs[i].text(256, 700, labels[i], va="center", ha="center")

    axs[5].imshow(target[:, :, idx[0][2]], cmap="Oranges")
    axs[5].axis("off")
    axs[5].text(256, 700, labels[5], va="center", ha="center")

    axs[2].annotate('Input', xy=(0.5, 1.2), xytext=(0.5, 1.2), xycoords='axes fraction',
                    fontsize=9, ha='center', va='bottom', bbox=dict(boxstyle='square', edgecolor='white', fc='white'),
                    arrowprops=dict(arrowstyle='-[, widthB=14, lengthB=0.7', lw=1.0))
    axs[5].annotate('Output', xy=(0.5, 1.2), xytext=(0.5, 1.2), xycoords='axes fraction',
                    fontsize=9, ha='center', va='bottom', bbox=dict(boxstyle='square', edgecolor='white', fc='white'),
                    arrowprops=dict(arrowstyle='-[, widthB=2.2, lengthB=0.7', lw=1.0))
    axs[2].grid(True)
    fig.canvas.draw()
    return fig


if __name__ == "__main__":
    main()

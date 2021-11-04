import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def latex(width, height, path):
    '''Decorator that sets latex parameters for plt'''
    def do_latex(func):
        def wrap(*args, **kwargs):

            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Palatino"
            plt.rcParams["font.size"] = 11
            fig = func(*args, **kwargs)
            cm = 1/2.54
            fig.set_size_inches(width*cm, height*cm, forward=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
        return wrap
    return do_latex


@latex(width=20, height=12, path="/Users/simongutwein/Desktop/example_mt1.pdf")
def example():

    import torch
    pat = "mt1"

    pred = np.array(torch.load(f"/Users/simongutwein/Studium/Masterarbeit/plan_predictions_test/{pat}/prediction.pt"))
    target = np.array(torch.load(f"/Users/simongutwein/Studium/Masterarbeit/plan_predictions_test/{pat}/target.pt"))

    ct = np.array(torch.load(f"/Users/simongutwein/Studium/Masterarbeit/plan_predictions_test/{pat}/training_data.pt"))[1]

    import pymedphys

    slice_ = np.argwhere(target == target.max())[0][2]

    gamma_options = {
        'dose_percent_threshold': 3,
        'distance_mm_threshold': 3,
        'lower_percent_dose_cutoff': 10,
        'interp_fraction': 3,  # Should be 10 or more for more accurate results
        'max_gamma': 2,
        'quiet': False,
        'local_gamma': False,
    }

    pred = pred[:, :, slice_-3:slice_+3]
    target = target[:, :, slice_-3:slice_+3]
    ct = ct[:, :, slice_-3:slice_+3]

    coords = (np.arange(0, 1.17*target.shape[0], 1.17), np.arange(
        0, 1.17*target.shape[1], 1.17), np.arange(0, 3*target.shape[2], 3))

    gamma_val = pymedphys.gamma(
        coords, np.array(target),
        coords, np.array(pred),
        **gamma_options)

    import matplotlib.pyplot as plt

    cut_top = 150

    total = 300

    reds = cm.get_cmap('Reds', 256)
    blues = cm.get_cmap('Blues', 256)
    top = reds(np.linspace(0, 1, int(1/2*total)))
    bottom = np.flip(blues(np.linspace(0, 1, int(1/2*total))), axis=0)
    newcmp = np.concatenate([bottom, top])
    newcmp = ListedColormap(newcmp)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    pred[pred < 0.1*target.max()] = np.nan
    target[target < 0.1*target.max()] = np.nan
    ct[ct < (80/1800)] = np.nan

    ax[0].imshow(ct[cut_top:, :, 3], cmap="bone")
    ax[0].imshow(target[cut_top:, :, 3], cmap="Oranges", alpha=0.8)
    ax[0].axis("off")
    ax[1].imshow(ct[cut_top:, :, 3], cmap="bone")
    ax[1].imshow(pred[cut_top:, :, 3],  cmap="Oranges", alpha=0.8)
    ax[1].axis("off")
    ax[2].imshow(ct[cut_top:, :, 3], cmap="bone")
    axs = ax[2].imshow(gamma_val[cut_top:, :, 3],  cmap=newcmp)
    cbar = fig.colorbar(axs, fraction=0.035, pad=0.04)
    cbar.ax.set_yticklabels(["  "])
    cbar.ax.tick_params(size=0, rotation=90)
    ax[2].axis("off")
    cbar.ax.text(6, 1, "1", rotation=90, va='center', ha="center", color='black')
    cbar.ax.text(6, 2, "failed", rotation=90, va='top', ha="center", color='black')
    cbar.ax.text(6, 0, "passed", rotation=90, va='bottom', ha="center", color='black')

    return fig


if __name__ == "__main__":
    example()
    # main()

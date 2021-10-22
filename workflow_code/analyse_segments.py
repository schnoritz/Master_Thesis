
import matplotlib.pylab as plt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


def latex(func):
    '''Decorator that sets latex parameters for plt'''

    def wrap(*args, **kwargs):

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 11
        result = func(*args, **kwargs)

        return result

    return wrap


def set_latex_params():
    cm = 1/2.54
    plt.rcParams["font.monospace"] = "Computer Modern"
    plt.rcParams["font.size"] = 11


def latex_plot(fig, width, height, path):

    cm = 1/2.54
    plt.rcParams["font.monospace"] = "Computer Modern"
    plt.rcParams["font.size"] = 12
    pl.figure(fig)
    fig.set_size_inches(width*cm, height*cm, forward=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_histogram(heights, positions):

    width = [(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
    pos = [(positions[i+1] + positions[i])/2 for i in range(len(positions)-1)]

    plt.bar(pos, heights, width=width, align="center", edgecolor='black', linewidth=1.2, alpha=0.5)


@latex
def main():

    import seaborn as sns

    results = "/Users/simongutwein/Desktop/tests.pkl"

    df = pd.read_pickle(results)
    sizes = df["size"]
    model = df["model"]
    model[model == "mixed_trained_UNET_1183.pt"] = "mixed"
    model[model == "prostate_trained_UNET_2234.pt"] = "prostate"
    df["model"] = model

    max_s = sizes.max()
    max_s = (np.ceil((max_s/10)+1)*10).astype(int)
    df['discrete'] = pd.cut(df["size"], list(range(0, max_s, 10)), include_lowest=True)
    df["discrete"] = [int(x.right) for x in df["discrete"]]
    hist = np.histogram(df["size"][::2], bins=list(range(0, max_s, 10)))

    fig, ax = plt.subplots()

    ax.bar(list(range(int(max_s/10)-1)), hist[0], alpha=0.2, width=1, linewidth=2, edgecolor="black", facecolor="black", zorder=0)
    ax2 = ax.twinx()
    ax2 = sns.boxplot(x="discrete", y="gamma", hue="model", data=df, palette="Reds", zorder=10)

    x_ticks = [f"({x} - {x+10}]" for x in range(0, max_s-10, 10)]
    ax.set_ylabel('Occurence')
    ax.set_xticklabels(x_ticks, rotation=45)

    ax2.set_ylabel(r'Gamma-Value /%')
    ax.set_xlabel("Fieldsize /$cm^2$")
    ax2.legend(bbox_to_anchor=(0.5, 1.1), borderaxespad=0, loc="center", ncol=2)

    latex_plot(fig, 14, 7.875, "/Users/simongutwein/Documents/GitHub/Master_Thesis/Masterarbeit_Text/Images/test.pdf")

    #tikzplotlib.save("/Users/simongutwein/Documents/GitHub/Master_Thesis/Masterarbeit_Text/Images/test.tex", axis_height=r"0.5\textwidth", axis_width=r"\textwidth")

    hist = np.histogram(df["size"][::2], bins=list(range(0, max_s, 10)))

    _, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[1] = sns.boxplot(x="discrete", y="gamma", hue="model", data=df, palette="Reds")
    # ax[].legend(bbox_to_anchor=(0.65, 1.2),
    #           borderaxespad=0)
    ax[1].set_xlabel("Fieldsize /mm")

    ax[0].bar(np.array(range(0, max_s-10, 10))+5, hist[0], width=10, edgecolor="black", linewidth=2, facecolor="firebrick")
    ax[0].set_ylabel('Occurence')
    ax[0].set_xlabel("Fieldsize /mm")
    ax[0].set_xlim(0, 130)
    ax[0].set_xticks(range(5, max_s-10, 10))
    ax[0].set_xticklabels(range(10, max_s, 10))

    plt.show()


if __name__ == "__main__":
    main()

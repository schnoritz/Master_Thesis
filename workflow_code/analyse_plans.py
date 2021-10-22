
import os
import pandas as pd
import numpy as np
import seaborn as sb

import matplotlib.pyplot as plt
import matplotlib.pylab as pl


def latex(width, height, path):
    '''Decorator that sets latex parameters for plt'''
    def do_latex(func):
        def wrap(*args, **kwargs):

            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.size"] = 11
            fig = func(*args, **kwargs)
            cm = 1/2.54
            fig.set_size_inches(width*cm, height*cm, forward=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
        return wrap
    return do_latex


def plot_histogram(heights, positions):

    width = [(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
    pos = [(positions[i+1] + positions[i])/2 for i in range(len(positions)-1)]

    plt.bar(pos, heights, width=width, align="center", edgecolor='black', linewidth=1.2, alpha=0.5)


@latex(width=14, height=12, path="/Users/simongutwein/Desktop/test.pdf")
def main():

    results = "/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_results"

    test_cases = [x for x in os.listdir(results) if not x.startswith(".")]

    info = []
    for test_case in test_cases:
        case_path = os.path.join(results, test_case)
        models = [x for x in os.listdir(case_path) if not x.startswith(".")]
        for model in models:
            gamma_path = os.path.join(case_path, model, "gamma.txt")

            with open(gamma_path) as fin:
                gamma = float(fin.readlines()[-1])

            print(test_case)
            segment_path = os.path.join("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_cases", test_case)
            segments = [x for x in os.listdir(segment_path) if not x.startswith(".") and not "ct" in x and not "dcm" in x]
            modulated = len(segments)

            info.append({
                "plan": test_case,
                "model": model,
                "gamma": gamma,
                "modulated": modulated
            })

    df = pd.DataFrame(info)
    df.to_excel("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_case_results.xlsx")
    df.to_pickle("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_case_results.pkl")

    df = pd.read_pickle("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_case_results.pkl")

    df.loc[df.model == "mixed_trained_UNET_1183.pt", "model"] = "mixed"
    df.loc[df.model == "prostate_trained_UNET_2234.pt", "model"] = "prostate"

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.5, 3]})
    sb.boxplot(x="gamma", y="model", data=df, palette="Reds", ax=ax[0], width=0.5)
    sb.swarmplot(y="model", x="gamma", data=df, color="black", ax=ax[0])

    sb.scatterplot(x="plan", y="gamma", style="model", hue="model", data=df, palette="Reds", markers=['D', 'X'], ax=ax[1])
    ax[0].set_xlabel("Gamma Passrate /%")
    ax[1].set_xlabel("Test Case")
    ax[1].tick_params(axis='x', rotation=-45)
    ax[1].set_ylabel("Gamma Passrate /%")
    ax[0].set_ylabel("")
    legend = ax[1].legend()
    legend.texts[0].set_text("mixed")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()

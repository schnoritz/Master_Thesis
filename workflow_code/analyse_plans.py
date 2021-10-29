
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
            plt.rcParams["font.serif"] = "Palatino"
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


@latex(width=20, height=12, path="/Users/simongutwein/Desktop/test.pdf")
def main():

    colors = ["#FF0B04", "#4374B3"]
    sb.set_palette(sb.color_palette(colors))

    # results = "/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_results"

    # test_cases = [x for x in os.listdir(results) if not x.startswith(".")]

    # info = []
    # for test_case in test_cases:
    #     case_path = os.path.join(results, test_case)
    #     models = [x for x in os.listdir(case_path) if not x.startswith(".")]
    #     for model in models:
    #         gamma_path = os.path.join(case_path, model, "gamma.txt")

    #         with open(gamma_path) as fin:
    #             gamma = float(fin.readlines()[-1])

    #         print(test_case)
    #         segment_path = os.path.join("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_cases", test_case)
    #         segments = [x for x in os.listdir(segment_path) if not x.startswith(".") and not "ct" in x and not "dcm" in x]
    #         modulated = len(segments)

    #         if "m" in test_case:
    #             test_modularity = "Mixed (2)"
    #         elif "l" in test_case:
    #             test_modularity = "Mixed (2)"
    #         elif "h" in test_case:
    #             test_modularity = "Mixed (2)"
    #         elif "p" in test_case:
    #             test_modularity = "Prostate (1)"
    #         elif "n" in test_case:
    #             test_modularity = "Lymphnodes (3)"

    #         info.append({
    #             "plan": test_case,
    #             "model": model,
    #             "gamma": gamma,
    #             "modulated": modulated,
    #             "test_modularity": test_modularity
    #         })

    # df = pd.DataFrame(info)

    # df.to_excel("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_case_results.xlsx")
    # df.to_pickle("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_case_results.pkl")

    df = pd.read_pickle("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_case_results.pkl")

    df.loc[df.model == "mixed_trained_UNET_1183.pt", "model"] = "Mixed Model (B)"
    df.loc[df.model == "prostate_trained_UNET_2234.pt", "model"] = "Prostate Model (A)"

    print(df.groupby(['test_modularity', 'model']).describe())
    test = df.groupby(['test_modularity', 'model']).describe()

    fig, ax = plt.subplots(1, 1)
    sb.violinplot(cut=0, inner="point", x="test_modularity", y="gamma", hue="model", data=df, hue_order=[
                  "Prostate Model (A)", "Mixed Model (B)"], order=["Prostate (1)", "Mixed (2)", "Lymphnodes (3)"], dodge=True)
    # sb.swarmplot(x="test_modularity", y="gamma", hue="model", data=df, color="black", dodge=True, size=4, hue_order=[
    # Prostate Model (A)", "Mixed Model (B)"], order = ["Prostate (1)", "Mixed (2)", "Lymphnodes (3)"])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: 2], labels[: 2])
    ax.set_ylabel("Gamma Passrate [%]")
    ax.set_xlabel("")
    #ax.set_ylim([-5, 105])
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()

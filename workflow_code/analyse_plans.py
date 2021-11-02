
import os
import pandas as pd
import numpy as np
import seaborn as sb
import scipy.stats as ss

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


@latex(width=14, height=12, path="/Users/simongutwein/Desktop/plans.pdf")
def main():

    # colors = ["#FF0B04", "#4374B3"]
    # sb.set_palette(sb.color_palette(colors))

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

    test_cases = ["Prostate", "Mixed", "Lymphnodes"]

    counts = []
    wilcox = []
    for i in range(3):
        temp_df = df.loc[df['test_modularity'].str.contains(test_cases[i])]
        count_df = temp_df[temp_df['model'].str.contains("(A)")]
        test_cases[i] = test_cases[i] + f"\nn={count_df.count()[0]}"
        counts.append(temp_df.count()[0])
        wilcox.append(ss.wilcoxon(temp_df[temp_df['model'].str.contains("(A)")]["gamma"], temp_df[temp_df['model'].str.contains("(B)")]["gamma"])[1])

    print(wilcox)

    for i in range(len(wilcox)):
        if wilcox[i] < 1E-3:
            wilcox[i] = f"***"
        elif wilcox[i] < 1E-2:
            wilcox[i] = f"**"
        elif wilcox[i] > 0.05:
            wilcox[i] = f"n.s."

    fig, ax = plt.subplots(1, 1)
    sb.violinplot(cut=0, inner="box", x="test_modularity", y="gamma", hue="model", data=df, hue_order=[
                  "Prostate Model (A)", "Mixed Model (B)"], palette="Reds", order=["Prostate (1)", "Mixed (2)", "Lymphnodes (3)"], dodge=True)
    # sb.swarmplot(x="test_modularity", y="gamma", hue="model", data=df, color="black", dodge=True, size=4, hue_order=[
    # Prostate Model (A)", "Mixed Model (B)"], order = ["Prostate (1)", "Mixed (2)", "Lymphnodes (3)"])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: 2], labels[: 2], loc="lower right")
    ax.set_xticklabels(test_cases)
    ax.set_ylabel("Gamma Passrate [%]")
    ax.set_xlabel("")
    ax.set_ylim([50, 109])

    for axs in ax.collections:
        axs.set_edgecolor('k')

    for axs in ax.get_children()[14:26]:
        axs.set_color('k')

    for i in range(3):
        x1, x2 = i-0.2, i+0.2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
        y, h, col = 103, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
        plt.text((x1+x2)*.5, y+h, wilcox[i], ha='center', va='bottom', color=col)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    main()

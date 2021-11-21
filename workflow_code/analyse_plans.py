
import os

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sb


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


@latex(width=12, height=14, path="/Users/simongutwein/Desktop/plans_new.pdf")
def main():

    colors = ["#FF0B04", "#4374B3"]
    sb.set_palette(sb.color_palette(colors))

    results = "/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/plan_predictions_load/"

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

            if "m" in test_case:
                test_modularity = "Mixed (2)"
            elif "l" in test_case:
                test_modularity = "Mixed (2)"
            elif "h" in test_case:
                test_modularity = "Mixed (2)"
            elif "p" in test_case:
                test_modularity = "Prostate (1)"
            elif "n" in test_case:
                test_modularity = "Lymphnodes (3)"

            info.append({
                "plan": test_case,
                "model": model,
                "gamma": gamma,
                "modulated": modulated,
                "test_modularity": test_modularity
            })

    df = pd.DataFrame(info)

    df.to_excel("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_case_results.xlsx")
    df.to_pickle("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_case_results.pkl")

    df = pd.read_pickle("/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_case_results.pkl")

    df.loc[df.model == "mixed_trained_UNET_1183.pt", "model"] = "Mixed Model"
    df.loc[df.model == "prostate_trained_UNET_2234.pt", "model"] = "Prostate Model"

    test_cases = ["Prostate", "Mixed", "Lymphnodes"]

    counts = []
    wilcox = []
    for i in range(3):
        temp_df = df.loc[df['test_modularity'].str.contains(test_cases[i])]
        count_df = temp_df[temp_df['model'].str.contains("Prostate")]
        test_cases[i] = test_cases[i] + f"\nn={count_df.count()[0]}"
        counts.append(temp_df.count()[0])
        wilcox.append(ss.wilcoxon(temp_df[temp_df['model'].str.contains("Prostate")]["gamma"], temp_df[temp_df['model'].str.contains("Mixed")]["gamma"])[1])
        # print(test_cases[i], temp_df[temp_df['model'].str.contains("Prostate")]["gamma"].mean(), temp_df[temp_df['model'].str.contains("Mixed")]["gamma"].mean(),
        #       temp_df[temp_df['model'].str.contains("Mixed")]["gamma"].mean()-temp_df[temp_df['model'].str.contains("Prostate")]["gamma"].mean())

    print(wilcox)
    for i in range(len(wilcox)):
        if wilcox[i] < 1E-3:
            wilcox[i] = f"***"
        elif wilcox[i] < 1E-2:
            wilcox[i] = f"**"
        elif wilcox[i] < 0.05:
            wilcox[i] = f"*"
        elif wilcox[i] > 0.05:
            wilcox[i] = f"n.s."

    entities = ["lt", "mt", "ht"]
    entities_label = ["Liver", "Mamma", "H&N"]
    entities_count = []
    entities_wilcox = []

    for i in range(3):
        temp_df = df.loc[df['plan'].str.contains(entities[i])]
        print(temp_df)
        count_df = temp_df[temp_df['model'].str.contains("Prostate")]
        entities_label[i] = entities_label[i] + f"\nn={count_df.count()[0]}"
        entities_count.append(temp_df.count()[0])

    df_entities = df.loc[df['plan'].str.contains("|".join(entities))]
    for i in range(3):
        df_entities.loc[df_entities['plan'].str.contains(entities[i]), 'plan'] = entities[i]

    fig, ax = plt.subplots(2, 1)

    sb.violinplot(cut=0, inner="box", x="plan", y="gamma", hue="model", data=df_entities, hue_order=[
                  "Prostate Model", "Mixed Model"], palette="Reds", order=["lt", "mt", "ht"], ax=ax[0], dodge=True, legend=True)

    ax[0].get_legend().remove()
    ax[0].set_xlim([-0.5, 2.5])
    ax[0].set_xticks([])
    ax[0].text(0, 116, entities_label[0], ha="center", va="top", fontsize=9)
    ax[0].text(1, 116, entities_label[1], ha="center", va="top", fontsize=9)
    ax[0].text(2, 116, entities_label[2], ha="center", va="top", fontsize=9)
    ax[0].set_xticklabels([])
    ax[0].set_ylabel("Gamma Passrate [%]")
    ax[0].set_xlabel("")
    #ax[0].tick_params(axis="x", direction="in", pad=-30)
    ax[0].set_ylim([0, 119])
    ax[0].plot([-0.5, 2.5], [100, 100], "--", linewidth=1, color="k")

    sb.violinplot(cut=0, inner="box", x="test_modularity", y="gamma", hue="model", data=df, hue_order=[
                  "Prostate Model", "Mixed Model"], palette="Reds", order=["Prostate (1)", "Mixed (2)", "Lymphnodes (3)"], dodge=True, ax=ax[1])

    # sb.swarmplot(x="test_modularity", y="gamma", hue="model", data=df, color="black", dodge=True, size=4, hue_order=[
    # Prostate Model (A)", "Mixed Model (B)"], order = ["Prostate (1)", "Mixed (2)", "Lymphnodes (3)"])
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles[: 2], ["Prostate", "Mixed"], loc="lower left", facecolor="white").set_zorder(2.5)
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xlim([-0.5, 2.5])
    ax[1].set_xticklabels(test_cases)
    ax[1].set_ylabel("Gamma Passrate [%]")
    ax[1].set_xlabel("")
    ax[1].set_ylim([0, 119])
    ax[1].plot([-0.5, 2.5], [100, 100], "--", linewidth=1, color="k")

    for axs in ax[1].collections:
        axs.set_edgecolor('k')
    for axs in ax[0].collections:
        axs.set_edgecolor('k')

    for axs in ax[1].get_children()[14:26]:
        axs.set_color('k')
    for axs in ax[0].get_children()[12:26]:
        axs.set_color('k')

    for i in range(1, 3):
        x1, x2 = i-0.2, i+0.2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
        y, h, col = 103, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
        plt.text((x1+x2)*.5, y+h, wilcox[i], ha='center', va='bottom', color=col)

    ax2 = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    ax2.axis("off")
    x, y = np.array([[0.4, 0.4, 0.11, 0.974, 0.7, 0.7], [0.1225, 0.525, 0.5642, 0.5642, 0.525, 0.1225]])
    line = Line2D(x, y, lw=1, color='k', zorder=0)
    ax2.add_line(line)
    x, y = np.array([[0.7, 0.7], [0.1225, 0.131]])
    line = Line2D(x, y, lw=1, color='k', zorder=0)
    ax2.add_line(line)
    ax2.text(0.55, 0.542, "Individual Analysis", va="center", ha="center")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.1)

    return fig


if __name__ == "__main__":
    main()

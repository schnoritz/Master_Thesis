
from operator import pos
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
from tqdm import tqdm
import scipy.stats as ss


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


def get_results(root):

    patients = [x for x in os.listdir(root) if not x.startswith(".")]
    results = []
    for patient in tqdm(patients):
        pat_dir = os.path.join(root, patient)
        segments = [x for x in os.listdir(pat_dir) if not x.startswith(".")]

        for segment in tqdm(segments):
            mix_dir = os.path.join(root, patient, segment, "mixed_trained_UNET_1183.pt", "gamma.txt")
            prost_dir = os.path.join(root, patient, segment, "prostate_trained_UNET_2234.pt", "gamma.txt")
            print(mix_dir)
            with open(mix_dir, "r") as fin:
                lines = fin.readlines()
                mix_result = lines[4]
                size = lines[8][7:-2]

            with open(prost_dir, "r") as fin:
                prost_result = fin.readlines()[4]

            results.append({
                "segment": segment,
                "mixed": mix_result,
                "prostate": prost_result,
                "fieldsize": size
            })

    df = pd.DataFrame(results)
    df.to_pickle("/Users/simongutwein/Desktop/segment_results.pkl")


@latex(width=14, height=8, path="/Users/simongutwein/Desktop/segs.pdf")
def fieldsize_analysis():

    # root = "/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/segment_results"
    # get_results(root)

    n = 15

    results = "/Users/simongutwein/Desktop/segment_results.pkl"

    df = pd.read_pickle(results)
    df = pd.melt(df, id_vars=['segment', "fieldsize"], value_vars=['mixed', 'prostate'])
    df["fieldsize"] = df["fieldsize"].astype(float)
    df["value"] = df["value"].replace("\n", "").astype(float)

    sizes = df["fieldsize"]
    max_s = sizes.max()
    max_s = (np.ceil((max_s/n)+1)*n).astype(int)
    df['discrete'] = pd.cut(df["fieldsize"], list(range(0, max_s, n)), include_lowest=True)
    df["discrete"] = [int(x.right) for x in df["discrete"]]

    hist = list(np.histogram(df["fieldsize"][::2], bins=list(range(0, max_s, n)), density=True))

    fig, ax = plt.subplots()
    ax.bar(list(range(int(max_s/n)-1)), hist[0]*100*n, width=1, linewidth=1, edgecolor="grey", facecolor="lightgrey", zorder=0)
    ax2 = ax.twinx()
    ax2 = sns.boxplot(x="discrete", y="value", hue="variable", data=df, palette="Reds", color=1, hue_order=["prostate", "mixed"], zorder=10, fliersize=2)
    for axs in ax2.get_children():
        print(axs)
        if isinstance(axs, mpl.lines.Line2D):
            axs.set_color('k')
            axs.set_linewidth(1)
        if isinstance(axs, mpl.patches.PathPatch):
            axs.set_edgecolor('k')
            axs.set_linewidth(1)

    _, labels = plt.xticks()
    labels = [int(labels[x].get_text()) for x in range(len(labels))]
    labels.insert(0, 0)
    x_ticks = [f"({labels[i]} - {labels[i+1]}]" for i in range(0, len(labels)-1)]
    ax.set_ylabel('Occurence /%')
    ax.set_xticklabels(x_ticks, rotation=45)

    ax2.set_ylabel(r'Gamma-Value /%')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")

    ax.set_xlabel("Fieldsize /$cm^2$")
    ax2.legend(bbox_to_anchor=(0.5, 1.1), borderaxespad=0, loc="center", ncol=2)

    return fig


@latex(width=7, height=10, path="/Users/simongutwein/Desktop/segs_all.pdf")
def overall_performance():

    # root = "/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/segment_results"
    # get_results(root)

    results = "/Users/simongutwein/Desktop/segment_results.pkl"

    df = pd.read_pickle(results)
    df = pd.melt(df, id_vars=['segment', "fieldsize"], value_vars=['mixed', 'prostate'])
    df["value"] = df["value"].replace("\n", "").astype(float)

    fig, ax = plt.subplots()
    ax = sns.violinplot(inner="box", x="variable", y="value", data=df, palette="Reds", order=["prostate", "mixed"], cut=0)
    #ax2 = sns.swarmplot(x="variable", y="value", data=df, color="black", size=1, order=["prostate", "mixed"])
    for axs in ax.get_children()[:4]:
        print(axs)
        axs.set_edgecolor('k')
    for axs in ax.get_children()[4:12]:
        print(axs)
        axs.set_color('k')

    ax.set_xticklabels(["Prostate Model", "Mixed Model"])
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylabel(r'Gamma-Value /%')
    ax.set_xlabel("")
    ax.set_ylim([0, 119])

    wilcox = np.round(ss.wilcoxon(df[df['variable'].str.match("prostate")]["value"], df[df['variable'].str.match("mixed")]["value"])[1], 5)

    n = df[df['variable'].str.match("prostate")].count()[0]

    if wilcox < 1E-3:
        wilcox = f"***"
    else:
        wilcox = f"p = {wilcox}\nn={n}"

    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = 108, 3, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    plt.text((x1+x2)*.5, y+h-2, wilcox, ha='center', va='bottom', color=col)

    plt.text(0.5, 107, f"n={n}", ha='center', va='center')

    return fig


@latex(width=12, height=14, path="/Users/simongutwein/Desktop/segs_test.pdf")
def test_set_performance():

    # root = "/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/segment_results"
    # get_results(root)

    results = "/Users/simongutwein/Desktop/segment_results.pkl"

    df = pd.read_pickle(results)
    df = pd.melt(df, id_vars=['segment', "fieldsize"], value_vars=['mixed', 'prostate'])
    df["value"] = df["value"].replace("\n", "").astype(float)

    df["test"] = ""
    df.loc[df['segment'].str.match("ht"), "test"] = "Mixed"
    df.loc[df['segment'].str.match("mt"), "test"] = "Mixed"
    df.loc[df['segment'].str.match("lt"), "test"] = "Mixed"
    df.loc[df['segment'].str.match("pt"), "test"] = "Prostate"
    df.loc[df['segment'].str.match("nt"), "test"] = "Lymphnodes"

    fig, ax = plt.subplots()
    ax = sns.violinplot(x="test", y="value", hue="variable", data=df, palette="Reds", inner="box", order=["Prostate", "Mixed", "Lymphnodes"], hue_order=["prostate", "mixed"], cut=0)
    for axs in ax.collections:
        axs.set_edgecolor('k')

    for axs in ax.get_children()[14:26]:
        axs.set_color('k')

    ax.set_ylim([0, 110])

    ax.set_ylabel(r'Gamma-Value /%')
    ax.set_xlabel("")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: 2], labels[: 2], loc="lower right")

    wilcox = []
    counts = []
    test_cases = ["Prostate", "Mixed", "Lymphnodes"]

    for i in range(3):
        temp_df = df.loc[df['test'].str.match(test_cases[i])]
        test_cases[i] = test_cases[i] + f"\nn={temp_df.count()[0]}"
        counts.append(temp_df.count()[0])
        wilcox.append(ss.wilcoxon(temp_df[temp_df['variable'].str.match("prostate")]["value"], temp_df[temp_df['variable'].str.match("mixed")]["value"])[1])

    print(wilcox)
    ax.set_xticklabels(test_cases)

    for i in range(len(wilcox)):
        if wilcox[i] < 1E-3:
            wilcox[i] = f"***"
        else:
            wilcox[i] = f"p={wilcox[i]}"

    for i in range(3):
        x1, x2 = i-0.2, i+0.2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
        y, h, col = 103, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
        plt.text((x1+x2)*.5, y+h, wilcox[i], ha='center', va='bottom', color=col)

    return fig


if __name__ == "__main__":
    fieldsize_analysis()
    # overall_performance()
    # test_set_performance()

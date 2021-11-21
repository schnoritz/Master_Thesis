
from operator import pos
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sb
from tqdm import tqdm
import scipy.stats as ss
from matplotlib.lines import Line2D


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


@latex(width=14, height=16, path="/Users/simongutwein/Desktop/segs_weight_fz.pdf")
def fieldsize_analysis():

    # root = "/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/segment_results"
    # get_results(root)

    # from pydicom import dcmread

    # root = "/Users/simongutwein/mnt/qb/baumgartner/sgutwein84/test_cases"
    # plans = [x for x in os.listdir(root) if not x.startswith(".")]

    # weights = []
    # for plan in plans:

    #     pf = os.path.join(root, plan, [x for x in os.listdir(os.path.join(root, plan)) if not x.startswith(".") and "plan" in x][0])

    #     dicom_file = dcmread(pf, force=True)

    #     meterset_weights = []
    #     for beam in dicom_file.BeamSequence:
    #         for control_point in beam.ControlPointSequence:
    #             meterset_weights.append(float(control_point.CumulativeMetersetWeight))

    #     segment_weights = [meterset_weights[i] - meterset_weights[i-1] for i in range(1, len(meterset_weights), 2)]
    #     print(f"Plan has {len(segment_weights)} segments.")
    #     segment_names = [f"{plan}_{x}" for x in range(len(segment_weights))]

    #     weights.extend(zip(segment_weights, segment_names))

    # df_weight = pd.DataFrame(weights)
    # df_weight.to_pickle("/Users/simongutwein/Desktop/weights_results.pkl")

    n = 15

    df_weight = pd.read_pickle("/Users/simongutwein/Desktop/weights_results.pkl")
    df_fz = pd.read_pickle("/Users/simongutwein/Desktop/segment_results.pkl")

    new_df = pd.merge(df_weight, df_fz, left_on=1, right_on="segment")
    new_df = new_df.drop(columns=1)
    new_df = new_df.rename(columns={0: 'weight'})
    new_df["fieldsize"] = new_df["fieldsize"].astype(float)
    new_df["mixed"] = new_df["mixed"].replace("\n", "").astype(float)
    new_df["prostate"] = new_df["prostate"].replace("\n", "").astype(float)

    bins = [(np.round(x, 2), np.round(x+0.1, 2)) for x in np.linspace(0, 1, 11)][:-1]

    bins = pd.IntervalIndex.from_tuples(bins)
    new_df["dis_weight"] = pd.cut(new_df["weight"], bins)
    new_df["dis_weight"] = [float(x.right) for x in new_df["dis_weight"]]
    new_df = pd.melt(new_df, id_vars=["weight", "segment", "fieldsize", "dis_weight"], value_vars=["prostate", "mixed"])

    sizes = new_df["fieldsize"]
    max_s = sizes.max()
    max_s = (np.ceil((max_s/n)+1)*n).astype(int)
    new_df['dis_fz'] = pd.cut(new_df["fieldsize"], list(range(0, max_s, n)), include_lowest=True)
    new_df["dis_fz"] = [int(x.right) for x in new_df["dis_fz"]]

    hist = list(np.histogram(new_df["fieldsize"][::2], bins=list(range(0, max_s, n)), density=True))
    print(hist)
    fig, ax = plt.subplots(2, 1)
    ax[0].bar(list(range(int(max_s/n)-1)), hist[0]*100*n, width=1, linewidth=1, edgecolor="grey", facecolor="lightgrey", zorder=0)
    ax2 = ax[0].twinx()
    ax2 = sb.boxplot(x="dis_fz", y="value", hue="variable", data=new_df, palette="Reds", color=1, hue_order=["prostate", "mixed"], zorder=10, fliersize=2)
    for axs in ax2.get_children():
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
    ax[0].set_ylabel('Occurence /%')
    ax[0].set_xticklabels(x_ticks, rotation=45)

    ax2.set_ylabel(r'Gamma-Value /%')
    ax[0].yaxis.tick_right()
    ax[0].yaxis.set_label_position("right")

    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")

    ax[0].set_xlabel("Fieldsize /$cm^2$")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[: 2], ["Prostate", "Mixed"], bbox_to_anchor=(0.5, 1.1), borderaxespad=0, loc="center", ncol=2)

    hist = list(np.histogram(new_df["weight"], bins=np.linspace(0, 1, 11), density=True))
    ax[1].bar(list(range(10)), hist[0]*10, width=1, linewidth=1, edgecolor="grey", facecolor="lightgrey")

    ax3 = ax[1].twinx()
    ax3 = sb.boxplot(x="dis_weight", y="value", hue="variable", data=new_df, palette="Reds", color=1, hue_order=["prostate", "mixed"], zorder=10, fliersize=2)
    for axs in ax3.get_children():
        if isinstance(axs, mpl.lines.Line2D):
            axs.set_color('k')
            axs.set_linewidth(1)
        if isinstance(axs, mpl.patches.PathPatch):
            axs.set_edgecolor('k')
            axs.set_linewidth(1)

    labels = list(ax3.get_xticks())
    labels.append(10)
    x_ticks = [f"({np.round(labels[i]*0.1,1)} - {np.round(labels[i+1]*0.1,1)}]" for i in range(0, len(labels)-1)]
    ax[1].set_ylabel('Occurence /%')
    ax[1].set_xticklabels(x_ticks, rotation=45)

    ax3.set_ylabel(r'Gamma-Value /%')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    ax3.yaxis.tick_left()
    ax3.yaxis.set_label_position("left")

    ax[1].set_xticklabels(x_ticks, rotation=45)
    ax[1].set_xlabel("Weight /1")
    ax3.get_legend().remove()
    plt.subplots_adjust(hspace=0.5)

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
    ax = sb.violinplot(inner="box", x="variable", y="value", data=df, palette="Reds", order=["prostate", "mixed"], cut=0)
    #ax2 = sb.swarmplot(x="variable", y="value", data=df, color="black", size=1, order=["prostate", "mixed"])
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

    counts = []
    wilcox = []
    test_cases = ["Prostate", "Mixed", "Lymphnodes"]
    for i in range(3):
        temp_df = df.loc[df['test'].str.contains(test_cases[i])]
        count_df = temp_df[temp_df['variable'].str.contains("pros")]
        test_cases[i] = test_cases[i] + f"\nn={count_df.count()[0]}"
        counts.append(temp_df.count()[0])
        wilcox.append(ss.wilcoxon(temp_df[temp_df['variable'].str.contains("pros")]["value"], temp_df[temp_df['variable'].str.contains("mix")]["value"])[1])

    entities = ["lt", "mt", "ht"]
    entities_label = ["Liver", "Mamma", "H&N"]
    entities_count = []

    for i in range(3):
        temp_df = df.loc[df['segment'].str.contains(entities[i])]
        count_df = temp_df[temp_df['variable'].str.contains("pros")]
        entities_label[i] = entities_label[i] + f"\nn={count_df.count()[0]}"
        entities_count.append(temp_df.count()[0])

    df_entities = df.loc[df['segment'].str.contains("|".join(entities))]
    for i in range(3):
        df_entities.loc[df_entities['segment'].str.contains(entities[i]), 'plan'] = entities[i]

    fig, ax = plt.subplots(2, 1)

    sb.violinplot(cut=0, inner="box", x="plan", y="value", hue="variable", data=df_entities, hue_order=[
                  "prostate", "mixed"], palette="Reds", order=["lt", "mt", "ht"], ax=ax[0], dodge=True, legend=True)

    ax[0].get_legend().remove()
    ax[0].set_xlim([-0.5, 2.5])
    ax[0].set_xticks([])
    # ax[0].set_xticklabels(entities_label)
    #ax[0].set_xticklabels(["", "", ""])
    ax[0].text(0, 116, entities_label[0], ha="center", va="top", fontsize=9)
    ax[0].text(1, 116, entities_label[1], ha="center", va="top", fontsize=9)
    ax[0].text(2, 116, entities_label[2], ha="center", va="top", fontsize=9)
    ax[0].set_ylabel("Gamma Passrate [%]")
    ax[0].set_xlabel("")
    #ax[0].tick_params(axis="x", direction="in", pad=-140)
    ax[0].set_ylim([0, 119])
    #ax[0].set_yticklabels([" ", " ", "0", "25", "50", "75", "100"])
    ax[0].plot([-0.5, 2.5], [0, 0], "--", linewidth=1, color="k")
    ax[0].plot([-0.5, 2.5], [100, 100], "--", linewidth=1, color="k")
    sb.violinplot(cut=0, inner="box", x="test", y="value", hue="variable", data=df, hue_order=[
                  "prostate", "mixed"], palette="Reds", order=["Prostate", "Mixed", "Lymphnodes"], dodge=True, ax=ax[1])

    # sb.swarmplot(x="test_modularity", y="gamma", hue="model", data=df, color="black", dodge=True, size=4, hue_order=[
    # Prostate Model (A)", "Mixed Model (B)"], order = ["Prostate (1)", "Mixed (2)", "Lymphnodes (3)"])
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles[: 2], labels[: 2], loc="lower left", facecolor="white").set_zorder(2.5)
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xlim([-0.5, 2.5])
    ax[1].set_xticklabels(test_cases)
    ax[1].set_ylabel("Gamma Passrate [%]")
    ax[1].set_xlabel("")
    ax[1].set_ylim([0, 119])
    ax[1].plot([-0.5, 2.5], [100, 100], "--", linewidth=1, color="k")

    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles[: 2], ["Prostate", "Mixed"], loc="lower left", facecolor="white").set_zorder(2.5)

    for axs in ax[1].collections:
        axs.set_edgecolor('k')
    for axs in ax[0].collections:
        axs.set_edgecolor('k')

    for axs in ax[1].get_children()[14:26]:
        axs.set_color('k')
    for axs in ax[0].get_children()[12:26]:
        axs.set_color('k')

    print(wilcox)
    for i in range(len(wilcox)):
        if wilcox[i] < 1E-3:
            wilcox[i] = f"***"
        elif wilcox[i] < 1E-2:
            wilcox[i] = f"**"
        elif wilcox[i] > 0.05:
            wilcox[i] = f"n.s."

    for i in range(3):
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

    # fig, ax = plt.subplots()
    # ax = sb.violinplot(x="test", y="value", hue="variable", data=df, palette="Reds", inner="box", order=["Prostate", "Mixed", "Lymphnodes"], hue_order=["prostate", "mixed"], cut=0)
    # for axs in ax.collections:
    #     axs.set_edgecolor('k')

    # for axs in ax.get_children()[14:26]:
    #     axs.set_color('k')

    # ax.set_ylim([0, 110])

    # ax.set_ylabel(r'Gamma-Value /%')
    # ax.set_xlabel("")
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[: 2], labels[: 2], loc="lower right")

    # wilcox = []
    # counts = []
    # test_cases = ["Prostate", "Mixed", "Lymphnodes"]

    # for i in range(3):
    #     temp_df = df.loc[df['test'].str.match(test_cases[i])]
    #     test_cases[i] = test_cases[i] + f"\nn={temp_df.count()[0]}"
    #     counts.append(temp_df.count()[0])
    #     wilcox.append(ss.wilcoxon(temp_df[temp_df['variable'].str.match("prostate")]["value"], temp_df[temp_df['variable'].str.match("mixed")]["value"])[1])

    # print(wilcox)
    # ax.set_xticklabels(test_cases)

    # for i in range(len(wilcox)):
    #     if wilcox[i] < 1E-3:
    #         wilcox[i] = f"***"
    #     else:
    #         wilcox[i] = f"p={wilcox[i]}"

    # for i in range(3):
    #     x1, x2 = i-0.2, i+0.2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    #     y, h, col = 103, 2, 'k'
    #     plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    #     plt.text((x1+x2)*.5, y+h, wilcox[i], ha='center', va='bottom', color=col)

    # return fig


if __name__ == "__main__":
    fieldsize_analysis()
    # overall_performance()
    # test_set_performance()

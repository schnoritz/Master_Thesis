import os
import torch
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


def latex(func):
    '''Decorator that reports the execution time.'''

    def wrap(*args, **kwargs):

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 12
        result = func(*args, **kwargs)

        return result

    return wrap


def latex_plot(fig, width, height, path):

    cm = 1/2.54
    plt.rcParams["font.monospace"] = "Computer Modern"
    plt.rcParams["font.size"] = 12
    pl.figure(fig)
    fig.set_size_inches(width*cm, height*cm, forward=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')


@latex
def main():

    phantoms = ["phantomP100T200_10x10", "phantomP200T200_10x10", "phantomP300T200_10x10"]
    root_dir = "/mnt/qb/baumgartner/sgutwein84/phantom_results"

    for phantom in phantoms:

        phantom_dir = os.path.join(root_dir, phantom)
        models = [x for x in os.listdir(phantom_dir) if not x.startswith(".") and not ".pdf" in x]

        phantom_info = phantom.split("_")[0].split("P")[1]
        slab_pos = int(phantom_info.split("T")[0])
        slab_thickness = int(phantom_info.split("T")[1])

        slice_ = int(100)
        depth = int(slab_pos + 0.5*slab_thickness)

        print(depth)

        predictions = []
        infos = []
        for model in models:
            infos.append(model.split("_")[0])
            model_dir = os.path.join(phantom_dir, model)
            target = torch.load(os.path.join(model_dir, "target.pt"))
            predictions.append(torch.load(os.path.join(model_dir, "prediction.pt")))

        target_x = target[depth, :, slice_]
        target_y = target[:, 256, slice_]
        target_z = target[depth, 256, :]

        predictions_x = []
        predictions_y = []
        predictions_z = []
        for prediction in predictions:
            predictions_x.append(prediction[depth, :, slice_])
            predictions_y.append(prediction[:, 256, slice_])
            predictions_z.append(prediction[depth, 256, :])

        colors = ['maroon', 'lightsalmon']

        fig, ax = plt.subplots(figsize=(7, 5))
        for num, p_x in enumerate(predictions_x):
            ax.plot(range(len(p_x)), p_x, color=colors[num])
        ax.plot(range(len(target_x)), target_x, color="black")
        plt.legend([*infos, "target"])
        ax.set_xlabel("pixels")
        ax.set_ylabel("output value")
        latex_plot(fig, 14, 7.875, os.path.join(phantom_dir, f"{phantom}_x.pdf"))

        fig, ax = plt.subplots(figsize=(7, 5))
        for num, p_y in enumerate(predictions_y):
            ax.plot(range(len(p_y)), p_y, color=colors[num])
        ax.plot(range(len(target_y)), target_y, color="black")
        plt.legend([*infos, "target"])
        ax.set_xlabel("pixels")
        ax.set_ylabel("output value")
        latex_plot(fig, 14, 7.875, os.path.join(phantom_dir, f"{phantom}_y.pdf"))

        fig, ax = plt.subplots(figsize=(7, 5))
        for num, p_z in enumerate(predictions_z):
            ax.plot(range(len(p_z)), p_z, color=colors[num])
        ax.plot(range(len(target_z)), target_z, color="black")
        plt.legend([*infos, "target"])
        ax.set_xlabel("pixels")
        ax.set_ylabel("output value")
        latex_plot(fig, 14, 7.875, os.path.join(phantom_dir, f"{phantom}_z.pdf"))


if __name__ == "__main__":
    main()

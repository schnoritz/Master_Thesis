import os
import torch
import matplotlib.pyplot as plt


def main():

    phantoms = ["phantomP50T200_10x10", "phantomP100T200_10x10", "phantomP150T200_10x10", "phantomP200T200_10x10", "phantomP250T200_10x10", "phantomP300T200_10x10"]
    root_dir = "/Users/simongutwein/Studium/Masterarbeit/phantom_results"

    for phantom in phantoms:

        phantom_dir = os.path.join(root_dir, phantom)
        models = [x for x in os.listdir(phantom_dir) if not x.startswith(".")]

        phantom_info = phantom.split("_")[0].split("P")[1]
        slab_pos = int(phantom_info.split("T")[0])
        slab_thickness = int(phantom_info.split("T")[1])

        slice_ = int(100)
        depth = int(slab_pos + 0.5*slab_thickness)

        print(depth)

        predictions = []
        for model in models:

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

        fig, ax = plt.subplots()
        for p_x in predictions_x:
            ax.plot(range(len(p_x)), p_x)
        ax.plot(range(len(target_x)), target_x)
        plt.savefig(os.path.join(phantom_dir, "x.png"))

        fig, ax = plt.subplots()
        for p_y in predictions_y:
            ax.plot(range(len(p_y)), p_y)
        ax.plot(range(len(target_y)), target_y)
        plt.savefig(os.path.join(phantom_dir, "y.png"))

        fig, ax = plt.subplots()
        for p_z in predictions_z:
            ax.plot(range(len(p_z)), p_z)
        ax.plot(range(len(target_z)), target_z)
        plt.savefig(os.path.join(phantom_dir, "z.png"))


if __name__ == "__main__":
    main()

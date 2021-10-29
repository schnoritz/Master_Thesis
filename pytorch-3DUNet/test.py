from operator import indexOf
import torch
from entire_volume_prediction import predict_volume
from model import Dose3DUNET
from test_model import load_model, analyse_gamma, save_data
import os
from model import Dose3DUNET

if __name__ == "__main__":

    model = Dose3DUNET(2, 1)

    batch = torch.randn((1, 2, 32, 32, 32))
    input = torch.randn((1, 2, 512, 512, 16))
    print(batch.shape, input.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    pred = model(batch)
    print(pred.shape)

# model_paths = ["/mnt/qb/baumgartner/sgutwein84/save/mixed_trained/UNET_1183.pt", "/mnt/qb/baumgartner/sgutwein84/save/prostate_trained/UNET_2234.pt"]
# phantoms = ["phantomP100T200_10x10", "phantomP200T200_10x10", "phantomP300T200_10x10"]

# for phantom in phantoms:
#     for model_path in model_paths:

#         model_name = "_".join(model_path.split("/")[-2:])
#         data_dir = f"/mnt/qb/baumgartner/sgutwein84/training/training_phantom/{phantom}"
#         save_path = "/mnt/qb/baumgartner/sgutwein84/phantom_results"
#         if not os.path.isdir(save_path):
#             os.mkdir(save_path)

#         test_case = data_dir.split("/")[-1]

#         input = torch.load(os.path.join(data_dir, "training_data.pt"))
#         target = torch.squeeze(torch.load(os.path.join(data_dir, "target_data.pt")))

#         model, device = load_model(model_path)
#         prediction = predict_volume(input, model, device, shift=1)

#         print(prediction.shape, target.shape)
#         px_sp = (1.17185, 1.17185, 3)

#         save_path = save_data(save_path, test_case, model_name, prediction, target)
#         analyse_gamma(target, prediction, px_sp, save_path, lower_cutoff=10)

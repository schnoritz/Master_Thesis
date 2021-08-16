import torch
import numpy as np
import pymedphys
from pynvml import *
import nibabel as nib
from model import Dose3DUNET
from entire_volume_prediction import predict_volume
import random


def gamma_sample(unet, device, segment, segment_dir, uq, lower_cutoff=10, local_gamma=False, partial_sample=2000, gamma_percentage=3, gamma_distance=3, volume_return=False):

    target_dose = torch.load(
        f"{segment_dir}{segment}/target_data.pt")
    target_dose = target_dose.squeeze()

    masks = torch.load(
        f"{segment_dir}{segment}/training_data.pt")

    torch.cuda.empty_cache()

    predicted_dose = predict_volume(masks, unet, device, uq)

    print(predicted_dose.shape, target_dose.shape)

    gamma_options = {
        'dose_percent_threshold': gamma_percentage,
        'distance_mm_threshold': gamma_distance,
        'lower_percent_dose_cutoff': lower_cutoff,
        'interp_fraction': 5,  # Should be 10 or more for more accurate results
        'max_gamma': 1.1,
        'quiet': True,
        'local_gamma': local_gamma,
        'random_subset': partial_sample
    }

    coords = (np.arange(0, 1.17*target_dose.shape[0], 1.17), np.arange(
        0, 1.17*target_dose.shape[1], 1.17), np.arange(0, 3*target_dose.shape[2], 3))

    gamma_val = pymedphys.gamma(
        coords, np.array(target_dose),
        coords, np.array(predicted_dose),
        **gamma_options)

    dat = ~np.isnan(gamma_val)
    dat2 = ~np.isnan(gamma_val[gamma_val <= 1])
    all = np.count_nonzero(dat)
    true = np.count_nonzero(dat2)

    unet.train()

    if volume_return:
        return np.round((true/all)*100, 2), predicted_dose, target_dose
    return np.round((true/all)*100, 2)


if __name__ == "__main__":

    uq = False

    dir = "/mnt/qb/baumgartner/sgutwein84/training_nodes/"
    model_path = "/mnt/qb/baumgartner/sgutwein84/save/bs32_ps32_5/UNET_896.pt"
    save_path = "/home/baumgartner/sgutwein84/container/test"

    segments = [x for x in os.listdir(dir) if not x.startswith(".")]
    segments = random.sample(segments, 10)
    #segments = ["p5_21", "p5_29", "p5_44", "p5_10", "p5_16", "p7_29", "p7_39", "p7_53", "p7_10", "p7_21", "p8_31", "p8_41", "p8_51", "p8_4", "p8_16", "p9_22", "p9_27", "p9_34", "p9_38", "p9_13"]
    gammas = []

    print(segments)

    for segment in segments:

        if os.path.isfile(f"{dir}{segment}/training_data.pt"):
            mask = torch.load(f"{dir}{segment}/training_data.pt")
            binary = mask[0]

            device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
            model_checkpoint = torch.load(model_path, map_location=device)
            model = Dose3DUNET(UQ=uq)
            model.load_state_dict(model_checkpoint['model_state_dict'])
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)
            gamma, pred, target = gamma_sample(model, device, segment, dir, uq=uq, volume_return=True, gamma_distance=3, gamma_percentage=3, lower_cutoff=10)
            gammas.append(gamma)
            print(f"Gamma-Passrate for segment {segment}: {gamma}%")

            dat = nib.Nifti1Image(np.array(pred), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{segment}_pred.nii.gz")
            dat = nib.Nifti1Image(np.array(target), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{segment}_target.nii.gz")
            dat = nib.Nifti1Image(np.array(target)-np.array(pred), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{segment}_diff.nii.gz")
            dat = nib.Nifti1Image(np.array(binary), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{segment}_binary.nii.gz")

        else:
            print("Missing Data for ", segment)

    gammas = np.array(gammas)
    mean_gammas = gammas.mean()
    std_gammas = gammas.std()

    print(mean_gammas, std_gammas)

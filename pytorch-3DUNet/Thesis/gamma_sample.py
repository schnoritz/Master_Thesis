import torch
import numpy as np
import pymedphys
from pynvml import *
import nibabel as nib
from model import Dose3DUNET
from entire_volume_prediction import predict_volume
import random


def gamma_sample(unet, device, segment, segment_dir, lower_cutoff=10, local_gamma=False, partial_sample=2000, gamma_percentage=3, gamma_distance=3, volume_return=False, shift=8):

    target_dose = torch.load(
        f"{segment_dir}{segment}/target_data.pt")
    target_dose = target_dose.squeeze()

    masks = torch.load(
        f"{segment_dir}{segment}/training_data.pt")

    torch.cuda.empty_cache()
    predicted_dose = predict_volume(masks, unet, device, shift=shift)

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

    dir = "/mnt/qb/baumgartner/sgutwein84/training/training_prostate/"
    model_path = "/mnt/qb/baumgartner/sgutwein84/save/bs128_ps32_lr5_2108241247/UNET_799.pt"
    save_path = "/home/baumgartner/sgutwein84/container/test"

    segments = [x for x in os.listdir(dir) if not x.startswith(".")]
    segments = random.sample(segments, 5)
    #segments = ['p12_7', 'p38_17', 'p32_12', 'p30_32', 'p39_18']
    segments = ["p26_17"]
    gammas = []

    print(segments)

    for segment in segments:

        if os.path.isfile(f"{dir}{segment}/training_data.pt"):
            mask = torch.load(f"{dir}{segment}/training_data.pt")
            binary = mask[0]

            device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
            model_checkpoint = torch.load(model_path, map_location=device)
            model = Dose3DUNET()
            model.load_state_dict(model_checkpoint['model_state_dict'])
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)
            gamma, pred, target = gamma_sample(model, device, segment, dir, volume_return=True, gamma_distance=3, gamma_percentage=3, lower_cutoff=10, partial_sample=10000, shift=16)
            diff = target-pred
            diff = diff[target > 0.1*target.max()]
            gammas.append(gamma)
            print(f"Gamma-Passrate for segment {segment}: {gamma}%")

            if not os.path.isdir(f"{save_path}/{segment}"):
                os.mkdir(f"{save_path}/{segment}")

            dat = nib.Nifti1Image(np.array(pred), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{segment}/{segment}_pred.nii.gz")
            dat = nib.Nifti1Image(np.array(target), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{segment}/{segment}_target.nii.gz")
            dat = nib.Nifti1Image(np.array(target)-np.array(pred), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{segment}/{segment}_diff.nii.gz")
            dat = nib.Nifti1Image(np.array(binary), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{segment}/{segment}_binary.nii.gz")

        else:
            print("Missing Data for ", segment)

    gammas = np.array(gammas)
    mean_gammas = gammas.mean()
    std_gammas = gammas.std()
    median_gammas = np.median(gammas)

    print(np.round(mean_gammas, 2), "%", np.round(median_gammas, 2), "%", np.round(std_gammas, 2))

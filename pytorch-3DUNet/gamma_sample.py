import torch
import numpy as np
import pymedphys
from pynvml import *
from model import Dose3DUNET


def gamma_sample(unet, device, segment, segment_dir, lower_cutoff=20, partial_sample=20000, gamma_percentage=3, gamma_distance=3):

    target_dose = torch.load(
        f"{segment_dir}{segment}/target_data.pt")
    target_dose = target_dose.squeeze()

    masks = torch.load(
        f"{segment_dir}{segment}/training_data.pt")
    masks = torch.unsqueeze(masks, 0)

    torch.cuda.empty_cache()

    with torch.no_grad():
        unet.eval()
        preds = []
        ps = 16
        for i in range(0, masks.shape[4], ps):
            mask = masks[0, :, :, :, i:i+ps]
            if mask.shape[3] < ps:
                num = int(ps-mask.shape[3])
                added = torch.zeros(
                    mask.shape[0], mask.shape[1], mask.shape[2], ps-mask.shape[3])
                mask = torch.cat((mask, added), 3)
            mask = mask.unsqueeze(0)
            mask = mask.to(device)

            pred = unet(mask)
            torch.cuda.empty_cache()

            preds.append(pred.cpu().detach().squeeze())

        end = torch.cat(preds, 2)
        end = end[:, :, :(-num)]

        gamma_options = {
            'dose_percent_threshold': gamma_percentage,
            'distance_mm_threshold': gamma_distance,
            'lower_percent_dose_cutoff': lower_cutoff,
            'interp_fraction': 5,  # Should be 10 or more for more accurate results
            'max_gamma': 1.1,
            'ram_available': 2**37,
            'quiet': True,
            'local_gamma': False,
            'random_subset': partial_sample
        }

        coords = (np.arange(0, 1.17*target_dose.shape[0], 1.17), np.arange(
            0, 1.17*target_dose.shape[1], 1.17), np.arange(0, 3*target_dose.shape[2], 3))

        gamma_val = pymedphys.gamma(
            coords, np.array(target_dose),
            coords, np.array(end),
            **gamma_options)

        dat = ~np.isnan(gamma_val)
        dat2 = ~np.isnan(gamma_val[gamma_val <= 1])
        all = np.count_nonzero(dat)
        true = np.count_nonzero(dat2)

        return np.round((true/all)*100, 2)


if __name__ == "__main__":

    segment = "p0_0"
    device = torch.device(
        "cuda") if torch.cuda.is_available else torch.device("cpu")
    model_checkpoint = torch.load(
        "/home/baumgartner/sgutwein84/container/pytorch-3DUNet/experiments/bs16_ps32/UNET_88.pt", map_location="cpu")
    model = Dose3DUNET()
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    gamma_sample(model, device, segment)

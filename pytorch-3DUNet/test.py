from torch.random import get_rng_state
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
from dataset import setup_loaders
from pprint import pprint
import random
from time import time
import dataqueue
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import pymedphys
import sys
from torch.nn.functional import interpolate
import gamma_sample
import nibabel as nib
from pydicom import dcmread, uid
from torch.nn.functional import interpolate
import argparse
from model_UQ import Dose3DUNET


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

        return np.round((true/all)*100, 2), end, target_dose, masks.squeeze()


if __name__ == "__main__":

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model_checkpoint = torch.load(
        "/home/baumgartner/sgutwein84/container/pytorch-3DUNet/experiments/bs32_ps32_corrected/UNET_267.pt", map_location="cpu")

    segment = torch.load(
        "/home/baumgartner/sgutwein84/container/training_prostate/p0_0/training_data.pt")
    print("loaded")
    segment = torch.nn.functional.pad(
        segment, (0, 0, -240, -240, -240, -240))
    segment = segment[:, :, 20:52]
    print(segment.shape)
    segment.to(device)

    model = Dose3DUNET()
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model = torch.nn.DataParallel(model)
    model.to(device)

    pred = model(segment)
    print(pred.shape)
    (mean, variance, sample_variance) = model.get_uncertainty(segment, T=5)
    print(variance.shape)

    for i in range(variance.shape[4]):
        plt.imshow(variance[0, 0, :, :, i].detach().numpy())
        plt.savefig(
            f"/Users/simongutwein/Studium/Masterarbeit/test/{i}.png")
        plt.show()

    # parser = argparse.ArgumentParser(description="Clean dosxyznrc folder.")

    # parser.add_argument(
    #     "segment",
    #     type=str,
    #     metavar="",
    #     help="",
    # )

    # args = parser.parse_args()

    # if not os.path.isdir(f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}"):
    #     os.makedirs(
    #         f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}")

    # segment_dir = "/home/baumgartner/sgutwein84/container/training_prostate/"

    # device = torch.device(
    #     "cuda") if torch.cuda.is_available else torch.device("cpu")

    # model_checkpoint = torch.load(
    #     "/home/baumgartner/sgutwein84/container/pytorch-3DUNet/experiments/bs32_ps32_corrected/UNET_267.pt", map_location="cpu")

    # model = Dose3DUNET()
    # model.load_state_dict(model_checkpoint['model_state_dict'])
    # model = torch.nn.DataParallel(model)
    # model.to(device)

    # gamma, pred, target, masks = gamma_sample(
    #     model, device, args.segment, segment_dir=segment_dir)
    # diff = pred-target
    # print(f"Gamma Value for {args.segment}: {gamma}")
    # binary = np.array(masks[0, :, :, :]).squeeze()
    # ct = np.array(masks[1, :, :, :]).squeeze()

    # img = nib.Nifti1Image(np.array(pred), np.eye(4))

    # img.header.get_xyzt_units()
    # img.to_filename(
    #     f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/predicted_dose.nii.gz")

    # img = nib.Nifti1Image(binary, np.eye(4))

    # img.header.get_xyzt_units()
    # img.to_filename(
    #     f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/binary_mask.nii.gz")

    # img = nib.Nifti1Image(np.array(target), np.eye(4))

    # img.header.get_xyzt_units()
    # img.to_filename(
    #     f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/target_dose.nii.gz")

    # img = nib.Nifti1Image(np.array(diff), np.eye(4))

    # img.header.get_xyzt_units()
    # img.to_filename(
    #     f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/difference.nii.gz")

    # img = nib.Nifti1Image(np.array(ct), np.eye(4))

    # img.header.get_xyzt_units()
    # img.to_filename(
    #     f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/ct.nii.gz")

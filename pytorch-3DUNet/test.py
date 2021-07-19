from torch.random import get_rng_state
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
from model import Dose3DUNET
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

    # data_path = "/home/baumgartner/sgutwein84/container/training_data_prostate/"

    # segments = [x.split("_")[0]
    #             for x in os.listdir(data_path) if not x.startswith(".")]
    # patients = list(dict.fromkeys(segments))
    # patients = [x + "_0" for x in patients]
    # print(patients)

    # for pat in patients:

    #     masks = torch.load(
    #         "/home/baumgartner/sgutwein84/container/training_data_prostate/" + pat + "/training_data.pt")
    #     target = torch.load(
    #         "/home/baumgartner/sgutwein84/container/training_data_prostate/" + pat + "/target_data.pt")

    #     plt.imshow(masks[1, :, 256, :])
    #     plt.savefig(
    #         f"/home/baumgartner/sgutwein84/container/predictions/test/{pat}.png")

    # masks = torch.load(data_path + "training_data.pt")
    # target = torch.load(data_path + "target_data.pt")

    # fig, ax = plt.subplots(1, 6, figsize=(60, 10))
    # ax[0].imshow(masks[0, :, :, 38])
    # ax[1].imshow(masks[1, :, :, 38])
    # ax[2].imshow(masks[2, :, :, 38])
    # ax[3].imshow(masks[3, :, :, 38])
    # ax[4].imshow(masks[4, :, :, 38])
    # ax[5].imshow(target[0, :, :, 38])
    # for i, title in enumerate(["Binary Beam Mask", "CT Volume", "Radiological Depth", "Beam Line Distance", "Source Distance", "Target Dose"]):
    #     ax[i].set_title(title, fontweight="bold", size=40)
    #     plt.setp(ax[i].get_xticklabels(), visible=False)
    #     plt.setp(ax[i].get_yticklabels(), visible=False)

    # plt.savefig("/home/baumgartner/sgutwein84/container/masks.png")

    # masks = torch.load(
    #     "/Users/simongutwein/home/baumgartner/sgutwein84/container/prostate_training_data/p3_3/training_data.pt")
    # masks = masks.squeeze()
    # print(masks.shape)
    # for i in masks[1]:
    #     plt.imshow(i[:, :, i])
    #     plt.show()

    # for seg in ["p36", "p37"]:
    #     files = [f"/Users/simongutwein/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/{seg}/" + x for x in os.listdir(
    #         f"/Users/simongutwein/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/{seg}") if not x.startswith(".")]

    #     cts = []
    #     for i in files:
    #         dat = dcmread(i, force=True)
    #         dat.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
    #         cts.append({
    #             'image': dat.pixel_array,
    #             'slice_location': dat.SliceLocation
    #         })

    #     ct_dict = sorted(cts, key=lambda d: d['slice_location'])

    #     images = []
    #     for i in range(len(ct_dict)):
    #         images.append(ct_dict[i]["image"])

    #     images = np.stack(images, axis=2)

    #     plt.imshow(images[:, 256, :])
    #     plt.show()

    parser = argparse.ArgumentParser(description="Clean dosxyznrc folder.")

    parser.add_argument(
        "segment",
        type=str,
        metavar="",
        help="",
    )

    args = parser.parse_args()

    if not os.path.isdir(f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}"):
        os.makedirs(
            f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}")

    segment_dir = "/home/baumgartner/sgutwein84/container/training_prostate/"

    device = torch.device(
        "cuda") if torch.cuda.is_available else torch.device("cpu")

    model_checkpoint = torch.load(
        "/home/baumgartner/sgutwein84/container/pytorch-3DUNet/experiments/bs32_ps32_corrected/UNET_267.pt", map_location="cpu")

    model = Dose3DUNET()
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model = torch.nn.DataParallel(model)
    model.to(device)

    gamma, pred, target, masks = gamma_sample(
        model, device, args.segment, segment_dir=segment_dir)
    diff = pred-target
    print(f"Gamma Value for {args.segment}: {gamma}")
    binary = np.array(masks[0, :, :, :]).squeeze()
    ct = np.array(masks[1, :, :, :]).squeeze()

    img = nib.Nifti1Image(np.array(pred), np.eye(4))

    img.header.get_xyzt_units()
    img.to_filename(
        f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/predicted_dose.nii.gz")

    img = nib.Nifti1Image(binary, np.eye(4))

    img.header.get_xyzt_units()
    img.to_filename(
        f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/binary_mask.nii.gz")

    img = nib.Nifti1Image(np.array(target), np.eye(4))

    img.header.get_xyzt_units()
    img.to_filename(
        f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/target_dose.nii.gz")

    img = nib.Nifti1Image(np.array(diff), np.eye(4))

    img.header.get_xyzt_units()
    img.to_filename(
        f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/difference.nii.gz")

    img = nib.Nifti1Image(np.array(ct), np.eye(4))

    img.header.get_xyzt_units()
    img.to_filename(
        f"/home/baumgartner/sgutwein84/container/predictions/{args.segment}/ct.nii.gz")

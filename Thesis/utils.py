import torch.nn as nn
import torch
import numpy as np
import os
from pydicom import dcmread, uid
import matplotlib.pyplot as plt
from pprint import pprint


def define_iso_center(egsinp, ct_path):

    iso_center = np.array(egsinp.split(",")[2:5], dtype=float)

    zero_point, slice_spacing, pixel_spacing = extract_phantom_data(ct_path, pixel_spacing=True)

    px_sp = np.array([pixel_spacing[0], pixel_spacing[1], slice_spacing]).astype(float)

    # wird hier geändert, da mein Koordinatensystem immer 3mm slice spacing hat
    # und das Isocentrum auf diesem Koordinatensystem berechnet wird
    px_sp[2] = 3
    iso_slice = np.zeros((3,))
    for i in range(3):
        iso_slice[i] = (iso_center[i] - zero_point[i]) / (px_sp[i]/10)

    iso_slice = np.round(iso_slice).astype(int)

    assert iso_slice[0] < 512, "isocenter lies outside of Volume"
    assert iso_slice[1] < 512, "isocenter lies outside of Volume"

    iso_slice[0], iso_slice[1] = iso_slice[1], iso_slice[0]

    return iso_slice, px_sp


def define_origin_position(egsinp, iso_center, px_sp, SID=1435):

    angle = np.radians(float(egsinp.split(",")[6]) - 270)

    pixel_SID = int(SID / px_sp[0])
    origin = np.array(
        [
            iso_center[0] - np.cos(angle) * pixel_SID,
            iso_center[1] + np.sin(angle) * pixel_SID,
            iso_center[2],
        ]
    ).astype("float")

    return origin


def get_angle(egsinp, radians=False):

    if radians == False:
        return float(egsinp.split(",")[6]) - 270
    else:
        return np.radians(float(egsinp.split(",")[6]) - 270)


def extract_phantom_data(ct_path, pixel_spacing=False):

    if ct_path[-1] != "/":
        ct_path += "/"

    ct_files = [
        x for x in os.listdir(ct_path) if "dcm" in x.lower() and not x.startswith(".")
    ]

    ct_dict = []
    for file in ct_files:
        with dcmread(ct_path + file, force=True) as dcmin:
            ct_dict.append(
                {"filename": file, "slice_location": dcmin.SliceLocation})

    ct_dict.sort(key=lambda t: t["slice_location"])
    first_file = dcmread(ct_path + ct_dict[0]["filename"], force=True)
    zero_point = np.array(first_file.ImagePositionPatient).astype(float)

    if pixel_spacing:
        with dcmread(ct_path + file, force=True) as dcmin:
            slice_spacing = int(dcmin.SliceThickness)
            pixel_spacing = dcmin.PixelSpacing

    if pixel_spacing:
        return zero_point/10, slice_spacing, pixel_spacing

    return zero_point/10


def get_angles(data_dir):
    angles = []

    segments = [x for x in os.listdir(data_dir) if "_" in x and not x.startswith(".")]

    for seg in segments:
        path = os.path.join(data_dir, seg, f"{seg}.egsinp")

        with open(path) as fin:
            lines = fin.readlines()
        angles.append(float(lines[5].split(",")[6])-270)

    return angles


def get_density_angles(angles):

    angles = np.array(angles)
    bins = np.array(np.linspace(0, 360, 720, endpoint=False))
    occ = np.zeros_like(bins)

    for num, _bin in enumerate(bins):
        occ[num] = ((_bin <= angles) & (angles < (_bin+(1/2)))).sum()

    return occ, bins


def create_polar_plot(angles):

    occ, bins = get_density_angles(angles)
    occ = occ/occ.sum()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(bins, occ)
    ax.set_theta_zero_location("N")
    ax.set_rgrids(np.round(np.linspace(0.5*occ.max(), occ.max(), 5), 2), angle=45)


def check_field_sizes(data_dir):

    sizes = []

    segments = [x for x in os.listdir(data_dir) if "_" in x and not x.startswith(".")]

    for seg in segments:
        path = os.path.join(data_dir, seg, f"beam_config_{seg}.txt")

        with open(path) as fin:
            lines = fin.readlines()

        size = 0
        for leaf_pair in lines:
            leaf_pair = leaf_pair.split(",")
            size += abs(float(leaf_pair[0]) - float(leaf_pair[1]))

        sizes.append(size)

    return np.array(sizes)


if __name__ == "__main__":

    path = "/Users/simongutwein/Studium/Masterarbeit/test/p0"
    egsinp = open(
        "/home/baumgartner/sgutwein84/container/output_prostate/p0_0/p0_0.egsinp"
    ).readlines()[5]

    iso_center = define_iso_center(egsinp, path)
    if path[-1] != "/":
        path += "/"

    ct_files = [
        x for x in os.listdir(path) if "dcm" in x.lower() and not x.startswith(".")
    ]

    ct_dict = []
    for file in ct_files:
        with dcmread(path + file, force=True) as dcmin:
            ct_dict.append(
                {"filename": file, "slice_location": dcmin.SliceLocation})

    ct_dict.sort(key=lambda t: t["slice_location"])

    ct = np.zeros((512, 512, len(ct_dict)))
    for num, file in enumerate(ct_dict):
        with dcmread(path + file["filename"], force=True) as dcmin:
            dcmin.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
            ct[:, :, num] = dcmin.pixel_array

    for i in range(ct.shape[2]):
        print(i)
        if i == iso_center[2]:
            plt.imshow(ct[:, :, i])
            plt.scatter(iso_center[0], iso_center[1], s=20, color='red')
            plt.show()
        else:
            plt.imshow(ct[:, :, i])
            plt.show()


def plot_patches(patches, target, idx):

    fig, axs = plt.subplots(1, 6, figsize=(24, 4))
    img = 0
    for b in patches:
        i = 0
        img += 1
        for p in b:
            axs[i].imshow(p[:, :, p.shape[2]//2])
            i += 1

        axs[5].imshow(target[img-1, 0, :, :, target.shape[2]//2])
        axs[5].set_title("Target Patch")
        for i in range(5):
            axs[i].set_title("Training Patch")

        name = "/home/baumgartner/sgutwein84/training_data/output_images/image" + \
            str(img) + str(idx) + ".png"

        if os.path.isfile(name):
            os.remove(name)

        plt.savefig(name)

    plt.close()


def define_calculation_device(use_gpu):

    if use_gpu:
        if torch.cuda.is_available():
            print("Using CUDA!")
            device = torch.device('cuda')
        else:
            print("Using CPU!")
            device = torch.device('cpu')
    else:
        print("Using CPU!")
        device = torch.device('cpu')

    if device.type == 'cuda':
        print(f"Device: {torch.cuda.get_device_name(0)}")

    return device


def save_model(model, optimizer, train_loss, validation_loss, save_dir, patches, save, epochs, generation, training_parameter):

    if torch.cuda.device_count() > 1:
        torch.save({
            'patches': patches,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'epochs': epochs,
            'model_generation': generation,
            'training_parameter': training_parameter

        }, save_dir + f"UNET_{generation}.pt")

    else:
        torch.save({
            'patches': patches,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'epochs': epochs,
            'model_generation': generation,
            'training_parameter': training_parameter

        }, save_dir + f"UNET_{generation}.pt")

    if type(save) == int:
        if os.path.isfile(save_dir + f"UNET_{save}.pt"):
            os.remove(save_dir + f"UNET_{save}.pt")


def check_improvement(epochs, top_k=5):

    curr_epoch = epochs[-1]
    epochs = sorted(epochs, key=lambda k: k['validation_loss'])
    if epochs.index(curr_epoch) < top_k:
        if len(epochs) > top_k:
            return epochs[top_k]['model_generation']
        else:
            return True
    else:
        return False


def get_training_data(train, target, device):

    train = train.float()
    target = target.float()

    if device.type == 'cuda':

        train = train.to(device)
        target = target.to(device)

    return train, target


def optimizer_to_device(optimizer, device):

    for check in optimizer.state.values():
        for k, v in check.items():
            if torch.is_tensor(v):
                check[k] = v.to(device)

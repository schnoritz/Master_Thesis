from matplotlib.pyplot import figimage
from numpy.lib.type_check import imag
from pydicom import dcmread
import numpy as np
from itertools import zip_longest
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def create_target_pred_dvh(struc_data: dict, target_dose: np.ndarray, prediction_dose: np.ndarray) -> dict:

    max_dose = target_dose.max()
    dvh_x = np.linspace(0, max_dose, 1000, endpoint=True)

    struc_data = create_dvh(struc_data, target_dose, dvh_x, "target")
    struc_data = create_dvh(struc_data, prediction_dose, dvh_x, "prediction")

    return struc_data


def create_dvh(struc_data: dict, dose_array: np.ndarray, dvh_x: list, attribute_name: str) -> dict:

    for struc in tqdm(struc_data):
        relevant_part = dose_array[struc["binary_mask"]].copy()
        num_vox = relevant_part.size

        if num_vox != 0:
            dvh_data = []
            for threshold in dvh_x:
                dvh_data.append((relevant_part[relevant_part >= threshold].astype(bool).size)/num_vox)

            struc[attribute_name + "_dvh"] = dvh_data
            struc["dvh_x"] = dvh_x
            struc[attribute_name + "_D2"] = dvh_x[find_nearest(dvh_data, 0.02)]
            struc[attribute_name + "_Dmean"] = relevant_part.mean()
            struc[attribute_name + "_D98"] = dvh_x[find_nearest(dvh_data, 0.98)]
            struc[attribute_name + "_D"] = dvh_x[find_nearest(dvh_data, 0.98)]

        else:

            struc[attribute_name + "_dvh"] = None
            struc["dvh_x"] = None
            struc[attribute_name + "_D2"] = None
            struc[attribute_name + "_Dmean"] = None
            struc[attribute_name + "_D98"] = None

    return struc_data


def create_binary(structure_data: dict, volume_shape: list) -> dict:

    for struc in structure_data:
        binary = np.zeros(volume_shape)
        for slice_ in struc["contour_data"]:
            poly = slice_[:, :2]
            img = binary[:, :, slice_[0, 2]].copy()
            binary[:, :, slice_[0, 2]] = cv2.drawContours(img, np.int32([poly]), -1, color=1, thickness=cv2.FILLED)
        struc["binary_mask"] = binary.astype(bool)

    return structure_data


def read_structures(strucure_file_path: str, image_origin: list, px_sp: list) -> dict:

    image_origin = np.array(image_origin)
    px_sp = np.array(px_sp)

    structures = []
    structures_data = []
    with dcmread(strucure_file_path, force=True) as dcm_in:
        for struc in dcm_in.StructureSetROISequence:
            structures.append({
                "struc_name": struc.ROIName,
                "struc_number": struc.ROINumber
            })

        for struc in dcm_in.ROIContourSequence:
            contour_data = []
            for data in struc.ContourSequence:
                contour_data.append(((np.array(data.ContourData).reshape((-1, 3))-image_origin)/px_sp).astype(int))
            structures_data.append({
                "struc_number": struc.ReferencedROINumber,
                "contour_data": contour_data
            })

    struc_data = [{**u, **v} for u, v in zip_longest(structures, structures_data, fillvalue={})]
    return struc_data


def analyse_structures(structure_file: str, origin: list, px_sp: list, volume_shape: list, target_dose: np.ndarray, prediction_dose: np.ndarray, diff=False) -> dict:

    structures = read_structures(structure_file, origin, px_sp)
    structures = create_binary(structures, volume_shape)
    structures = create_target_pred_dvh(structures, target_dose, prediction_dose)
    if diff:
        structures = dvh_diff(structures, window_size=10)

    return structures


def plot_dvh(structures: dict, path: str, diff=False) -> None:

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Palatino"
    plt.rcParams["font.size"] = 11

    colors = ['red', 'orange', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black', 'gold']

    legend = []

    plot_struc = ["ctv", "ptv", "bladder", "rectum", "femur", "penile", "uret"]

    if diff:
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))

        for num, struc in enumerate(structures):
            if not "z" in struc["struc_name"].lower():
                ax[0].plot(struc["dvh_x"], struc["target_dvh"], color=colors[num])
                ax[0].plot(struc["dvh_x"], struc["prediction_dvh"], "--", color=colors[num])
                ax[1].plot(struc["dvh_x"], struc["dvh_diff_mean"], color=colors[num])
                legend.append(struc["struc_name"])
                legend.append(struc["struc_name"] + " pred")

        ax[0].legend(legend, bbox_to_anchor=(2.3, 0.5), loc='center left')
        fig.subplots_adjust(right=0.7)
        ax[1].set_ylim([-0.5, 0.5])
        ax[1].grid(color='lightgrey', linestyle='-', linewidth=1)
        plt.savefig(path)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        num = 0
        for struc in structures:
            if not "z" in struc["struc_name"].lower() and not struc["dvh_x"] is None:
                for s in plot_struc:
                    if s in struc["struc_name"].lower():
                        ax.plot(struc["dvh_x"], np.array(struc["target_dvh"])*100, color=colors[num])
                        ax.plot(struc["dvh_x"], np.array(struc["prediction_dvh"])*100, "--", color=colors[num])
                        legend.append(struc["struc_name"])
                        legend.append(struc["struc_name"] + " pred")
                        num += 1

        ax.legend(legend, bbox_to_anchor=(1.05, 0.5), loc='center left')
        fig.subplots_adjust(right=0.75)
        ax.set_xlabel("Dose \Gy")
        ax.set_ylabel("Percentage Volume /%")
    cm = 1/2.54
    fig.set_size_inches(14*cm, 10*cm, forward=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')


def dvh_diff(structures: dict, window_size: int) -> dict:

    for struc in structures:
        if not "z" in struc["struc_name"].lower():

            target = np.array(struc["target_dvh"])
            prediction = np.array(struc["prediction_dvh"])

            running_mean = []
            for _ in range(np.floor(window_size/2).astype(int)):
                running_mean.append(np.nan)

            for idx in range(0, len(target)-window_size):
                running_mean.append((target[idx:idx+window_size]-prediction[idx:idx+window_size]).mean())

            for _ in range(np.ceil(window_size/2).astype(int)):
                running_mean.append(np.nan)

        struc["dvh_diff_mean"] = np.array(running_mean)

    return structures


def dvh_values_to_xlsx(structures: dict, path: str) -> None:

    struc_df = pd.DataFrame(structures)
    sub_df = struc_df[["struc_name", "target_D98", "prediction_D98",  "target_D2", "prediction_D2", "target_Dmean", "prediction_Dmean"]]
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    sub_df.to_excel(writer, index=False, sheet_name='DVH_analysis')
    workbook = writer.book
    worksheet = writer.sheets['DVH_analysis']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('B:H', None, format1)  # Adds formatting to column C
    writer.save()


def main():

    import torch
    import os

    patients = ["pt0"]
    models = ["prostate_trained_UNET_2234.pt", "mixed_trained_UNET_1183.pt"]
    for patient in patients:
        for model in models:
            print(patient, model)
            root_dir = "/Users/simongutwein/Studium/Masterarbeit/preds/" + patient
            data_dir = os.path.join(root_dir, model)

            target_dose = np.array(torch.load(os.path.join(data_dir, f"target.pt")))*5.811/100*39
            prediction_dose = np.array(torch.load(os.path.join(data_dir, f"prediction.pt")))*5.811/100*39
            print(target_dose.shape, prediction_dose.shape)

            ct_file = os.path.join(root_dir, "cts/CT_image0.dcm")

            with dcmread(ct_file, force=True) as fin:
                origin = np.array(fin.ImagePositionPatient).astype(float)
                px_sp = fin.PixelSpacing
                px_sp.append(3)
                px_sp = np.array(px_sp).astype(float)
                print(origin, px_sp)

            structure_file = os.path.join(root_dir, f"{patient}_strctr.dcm")

            structures = analyse_structures(structure_file, origin, px_sp, target_dose.shape, target_dose, prediction_dose)

            plot_dvh(structures, os.path.join(data_dir, f"dvh_img.pdf"), diff=False)
            dvh_values_to_xlsx(structures, os.path.join(data_dir, f"dvh_data.xlsx"))


if __name__ == "__main__":

    main()

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
            struc[attribute_name + "_D98"] = dvh_x[find_nearest(dvh_data, 0.02)]
            struc[attribute_name + "_Dmean"] = relevant_part.mean()
            struc[attribute_name + "_D2"] = dvh_x[find_nearest(dvh_data, 0.98)]

        else:

            struc[attribute_name + "_dvh"] = None
            struc["dvh_x"] = None
            struc[attribute_name + "_D98"] = None
            struc[attribute_name + "_Dmean"] = None
            struc[attribute_name + "_D2"] = None

    return struc_data


def create_binary(structure_data: dict, volume_shape: list) -> dict:

    for struc in structure_data:
        binary = np.zeros(volume_shape)
        for slice_ in struc["contour_data"]:
            poly = slice_[:, :2]
            img = binary[:, :, slice_[0, 2]].copy()
            #binary[:, :, slice_[0, 2]] = cv2.fillPoly(img=img, pts=np.int32([poly]), color=1)
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

    colors = [
        '#F0F8FF', '#FAEBD7', '#FFEFDB', '#EEDFCC', '#CDC0B0', '#8B8378', '#00FFFF', '#7FFFD4', '#76EEC6', '#66CDAA', '#458B74', '#F0FFFF', '#E0EEEE', '#C1CDCD', '#838B8B', '#E3CF57', '#F5F5DC',
        '#FFE4C4', '#EED5B7', '#CDB79E', '#8B7D6B', '#000000', '#FFEBCD', '#0000FF', '#0000EE', '#0000CD', '#00008B', '#8A2BE2', '#9C661F', '#A52A2A', '#FF4040', '#EE3B3B', '#CD3333', '#8B2323',
        '#DEB887', '#FFD39B', '#EEC591', '#CDAA7D', '#8B7355', '#8A360F', '#8A3324', '#5F9EA0', '#98F5FF', '#8EE5EE', '#7AC5CD', '#53868B', '#FF6103', '#FF9912', '#ED9121', '#7FFF00', '#76EE00',
        '#66CD00', '#458B00', '#D2691E', '#FF7F24', '#EE7621', '#CD661D', '#8B4513', '#3D59AB', '#3D9140', '#808A87', '#FF7F50', '#FF7256', '#EE6A50', '#CD5B45', '#8B3E2F', '#6495ED', '#FFF8DC',
        '#EEE8CD', '#CDC8B1', '#8B8878', '#DC143C', '#00EEEE', '#00CDCD', '#008B8B', '#B8860B', '#FFB90F', '#EEAD0E', '#CD950C', '#8B6508', '#A9A9A9', '#006400', '#BDB76B', '#556B2F', '#CAFF70',
        '#BCEE68', '#A2CD5A', '#6E8B3D', '#FF8C00', '#FF7F00', '#EE7600', '#CD6600', '#8B4500', '#9932CC', '#BF3EFF', '#B23AEE', '#9A32CD', '#68228B', '#E9967A', '#8FBC8F', '#C1FFC1', '#B4EEB4',
        '#9BCD9B', '#698B69', '#483D8B', '#2F4F4F', '#97FFFF', '#8DEEEE', '#79CDCD', '#528B8B', '#00CED1', '#9400D3', '#FF1493', '#EE1289', '#CD1076', '#8B0A50', '#00BFFF', '#00B2EE', '#009ACD',
        '#00688B', '#696969', '#1E90FF', '#1C86EE', '#1874CD', '#104E8B', '#FCE6C9', '#00C957', '#B22222', '#FF3030', '#EE2C2C', '#CD2626', '#8B1A1A', '#FF7D40', '#FFFAF0', '#228B22', '#DCDCDC',
        '#F8F8FF', '#FFD700', '#EEC900', '#CDAD00', '#8B7500', '#DAA520', '#FFC125', '#EEB422', '#CD9B1D', '#8B6914', '#808080', '#030303', '#1A1A1A', '#1C1C1C', '#1F1F1F', '#212121', '#242424',
        '#262626', '#292929', '#2B2B2B', '#2E2E2E', '#303030', '#050505', '#333333', '#363636', '#383838', '#3B3B3B', '#3D3D3D', '#404040', '#424242', '#454545', '#474747', '#4A4A4A', '#080808',
        '#4D4D4D', '#4F4F4F', '#525252', '#545454', '#575757', '#595959', '#5C5C5C', '#5E5E5E', '#616161', '#636363', '#0A0A0A', '#666666', '#6B6B6B', '#6E6E6E', '#707070', '#737373', '#757575',
        '#787878', '#7A7A7A', '#7D7D7D', '#0D0D0D', '#7F7F7F', '#828282', '#858585', '#878787', '#8A8A8A', '#8C8C8C', '#8F8F8F', '#919191', '#949494', '#969696', '#0F0F0F', '#999999', '#9C9C9C',
        '#9E9E9E', '#A1A1A1', '#A3A3A3', '#A6A6A6', '#A8A8A8', '#ABABAB', '#ADADAD', '#B0B0B0', '#121212', '#B3B3B3', '#B5B5B5', '#B8B8B8', '#BABABA', '#BDBDBD', '#BFBFBF', '#C2C2C2', '#C4C4C4',
        '#C7C7C7', '#C9C9C9', '#141414', '#CCCCCC', '#CFCFCF', '#D1D1D1', '#D4D4D4', '#D6D6D6', '#D9D9D9', '#DBDBDB', '#DEDEDE', '#E0E0E0', '#E3E3E3', '#171717', '#E5E5E5', '#E8E8E8', '#EBEBEB',
        '#EDEDED', '#F0F0F0', '#F2F2F2', '#F7F7F7', '#FAFAFA', '#FCFCFC', '#008000', '#00FF00', '#00EE00', '#00CD00', '#008B00', '#ADFF2F', '#F0FFF0', '#E0EEE0', '#C1CDC1', '#838B83', '#FF69B4',
        '#FF6EB4', '#EE6AA7', '#CD6090', '#8B3A62', '#CD5C5C', '#FF6A6A', '#EE6363', '#CD5555', '#8B3A3A', '#4B0082', '#FFFFF0', '#EEEEE0', '#CDCDC1', '#8B8B83', '#292421', '#F0E68C', '#FFF68F',
        '#EEE685', '#CDC673', '#8B864E', '#E6E6FA', '#FFF0F5', '#EEE0E5', '#CDC1C5', '#8B8386', '#7CFC00', '#FFFACD', '#EEE9BF', '#CDC9A5', '#8B8970', '#ADD8E6', '#BFEFFF', '#B2DFEE', '#9AC0CD',
        '#68838B', '#F08080', '#E0FFFF', '#D1EEEE', '#B4CDCD', '#7A8B8B', '#FFEC8B', '#EEDC82', '#CDBE70', '#8B814C', '#FAFAD2', '#D3D3D3', '#FFB6C1', '#FFAEB9', '#EEA2AD', '#CD8C95', '#8B5F65',
        '#FFA07A', '#EE9572', '#CD8162', '#8B5742', '#20B2AA', '#87CEFA', '#B0E2FF', '#A4D3EE', '#8DB6CD', '#607B8B', '#8470FF', '#778899', '#B0C4DE', '#CAE1FF', '#BCD2EE', '#A2B5CD', '#6E7B8B',
        '#FFFFE0', '#EEEED1', '#CDCDB4', '#8B8B7A', '#32CD32', '#FAF0E6', '#FF00FF', '#EE00EE', '#CD00CD', '#8B008B', '#03A89E', '#800000', '#FF34B3', '#EE30A7', '#CD2990', '#8B1C62', '#BA55D3',
        '#E066FF', '#D15FEE', '#B452CD', '#7A378B', '#9370DB', '#AB82FF', '#9F79EE', '#8968CD', '#5D478B', '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#E3A869', '#191970', '#BDFCC9',
        '#F5FFFA', '#FFE4E1', '#EED5D2', '#CDB7B5', '#8B7D7B', '#FFE4B5', '#FFDEAD', '#EECFA1', '#CDB38B', '#8B795E', '#000080', '#FDF5E6', '#808000', '#6B8E23', '#C0FF3E', '#B3EE3A', '#9ACD32',
        '#698B22', '#FF8000', '#FFA500', '#EE9A00', '#CD8500', '#8B5A00', '#FF4500', '#EE4000', '#CD3700', '#8B2500', '#DA70D6', '#FF83FA', '#EE7AE9', '#CD69C9', '#8B4789', '#EEE8AA', '#98FB98',
        '#9AFF9A', '#90EE90', '#7CCD7C', '#548B54', '#BBFFFF', '#AEEEEE', '#96CDCD', '#668B8B', '#DB7093', '#FF82AB', '#EE799F', '#CD6889', '#8B475D', '#FFEFD5', '#FFDAB9', '#EECBAD', '#CDAF95',
        '#8B7765', '#33A1C9', '#FFC0CB', '#FFB5C5', '#EEA9B8', '#CD919E', '#8B636C', '#DDA0DD', '#FFBBFF', '#EEAEEE', '#CD96CD', '#8B668B', '#B0E0E6', '#800080', '#9B30FF', '#912CEE', '#7D26CD',
        '#551A8B', '#872657', '#C76114', '#FF0000', '#EE0000', '#CD0000', '#8B0000', '#BC8F8F', '#FFC1C1', '#EEB4B4', '#CD9B9B', '#8B6969', '#4169E1', '#4876FF', '#436EEE', '#3A5FCD', '#27408B',
        '#FA8072', '#FF8C69', '#EE8262', '#CD7054', '#8B4C39', '#F4A460', '#308014', '#54FF9F', '#4EEE94', '#43CD80', '#2E8B57', '#FFF5EE', '#EEE5DE', '#CDC5BF', '#8B8682', '#5E2612', '#8E388E',
        '#C5C1AA', '#71C671', '#555555', '#1E1E1E', '#282828', '#515151', '#5B5B5B', '#848484', '#8E8E8E', '#B7B7B7', '#C1C1C1', '#EAEAEA', '#F4F4F4', '#7D9EC0', '#AAAAAA', '#8E8E38', '#C67171',
        '#7171C6', '#388E8E', '#A0522D', '#FF8247', '#EE7942', '#CD6839', '#8B4726', '#C0C0C0', '#87CEEB', '#87CEFF', '#7EC0EE', '#6CA6CD', '#4A708B', '#6A5ACD', '#836FFF', '#7A67EE', '#6959CD',
        '#473C8B', '#708090', '#C6E2FF', '#B9D3EE', '#9FB6CD', '#6C7B8B', '#FFFAFA', '#EEE9E9', '#CDC9C9', '#8B8989', '#00FF7F', '#00EE76', '#00CD66', '#008B45', '#4682B4', '#63B8FF', '#5CACEE',
        '#4F94CD', '#36648B', '#D2B48C', '#FFA54F', '#EE9A49', '#CD853F', '#8B5A2B', '#008080', '#D8BFD8', '#FFE1FF', '#EED2EE', '#CDB5CD', '#8B7B8B', '#FF6347', '#EE5C42', '#CD4F39', '#8B3626',
        '#40E0D0', '#00F5FF', '#00E5EE', '#00C5CD', '#00868B', '#00C78C', '#EE82EE', '#D02090', '#FF3E96', '#EE3A8C', '#CD3278', '#8B2252', '#808069', '#F5DEB3', '#FFE7BA', '#EED8AE', '#CDBA96',
        '#8B7E66', '#FFFFFF', '#F5F5F5', '#FFFF00', '#EEEE00', '#CDCD00', '#8B8B00']

    legend = []

    plot_struc = ["ctv", "ptv", "bladder", "rectum", "femur"]

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

        for num, struc in enumerate(structures):
            if not "z" in struc["struc_name"].lower() and not struc["dvh_x"] is None:
                ax.plot(struc["dvh_x"], struc["target_dvh"], color=colors[num])
                ax.plot(struc["dvh_x"], struc["prediction_dvh"], "--", color=colors[num])
                legend.append(struc["struc_name"])
                legend.append(struc["struc_name"] + " pred")

        ax.legend(legend, bbox_to_anchor=(1.05, 0.5), loc='center left')
        fig.subplots_adjust(right=0.75)
        plt.savefig(path)


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

    patients = ["nt1"]
    for patient in patients:
        print(patient)
        root_dir = "/Users/simongutwein/Studium/Masterarbeit/plan_predictions_test/" + patient

        target_dose = np.array(torch.load(os.path.join(root_dir, f"{patient}_target.pt")))
        prediction_dose = np.array(torch.load(os.path.join(root_dir, f"{patient}_prediction.pt")))
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

        plot_dvh(structures, os.path.join(root_dir, f"{patient}_dvh_img.png"), diff=False)
        dvh_values_to_xlsx(structures, os.path.join(root_dir, f"{patient}_dvh_data.xlsx"))


if __name__ == "__main__":
    main()

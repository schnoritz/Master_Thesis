import argparse
import os
from model import Dose3DUNET
import torch
from plan_prediction import predict_plan
import nibabel as nib
import numpy as np
from dvh_from_structures import analyse_structures, plot_dvh, dvh_values_to_xlsx
from pydicom import dcmread
import pymedphys
import sys


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-test_dir',
        action='store',
        required=True,
        dest='test_dir'
    )

    parser.add_argument(
        '-model_path',
        action='store',
        required=True,
        dest='model_path'
    )

    parser.add_argument(
        '-save_dir',
        action='store',
        required=True,
        dest='save_dir'
    )

    return parser.parse_args()


def load_model(model_path):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model_checkpoint = torch.load(model_path, map_location=device)
    model = Dose3DUNET()
    model.load_state_dict(model_checkpoint['model_state_dict'])
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for predicting.")
        model = torch.nn.DataParallel(model)

    model.to(device)
    return model, device


def save_data(save_dir, test_case, model_name, predicted_dose, target_dose):

    case_save_path = os.path.join(save_dir, test_case)
    case_model_save_path = os.path.join(case_save_path, model_name)

    if not os.path.isdir(case_save_path):
        print(case_save_path)
        os.mkdir(case_save_path)

    if not os.path.isdir(case_model_save_path):
        print(case_model_save_path)
        os.mkdir(case_model_save_path)

    torch.save(predicted_dose, f"{case_model_save_path}/prediction.pt")
    torch.save(target_dose, f"{case_model_save_path}/target.pt")
    dat = nib.Nifti1Image(np.array(target_dose), np.eye(4))
    dat.header.get_xyzt_units()
    dat.to_filename(f"{case_model_save_path}/target.nii.gz")
    dat = nib.Nifti1Image(np.array(predicted_dose), np.eye(4))
    dat.header.get_xyzt_units()
    dat.to_filename(f"{case_model_save_path}/predicted.nii.gz")

    return case_model_save_path


def analyse_gamma(target_dose, predicted_dose, px_sp, save_path, gamma_percentage=3, gamma_distance=3, lower_cutoff=40, local_gamma=False, partial_sample=100000, ):

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

    coords = (np.arange(0, px_sp[0]*target_dose.shape[0], px_sp[0]), np.arange(
        0, px_sp[1]*target_dose.shape[1], px_sp[1]), np.arange(0, 3*target_dose.shape[2], 3))

    gamma_val = pymedphys.gamma(
        coords, np.array(target_dose),
        coords, np.array(predicted_dose),
        **gamma_options)

    dat = ~np.isnan(gamma_val)
    dat2 = ~np.isnan(gamma_val[gamma_val <= 1])
    all = np.count_nonzero(dat)
    true = np.count_nonzero(dat2)

    with open(f"{save_path}/gamma.txt", "w+") as fout:
        print(gamma_options, file=fout)
        print("\n\n", file=fout)
        print(np.round((true/all)*100, 4), file=fout)


def analyse_dvh(structure_file, ct_file, predicted_dose, target_dose, save_path):

    with dcmread(ct_file, force=True) as dcm_in:

        origin = np.array(dcm_in.ImagePositionPatient)
        px_sp = dcm_in.PixelSpacing
        px_sp.append(3)
        px_sp = np.array(px_sp)

    structures = analyse_structures(structure_file, origin, px_sp, predicted_dose.shape, target_dose, predicted_dose)
    plot_dvh(structures, f"{save_path}/dvh_img.png")
    dvh_values_to_xlsx(structures, f"{save_path}/dvh_data.xlsx")

    return px_sp


def main():
    args = parse()

    model_path = args.model_path
    if model_path[-1] != "/":
        model_path += "/"
    test_dir = args.test_dir
    save_dir = args.save_dir

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    model_origin = "_".join(model_path.split("/")[-3:-1])
    print(model_origin)

    test_cases = [x for x in os.listdir(test_dir) if not x.startswith(".") and os.path.isdir(os.path.join(test_dir, x))]

    model, device = load_model(model_path[:-1])

    for test_case in test_cases:
        print(test_case)
        sys.stdout.flush()
        case_path = os.path.join(test_dir, test_case)
        plan_file = os.path.join(test_dir, test_case, test_case + "_plan.dcm")
        structure_file = os.path.join(test_dir, test_case, test_case + "_strctr.dcm")
        ct_file = os.path.join(test_dir, test_case, "ct", "CT_image0.dcm")

        if os.path.isfile(plan_file) and os.path.isfile(structure_file) and os.path.isfile(ct_file):

            print(case_path, plan_file, structure_file, ct_file)
            target_dose, predicted_dose = predict_plan(model, device, plan_file, test_case, case_path, shift=16)
            target_dose, predicted_dose = np.array(target_dose), np.array(predicted_dose)
            print(target_dose.shape)

            # target_dose = np.array(torch.load("/mnt/qb/baumgartner/sgutwein84/test_cases/pt0/pt0_0/target_data.pt").squeeze())
            # predicted_dose = np.array(torch.load("/mnt/qb/baumgartner/sgutwein84/test_cases/pt0/pt0_0/target_data.pt").squeeze())

            save_path = save_data(save_dir, test_case, model_origin, predicted_dose, target_dose)
            px_sp = analyse_dvh(structure_file, ct_file, predicted_dose, target_dose, save_path)
            analyse_gamma(target_dose, predicted_dose, px_sp, save_path)

        else:
            print("Missing File for ", test_case)


if __name__ == "__main__":
    main()

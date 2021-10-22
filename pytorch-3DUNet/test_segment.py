from dataclasses import field
import os
import torch
from test_model import load_model
from entire_volume_prediction import predict_volume
import numpy as np
import pymedphys
import nibabel as nib


def save_data(save_dir, test_case, model_name, segment, predicted_dose, target_dose):

    case_save_path = os.path.join(save_dir, test_case)
    segment_save_path = os.path.join(case_save_path, segment)
    case_model_save_path = os.path.join(segment_save_path, model_name)

    if not os.path.isdir(case_save_path):
        print(case_save_path)
        os.mkdir(case_save_path)

    if not os.path.isdir(segment_save_path):
        print(segment_save_path)
        os.mkdir(segment_save_path)

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


def analyse_gamma(target_dose, predicted_dose, px_sp, gamma_percentage=3, gamma_distance=3, lower_cutoff=40, local_gamma=False, partial_sample=100000):

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

    return np.round((true/all)*100, 4), gamma_options


def main():

    root_dir = "/mnt/qb/baumgartner/sgutwein84/test_cases"
    save_dir = "/mnt/qb/baumgartner/sgutwein84/segment_results"

    models = ["/mnt/qb/baumgartner/sgutwein84/save/mixed_trained/UNET_1183.pt", "/mnt/qb/baumgartner/sgutwein84/save/prostate_trained/UNET_2234.pt"]

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    test_cases = [x for x in os.listdir(root_dir) if not x.startswith(".")]
    print(test_cases)

    for test_case in test_cases:
        path = os.path.join(root_dir, test_case)
        segments = [x for x in os.listdir(path) if not x.startswith(".") and not "ct" in x and not "dcm" in x]

        for segment in segments:
            segment_path = os.path.join(path, segment)
            target_data = torch.load(os.path.join(segment_path, "target_data.pt"))
            target_data = target_data.squeeze()
            training_data = torch.load(os.path.join(segment_path, "training_data.pt"))
            field_size = training_data[0].max()

            for model_path in models:

                model_name = "_".join(model_path.split("/")[-2:])

                save = os.path.join(save_dir, test_case, segment, model_name)

                if os.path.isdir(save):
                    print(save, " already exists")
                    continue

                model, device = load_model(model_path)
                prediction = predict_volume(training_data, model, device, shift=2)

                px_sp = (1.17185, 1.17185, 3)

                save_path = save_data(save_dir, test_case, model_name, segment, prediction, target_data)
                gamma, gamma_options = analyse_gamma(target_data, prediction, px_sp, lower_cutoff=10, partial_sample=20000)

                with open(f"{save_path}/gamma.txt", "w+") as fout:
                    print(gamma_options, file=fout)
                    print("\n\n", file=fout)
                    if all != 0:
                        print(gamma, file=fout)
                    else:
                        print(0, file=fout)
                    print("\n\n", file=fout)
                    print(field_size, file=fout)


if __name__ == "__main__":
    main()

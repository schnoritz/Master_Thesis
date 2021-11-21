
from sys import path
import sys
from pydicom import dcmread
import os
from natsort import natsorted
import pandas as pd
import torch
import numpy as np
from time import time
import pymedphys
import sys


def predict_plan(model, plan_path, patient, patient_path):

    dicom_file = dcmread(plan_path, force=True)

    meterset_weights = []
    beam_metersets = []
    beam_nums = []
    for beam in dicom_file.FractionGroupSequence[0].ReferencedBeamSequence:
        beam_metersets.append({
            "beam_number": beam.ReferencedBeamNumber,
            "beam_meterset": beam.BeamMeterset
        })

    for beam in dicom_file.BeamSequence:
        for control_point in beam.ControlPointSequence:
            beam_nums.append(beam.BeamNumber)
            meterset_weights.append(float(control_point.CumulativeMetersetWeight))

    segment_weights = [meterset_weights[i] - meterset_weights[i-1] for i in range(1, len(meterset_weights), 2)]
    beam_nums = beam_nums[::2]
    patient_segments = [x for x in os.listdir(patient_path) if patient + "_" in x and not "plan" in x and not "strctr" in x]
    patient_segments = natsorted(patient_segments)
    assert len(segment_weights) == len(patient_segments), f"Number of segments ({len(patient_segments)}) does not match number of weights ({len(segment_weights)})!"
    print(f"Plan has {len(segment_weights)} segments.")

    segment_weights = pd.DataFrame({
        'beam_number': beam_nums,
        'segment': patient_segments,
        'weight': segment_weights,
    })

    beam_metersets = pd.DataFrame(beam_metersets)

    segment_weights['beam_number'] = segment_weights['beam_number'].map(beam_metersets.set_index('beam_number')['beam_meterset'])

    final_target_dose = np.zeros_like(torch.load(os.path.join(patient_path, segment_weights.iloc[0]['segment'], model, "target.pt")).squeeze())
    final_prediction_dose = np.zeros_like(final_target_dose)

    for _, row in segment_weights.iterrows():

        print(row['segment'], end="")
        start = time()
        target = torch.load(os.path.join(patient_path, row['segment'], model, "target.pt")).squeeze()
        pred = torch.load(os.path.join(patient_path, row['segment'], model, "prediction.pt")).squeeze()
        print("Loading: ", np.round(time()-start, 2), " seconds")
        final_target_dose = np.add(final_target_dose, target * row['weight'] * row["beam_number"]/100)
        final_prediction_dose = np.add(final_prediction_dose, pred * row['weight'] * row["beam_number"]/100)

    return final_target_dose, final_prediction_dose


if __name__ == "__main__":

    import nibabel as nib

    root = "/mnt/qb/baumgartner/sgutwein84/segment_results"
    patients = [x for x in os.listdir(root) if not x.startswith(".")]
    #patients = ["nt0"]
    models = ["prostate_trained_UNET_2234.pt", "mixed_trained_UNET_1183.pt"]

    for patient in patients:
        for model in models:
            print(patient, "  ", model)

            patient_dir = f"/mnt/qb/baumgartner/sgutwein84/segment_results/{patient}/"
            save_pat_path = f"/mnt/qb/baumgartner/sgutwein84/plan_predictions_load/{patient}"
            save_path = save_pat_path + f"/{model}"
            plan_path = f"/mnt/qb/baumgartner/sgutwein84/test_cases/{patient}/{patient}_plan.dcm"

            if not os.path.isdir(save_pat_path):
                os.mkdir(save_pat_path)

            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            else:
                print(patient, "  ", model, " already exists!")
                continue

            sys.stdout.flush()

            target_plan, predicted_plan = predict_plan(model, plan_path, patient, patient_dir)

            gamma_options = {
                'dose_percent_threshold': 3,
                'distance_mm_threshold': 3,
                'lower_percent_dose_cutoff': 10,
                'interp_fraction': 10,  # Should be 10 or more for more accurate results
                'max_gamma': 1.1,
                'quiet': False,
                'local_gamma': False,
                'random_subset': 20000
            }

            coords = (np.arange(0, 1.17*target_plan.shape[0], 1.17), np.arange(
                0, 1.17*target_plan.shape[1], 1.17), np.arange(0, 3*target_plan.shape[2], 3))

            gamma_val = pymedphys.gamma(
                coords, np.array(target_plan),
                coords, np.array(predicted_plan),
                **gamma_options)

            dat = ~np.isnan(gamma_val)
            dat2 = ~np.isnan(gamma_val[gamma_val <= 1])
            all = np.count_nonzero(dat)
            true = np.count_nonzero(dat2)

            print("Gamma pass rate:", np.round((true/all)*100, 2), "%")

            with open(f"{save_path}/gamma.txt", "w+") as fout:
                print(gamma_options, file=fout)
                print("\n\n", file=fout)
                if all != 0:
                    print(np.round((true/all)*100, 4), file=fout)
                else:
                    print(0, file=fout)

            torch.save(predicted_plan, f"{save_path}/prediction.pt")
            torch.save(target_plan, f"{save_path}/target.pt")

            dat = nib.Nifti1Image(np.array(target_plan), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{patient}_target_plan.nii.gz")
            dat = nib.Nifti1Image(np.array(predicted_plan), np.eye(4))
            dat.header.get_xyzt_units()
            dat.to_filename(f"{save_path}/{patient}_predicted_plan.nii.gz")

            sys.stdout.flush()

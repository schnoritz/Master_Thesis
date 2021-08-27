
from sys import path
from pydicom import dcmread
import os
from natsort import natsorted
import pandas as pd
import torch
import numpy as np
from entire_volume_prediction import predict_volume
from time import time


def predict_plan(model, device, plan_path, patient, segments_path):

    dicom_file = dcmread(plan_path, force=True)

    meterset_weights = []
    for beam in dicom_file.BeamSequence:
        for control_point in beam.ControlPointSequence:
            meterset_weights.append(float(control_point.CumulativeMetersetWeight))

    segment_weights = [meterset_weights[i] - meterset_weights[i-1] for i in range(1, len(meterset_weights), 2)]

    patient_segments = [x for x in os.listdir(segments_path) if patient + "_" in x]
    patient_segments = natsorted(patient_segments)
    assert len(segment_weights) == len(patient_segments), f"Number of segments ({len(patient_segments)}) does not match number of weights ({len(segment_weights)})!"
    print(f"Plan has {len(segment_weights)} segments.")

    segment_weights = pd.DataFrame({
        'segment': patient_segments,
        'weight': segment_weights
    })

    final_target_dose = np.zeros_like(torch.load(os.path.join(segments_path, segment_weights.iloc[0]['segment'], "target_data.pt")).squeeze())
    final_prediction_dose = np.zeros_like(final_target_dose)

    start = time()
    for _, row in segment_weights.iterrows():

        print(row['segment'])
        target = torch.load(os.path.join(segments_path, row['segment'], "target_data.pt")).squeeze()
        masks = torch.load(os.path.join(segments_path, row['segment'], "training_data.pt")).squeeze()

        final_target_dose = np.add(final_target_dose, target * row['weight'])
        final_prediction_dose = np.add(final_prediction_dose, predict_volume(masks, model, device) * row['weight'])

    print(f"Total plan prediction took: {np.round(time()-start,2)} seconds.")

    return final_target_dose, final_prediction_dose


if __name__ == "__main__":

    from model import Dose3DUNET
    import nibabel as nib

    patients = ["p0", "l0"]
    entities = ["hlmp", "hlmp"]
    for patient, entity in zip(patients, entities):

        model_path = "/mnt/qb/baumgartner/sgutwein84/save/bs128_ps32_lr4_2108231722/UNET_270.pt"
        segment_dir = f"/mnt/qb/baumgartner/sgutwein84/training/training_{entity}/"
        save_path = "/home/baumgartner/sgutwein84/container/test"
        plan_path = f"/mnt/qb/baumgartner/sgutwein84/planfiles/{patient}_plan.dcm"

        device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
        model_checkpoint = torch.load(model_path, map_location=device)
        model = Dose3DUNET()
        model.load_state_dict(model_checkpoint['model_state_dict'])
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for predicting.")
            model = torch.nn.DataParallel(model)

        model.to(device)

        target_plan, predicted_plan = predict_plan(model, device, plan_path, patient, segment_dir)

        gamma_options = {
            'dose_percent_threshold': 3,
            'distance_mm_threshold': 3,
            'lower_percent_dose_cutoff': 10,
            'interp_fraction': 10,  # Should be 10 or more for more accurate results
            'max_gamma': 1.1,
            'quiet': True,
            'local_gamma': False,
            'random_subset': 20000
        }

        import pymedphys

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

        torch.save(predicted_plan, f"/home/baumgartner/sgutwein84/container/test/{patient}_prediction.pt")
        torch.save(target_plan, f"/home/baumgartner/sgutwein84/container/test/{patient}_target.pt")

        dat = nib.Nifti1Image(np.array(target_plan), np.eye(4))
        dat.header.get_xyzt_units()
        dat.to_filename(f"{save_path}/{patient}_target_plan.nii.gz")
        dat = nib.Nifti1Image(np.array(predicted_plan), np.eye(4))
        dat.header.get_xyzt_units()
        dat.to_filename(f"{save_path}/{patient}_predicted_plan.nii.gz")

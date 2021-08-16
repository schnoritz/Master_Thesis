
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

    model_path = "/mnt/qb/baumgartner/sgutwein84/save/bs32_ps32_5/UNET_896.pt"
    segment_dir = "/mnt/qb/baumgartner/sgutwein84/training_mamma/"
    save_path = "/home/baumgartner/sgutwein84/container/test"
    patient = "m0"
    plan_path = f"/mnt/qb/baumgartner/sgutwein84/{patient}_plan.dcm"

    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    model_checkpoint = torch.load(model_path, map_location=device)
    model = Dose3DUNET(UQ=False)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for predicting.")
        model = torch.nn.DataParallel(model)

    model.to(device)

    target_plan, predicted_plan = predict_plan(model, device, plan_path, patient, segment_dir)

    torch.save(predicted_plan, f"/home/baumgartner/sgutwein84/container/test/{patient}_prediction.pt")
    torch.save(target_plan, f"/home/baumgartner/sgutwein84/container/test/{patient}_target.pt")

    dat = nib.Nifti1Image(np.array(target_plan), np.eye(4))
    dat.header.get_xyzt_units()
    dat.to_filename(f"{save_path}/{patient}_target_plan.nii.gz")
    dat = nib.Nifti1Image(np.array(predicted_plan), np.eye(4))
    dat.header.get_xyzt_units()
    dat.to_filename(f"{save_path}/{patient}_predicted_plan.nii.gz")

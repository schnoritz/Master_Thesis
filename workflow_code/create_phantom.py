from pydicom import dcmread, uid
import numpy as np
import os

import matplotlib.pyplot as plt


def create_beam_files(save_path: str, phantom_name: str, fieldsize: int):

    for suffix in [".txt", ".egsinp"]:
        filename = f"/Users/simongutwein/Studium/Masterarbeit/.test_fields/{fieldsize}x{fieldsize}{suffix}"
        target_filename = os.path.join(save_path, phantom_name, f"beam_config_{phantom_name}_{fieldsize}x{fieldsize}{suffix}")
        os.popen(f'cp {filename} {target_filename}')


def create_dose_phantom(save_path: str, phantom_name: str) -> None:

    save_dir = os.path.join(save_path, phantom_name)

    dose = dcmread("/Users/simongutwein/Studium/Masterarbeit/.sample_ct/dose_template.dcm", force=True)

    dose.SliceThickness = 3
    dose.NumberOfFrames = "200"
    dose.ImagePositionPatient = [0, 0, 0]
    size = (200, 200, 200)

    phantom_dose = np.zeros(size).astype(np.uint16)
    dose.Columns = phantom_dose.shape[0]
    dose.Rows = phantom_dose.shape[1]
    dose.PixelData = phantom_dose.tobytes()
    dose.GridFrameOffsetVector = list(range(0, 200*3, 3))

    dose.save_as(os.path.join(save_dir, f"{phantom_name}_dose.dcm"))


def create_image_data(slab_position: int, slab_thickness: int, slab_value: int, ct_size: tuple, water_phantom: bool) -> np.ndarray:

    if water_phantom:
        return (np.ones((ct_size[0], ct_size[1]))*1024).astype(np.uint16)

    top = np.zeros((slab_position, ct_size[1]))
    slab = np.ones((slab_thickness, ct_size[1]))*slab_value
    bot = np.zeros((ct_size[0]-slab_position-slab_thickness, ct_size[1]))

    return np.concatenate((top, slab, bot)).astype(np.uint16)


def create_ct_phantom(slab_position: int, slab_thickness: int, slab_value: int, save_path: str, phantom_name: str, water_phantom=False) -> None:

    save_dir = os.path.join(save_path, phantom_name)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    ct = dcmread("/Users/simongutwein/Studium/Masterarbeit/.sample_ct/ct_template.dcm", force=True)

    ct_size = (512, 512)
    slice_thickness = 3
    pixel_spacing = (1.171875, 1.171875)
    slices = int((ct_size[0]*pixel_spacing[0])/slice_thickness)

    pixel_array = create_image_data(slab_position, slab_thickness, slab_value, ct_size, water_phantom)

    ct.Columns = pixel_array.shape[0]
    ct.Rows = pixel_array.shape[1]
    ct.PixelData = pixel_array.tobytes()

    locations = list(range(0, 200*3, 3))

    for i in range(slices):
        ct.ImagePositionPatient = [0, 0, locations[i]]
        ct.SliceLocation = locations[i]
        ct.save_as(os.path.join(save_dir, f"CT_image{i}.dcm"))


def main():

    save_path = "/Users/simongutwein/Studium/Masterarbeit/phantoms"

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    positions = [100, 200, 300]
    thickness = 200
    HU_value = 1024
    fieldsize = 10

    water_phantom = False

    for position in positions:

        if water_phantom:
            phantom_name = "water_phantom"
        else:
            phantom_name = f"phantomP{position}T{thickness}"

        create_ct_phantom(position, thickness, HU_value, save_path, phantom_name, water_phantom=water_phantom)
        create_dose_phantom(save_path, phantom_name)
        create_beam_files(save_path, phantom_name, fieldsize)


if __name__ == "__main__":

    main()

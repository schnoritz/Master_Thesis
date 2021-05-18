import numpy as np


def define_iso_center(egsinp, egsphant, px_sp=np.array([1.171875, 1.171875, 3])):

    iso_center = np.array(egsinp.split(",")[2:5], dtype=float)

    dimension, zero_point = extract_phantom_data(egsphant)

    iso_pos = np.round((iso_center - zero_point) / 0.3).astype(int)
    iso_pos[0], iso_pos[1] = iso_pos[1], iso_pos[0]
    iso_pos[0] += int(dimension[0] - dimension[1])
    iso_pos = np.round(iso_pos * np.array([3, 3, 3])/px_sp).astype(int)

    assert iso_pos[0] < 512, "isocenter lies outside of Volume"
    assert iso_pos[1] < 512, "isocenter lies outside of Volume"

    return iso_pos


def define_origin_position(egsinp, iso_center, px_sp=1.171875, SID=1435):

    angle = np.radians(float(egsinp.split(",")[6]) - 270)

    pixel_SID = int(SID / px_sp)
    origin = np.array([
        iso_center[0] - np.cos(angle) * pixel_SID,
        iso_center[1] + np.sin(angle) * pixel_SID,
        iso_center[2]]).astype("float")

    return origin


def get_num_slices(egsphant):

    return int(open(egsphant,"r").readlines()[6].strip().split("  ")[2])


def get_angle(egsinp, radians=False):

    if radians == False:
        return float(egsinp.split(",")[6]) - 270
    else:
        return np.radians(float(egsinp.split(",")[6]) - 270)


def extract_phantom_data(egsphant):

    with open(egsphant, 'r') as fin:
        lines = fin.readlines()
        dimension = np.array(lines[6].strip().split("  "),dtype=np.uint16)
        zero_point = [float(lines[7].strip().split("  ")[0]),
                      float(lines[8].strip().split("  ")[0]),
                      float(lines[9].strip().split("  ")[0])]

    return dimension, zero_point
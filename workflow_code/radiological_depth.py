import numpy as np
import torch
from tqdm import tqdm
from depth_ray import Ray

def radiological_depth(ct, egsinp):

    egsinp_lines = open(egsinp).readlines()

    iso_center = define_iso_center(egsinp_lines[5])
    origin_position = define_origin_position(egsinp_lines[5], iso_center)

    ct_volume = np.array(torch.load(ct), dtype=float)

    depth = calc_depth(ct_volume, origin_position)

    return torch.tensor(depth, dtype=torch.float16)


def calc_depth(ct_volume, origin_position):

    depth_volume = np.zeros_like(ct_volume, dtype=float)

    for x in tqdm(range(ct_volume.shape[0])):
        for y in tqdm(range(ct_volume.shape[1])):
            for z in range(ct_volume.shape[2]):
                vox = np.array([x, y, z], dtype=float)
                ray = Ray(origin_position, vox, ct_volume)
                depth_volume[x, y, z] = ray.radiological_depth

    return depth_volume


def define_iso_center(egsinp):

    return np.array(egsinp.split(",")[2:5], dtype=np.float16)


def define_origin_position(egsinp, iso_center, px_sp=1.171875):

    angle = np.radians(float(egsinp[6])-270)

    pixel_SID = int(1435/px_sp)
    origin = np.array([iso_center[0] - np.cos(angle)*pixel_SID,
                       iso_center[1] + np.sin(angle)*pixel_SID,
                       iso_center[2]]).astype("float")

    return origin


if __name__ == "__main__":

    radiological_depth(
        "/home/baumgartner/sgutwein84/container/output/p.pt",
        "/home/baumgartner/sgutwein84/container/output/p_0_2x2/p_0_2x2.egsinp")

import numpy as np
import torch
from tqdm import tqdm
from depth_ray import Ray
from utils import define_iso_center, define_origin_position


def radiological_depth(ct, egsinp, egsphant, mask=False, tensor=False):

    egsinp_lines = open(egsinp).readlines()

    iso_center = define_iso_center(egsinp_lines[5], egsphant)
    origin_position = define_origin_position(egsinp_lines[5], iso_center)

    if type(ct) == str:
        ct_volume = np.load(ct).astype(np.float)

    elif isinstance(ct, np.ndarray):
        ct_volume = ct.astype(float)

    elif isinstance(ct, torch.Tensor):
        ct_volume = np.array(ct).astype(float)

    depth = calc_depth(ct_volume, origin_position, mask)

    if tensor:
        return torch.tensor(depth, dtype=torch.float32)
    else:
        return depth.astype(np.float32)


def calc_depth(ct_volume, origin_position, mask):

    depth_volume = np.zeros_like(ct_volume, dtype=float)

    if not isinstance(mask, np.ndarray):
        for x in tqdm(range(ct_volume.shape[0])):
            for y in range(ct_volume.shape[1]):
                for z in range(ct_volume.shape[2]):
                    vox = np.array([x, y, z], dtype=float)
                    ray = Ray(origin_position, vox, ct_volume)
                    depth_volume[x, y, z] = ray.radiological_depth

    elif isinstance(mask, np.ndarray):
        for x in tqdm(range(ct_volume.shape[0])):
            for y in range(ct_volume.shape[1]):
                for z in range(ct_volume.shape[2]):

                    if mask[x, y, z] != 0:
                        vox = np.array([x, y, z], dtype=float)
                        ray = Ray(origin_position, vox, ct_volume)
                        depth_volume[x, y, z] = ray.radiological_depth

                    else:
                        depth_volume[x, y, z] = 0

    return depth_volume


# if __name__ == "__main__":

#     depth_volume = radiological_depth(
#         "/home/baumgartner/sgutwein84/container/output/p.pt",
#         "/home/baumgartner/sgutwein84/container/output/p_0_2x2/p_0_2x2.egsinp",
#         "/home/baumgartner/sgutwein84/container/output/p.egsphant")

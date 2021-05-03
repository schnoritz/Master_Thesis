import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy import ndimage
from time import time
from pydicom import dcmread


def get_leaf_positions(path):

    if path.split(".")[-1] == "dcm":
        with dcmread(path) as dcm:
            leafes = np.array(dcm.BeamSequence[0].ControlPointSequence[0].BeamLimitingDevicePositionSequence[1][0x300a, 0x011c][:], float)/10
            jaws = np.array(dcm.BeamSequence[0].ControlPointSequence[0].BeamLimitingDevicePositionSequence[0][0x300a, 0x011c][:], float)

            positions = np.array([leafes[:80], leafes[80:]]).T
            
    else:

        with open(path, "r") as fin:
            lines = fin.readlines()

        positions = []
        for line in lines:
            positions.append(np.array(line.split(",")[:2], dtype=float))

    if 'jaws' in locals():
        return positions, jaws
    else: 
        return np.array(positions), None



def get_first_layer(positions, jaws, slice_increase, transversal_size, sagital_size, dim):

    fl_positions = []
    for leaf in positions:
        fl_positions.append(leaf * slice_increase)

    #0.05cm * 220 -> genaugikeit der leafes auf halben mm
    fl = np.ones((440, 80))  

    i = -1
    for top, bot in fl_positions:
        i += 1
        if top != 0:
            fl[:219+int(np.round(top*20)), i] = 0
        if bot != 0:
            fl[219+int(np.round(bot*20)):, i] = 0

    plt.imshow(resize(fl, (400, 400)))
    plt.show()

    fl = resize(fl, (transversal_size, sagital_size))
    l_jaw_idx = sagital_size//2 + int(np.round(jaws[0]))
    r_jaw_idx = sagital_size//2 + int(np.round(jaws[1]))
    fl[:, :l_jaw_idx] = 0
    fl[:, r_jaw_idx:] = 0

    t_padding = [int(np.floor((dim[2]-transversal_size) /
                              2)), int(np.ceil((dim[2]-transversal_size)/2))]
    s_padding = [int(np.floor((dim[0]-sagital_size) /
                              2)), int(np.ceil((dim[0]-sagital_size)/2))]

    fl = np.pad(fl, ((t_padding[0], t_padding[1]),
                (s_padding[0], s_padding[1])), 'constant', constant_values=0)

    return fl


def scale_beam(fl, dim, scale):

    vol = np.empty((fl.shape[0], fl.shape[1], dim[0]))
    
    for slice in range(dim[0]):
        vol[:, :, slice] = clipped_zoom(fl, 1+scale*slice)

    return vol


def clipped_zoom(img, scale):

    h, w = img.shape[:2]
    zoom_tuple = (scale, scale)

    zh = int(np.ceil(h / scale))
    zw = int(np.ceil(w / scale))
    top = (h - zh) // 2
    left = (w - zw) // 2

    out = ndimage.zoom(img[top:top+zh, left:left+zw],
                       zoom_tuple, order=1, prefilter=False)

    trim_top = ((out.shape[0] - h) // 2)
    trim_left = ((out.shape[1] - w) // 2)

    out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    return out


if __name__ == "__main__":

    #file = "/Users/simongutwein/Studium/Masterarbeit/MR-Linac_model_10x10_0x0.egsinp"
    file = "/Users/simongutwein/Studium/Masterarbeit/MbaseMRL.dcm"
    voxel_size = [1.171875, 1.171875, 3]
    SID = 1435
    angle = 0

    fieldsize_px = [int(np.round(220 / voxel_size[2])),
                    int(np.round(576 / voxel_size[0]))]

    output_dim = [512, 512, 110]
    rotated_dim = np.array(
        [1.01*np.sqrt(2*(output_dim[0]**2)), 1.01*np.sqrt(2*(output_dim[1]**2)), output_dim[2]], dtype=int)

    print(rotated_dim)

    iso_center = np.array(
        [rotated_dim[0]/2-1, rotated_dim[0]/2-1, 54], dtype=int)

    increase_per_slice = abs(1 - SID/(SID - voxel_size[0]))
    first_layer_factor = 1 - (rotated_dim[0] - iso_center[0]) * increase_per_slice
    
    leaf_positions, jaws = get_leaf_positions(file)
    jaws = jaws/voxel_size[0]
    first_layer = get_first_layer(leaf_positions, jaws, first_layer_factor, fieldsize_px[0], fieldsize_px[1], rotated_dim)
    
    plt.imshow(first_layer)
    plt.show()

    beam = scale_beam(first_layer, rotated_dim, increase_per_slice)

    print(beam.shape)

    rotated = ndimage.rotate(beam, angle, axes=(1,2), reshape=False, prefilter=False)

    crop = (rotated_dim - output_dim)//2

    output = rotated[crop[2]:crop[2]+output_dim[2], crop[1]:crop[1]+output_dim[1], crop[0]:crop[0]+output_dim[0]]
    output = np.transpose(output, (2,1,0))
    for i in range(output.shape[1]):
        plt.imshow(output[i, :, :])
        plt.show()

    print(output.shape)

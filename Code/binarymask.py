import numpy as np
import matplotlib.pyplot as plt
#from skimage.transform import resize
from cv2 import resize
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

        jaws = np.array(positions[-1])*10

        positions = positions[:-1]

    return positions, jaws


def get_first_layer(positions, jaws, first_layer_fac, transversal_size, sagital_size, dim):


    fl_positions = []
    for leaf in positions:
        fl_positions.append(leaf * first_layer_fac)

    #0.05cm * 220 -> genaugikeit der leafes auf halben mm
    fl = np.ones((440, 80))  

    i = 0
    for top, bot in fl_positions:
        if top != 0:
            fl[:219+int(np.round(top*20)), i] = 0
        if bot != 0:
            fl[219+int(np.round(bot*20)):, i] = 0
        i+=1

    fl = resize(fl, (sagital_size, transversal_size))

    l_jaw_idx = int(np.round(sagital_size/2)) + int(np.round(jaws[0]*first_layer_fac))
    r_jaw_idx = int(np.round(sagital_size/2)) + int(np.round(jaws[1]*first_layer_fac))
    fl[:, :l_jaw_idx] = 0
    fl[:, r_jaw_idx+1:] = 0

    t_padding = [int(np.ceil((dim[2]-transversal_size) /
                              2)), int(np.floor((dim[2]-transversal_size)/2) )]
    s_padding = [int(np.ceil((dim[0]-sagital_size) /
                              2)), int(np.floor((dim[0]-sagital_size)/2))]

    fl = np.pad(fl, ((t_padding[0], t_padding[1]),
                (s_padding[0], s_padding[1])), 'constant', constant_values=0)

    return fl


def scale_beam(fl, dim, scale):

    vol = np.empty((fl.shape[0], fl.shape[1], dim[0]))
    
    for slice in range(dim[0]):
        vol[:, :, slice] = clipped_zoom(fl, scale[slice])

    return vol


def clipped_zoom(img, scale):

    h, w = img.shape[:2]
    new_h = int(np.ceil(scale*h))
    new_w = int(np.ceil(scale*w))
    top = int(np.round((new_h - h) / 2,0))
    left = int(np.round((new_w - w) / 2,0))
    out = resize(img,(new_w, new_h))
    out = out[top:top+h, left:left+w]

    return out


def create_binary_mask(egsphant, egsinp, angle):

    voxel_size = np.array([1.171875, 1.171875, 3])
    SID = 1435
    
    iso_center, num_slices = get_iso_center(egsphant, egsinp, voxel_size)
    output_dim = np.array([512, 512, num_slices], dtype=int)
    shift = output_dim//2 - iso_center 

    rotated_dim = np.array(
        [1.1*np.sqrt(2*(output_dim[0]**2)), 1.1*np.sqrt(2*(output_dim[1]**2)), output_dim[2]+2*shift[2]], dtype=int)

    fac = [(voxel_size[0] * (SID + i * voxel_size[0]))/(2 * SID) / (voxel_size[0]/2) for i in range(0-iso_center[0], rotated_dim[0] - iso_center[0])]

    leaf_positions, jaws = get_leaf_positions(file)
    jaws = jaws/voxel_size[0]

    fieldsize_px = [int(np.round(220 / voxel_size[2])),
                    int(np.round(570 / voxel_size[0]))]

    first_layer = get_first_layer(
        leaf_positions, jaws, fac[0], fieldsize_px[0], fieldsize_px[1], rotated_dim)

    fac = [fac[i]/fac[0] for i in range(1,len(fac))]
    fac.insert(0, 1)

    beam = scale_beam(first_layer, rotated_dim, fac)

    beam = np.transpose(beam, (2,1,0))

    rotated = ndimage.rotate(beam, -angle, axes=(
        0, 1), reshape=False, prefilter=False)
    print(rotated.shape)

    len_vec = np.sqrt(shift[0]**2 + shift[1]**2)
    center_shift = np.array([np.round(len_vec * np.sin(np.radians(angle))), np.round(len_vec * np.cos(np.radians(angle))), shift[2]], dtype=int)

    #########################################################################################

    center_window = np.array([(rotated.shape[0]-output_dim[0])//2, (rotated.shape[1]-output_dim[1])//2, (rotated.shape[2]-output_dim[2])//2])
    shifted_window = center_window - center_shift
    crop = np.array([ 0, 0, 0], dtype=int)

    output = rotated[shifted_window[0]:shifted_window[0] + output_dim[0],
                     shifted_window[1]:shifted_window[1] + output_dim[1], 
                     shifted_window[2]:shifted_window[2] + output_dim[2]]

    print(output.shape)

    #########################################################################################

    return output

def get_iso_center(egsphant_path, egsinp_file, ct_voxel_size):

    fac = np.array([3, 3, 3])/ct_voxel_size

    with open(egsphant, "r") as fin:
        lines = fin.readlines()
    shape = np.array(
        list(filter(None, np.array(lines[6].strip().split(" ")))), dtype=int)
    print(shape)

    pos = np.array([lines[7].strip().split(" ")[0], lines[8].strip().split(" ")[
                   0], lines[9].strip().split(" ")[0]], dtype=float)

    with open(egsinp_file, "r") as fin:

        lines = fin.readlines()
        iso = np.array(lines[5].split(",")[2:5], dtype=float)

    iso_pos_vox = np.round((iso-pos)/0.3).astype(int)
    
    top = int(shape[0]-shape[1])
    iso_pos_vox[1] += top

    iso_pos_vox = np.round(iso_pos_vox*fac).astype(int)
    iso_pos_vox[0], iso_pos_vox[1] = iso_pos_vox[1], iso_pos_vox[0]

    return iso_pos_vox, shape[2]



if __name__ == "__main__":

    egsinp = "/Users/simongutwein/Studium/Masterarbeit/p.egsinp"
    egsphant = "/Users/simongutwein/Studium/Masterarbeit/p.egsphant"


    for angle in [90]:
        for fz in [2]:#[2,3,4,5,6,7,8,9,10]:
            file = f"/home/baumgartner/sgutwein84/container/training_data/training_fields/MR-Linac_model_{fz}x{fz}.txt"
            binary_mask = create_binary_mask(egsphant, egsinp, angle=angle)

            dose = np.load(
                f"/home/baumgartner/sgutwein84/container/training_data/training/target/p_{int(angle)}_{fz}x{fz}.npy")

            im = np.zeros((512, 512))
            im[254:256, 254:256] = 1
            plt.imshow(im)
            plt.imshow(binary_mask[:, :, 55],alpha =0.5)
            plt.imshow(dose[:, :, 55], alpha = 0.5)
            plt.show()
            
            plt.savefig(f"/home/baumgartner/sgutwein84/container/utils/test/image_{int(angle)}_{fz}x{fz}")
            plt.close()
            print(f"Image for Angle {angle} and fieldsize {fz} saved!")

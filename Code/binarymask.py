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


def get_first_layer(positions, jaws, first_layer_fac, transversal_size, sagital_size, dim, pixel_shift):


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


def get_iso_center(dim, plan_file_dat, voxel_size):

    shift = plan_file_dat.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
    pixel_shift = np.round((np.array(shift)/voxel_size[0])).astype(int)
    pixel_shift = [pixel_shift[1], pixel_shift[0]]

    iso_center = np.array(
        [dim[0]//2-1-pixel_shift[1], dim[1]//2-1, dim[2]//2-1], dtype=int)

    return iso_center, pixel_shift

def create_binary_mask(file, plan_file, angle):

    voxel_size = [1.171875, 1.171875, 3]
    SID = 1435
    
    plan_file_dat = dcmread(plan_file)

    fieldsize_px = [int(np.round(220 / voxel_size[2])),
                    int(np.round(570 / voxel_size[0]))]

    output_dim = [512, 512, 110]
    rotated_dim = np.array(
        [1.1*np.sqrt(2*(output_dim[0]**2)), 1.1*np.sqrt(2*(output_dim[1]**2)), output_dim[2]], dtype=int)

    iso_center, pixel_shift = get_iso_center(rotated_dim, plan_file_dat, voxel_size)

    fac = [(voxel_size[0] * (SID + i * voxel_size[0]))/(2 * SID) / (voxel_size[0]/2) for i in range(0-iso_center[0], rotated_dim[0] - iso_center[0])]

    leaf_positions, jaws = get_leaf_positions(file)
    jaws = jaws/voxel_size[0]

    first_layer = get_first_layer(
        leaf_positions, jaws, fac[0], fieldsize_px[0], fieldsize_px[1], rotated_dim, pixel_shift)

    fac = [fac[i]/fac[0] for i in range(1,len(fac))]
    fac.insert(0, 1)

    beam = scale_beam(first_layer, rotated_dim, fac)

    beam = np.transpose(beam, (2,1,0))

    rotated = ndimage.rotate(beam, -angle, axes=(
        0, 1), reshape=False, prefilter=False)

    # len_vec = np.sqrt(pixel_shift[0]**2 + pixel_shift[1]**2)
    # center_shift = [len_vec * np.cos(np.radians(angle)), len_vec * np.sin(np.radians(angle))]

    crop = np.array([(rotated.shape[0]-output_dim[0])//2 - int(np.round(pixel_shift[0]/2)),
                    (rotated.shape[1]-output_dim[1])//2,
                    (rotated.shape[2]-output_dim[2])//2], dtype=int)


    output = rotated[crop[0]:crop[0]+output_dim[0], 
                     crop[1]:crop[1]+output_dim[1],
                     crop[2]:crop[2]+output_dim[2]]

    return output

if __name__ == "__main__":

    #file = "/Users/simongutwein/Studium/Masterarbeit/MbaseMRL.dcm"
    #plan_file = "/Users/simongutwein/Studium/Masterarbeit/MbaseMRL.dcm"
    plan_file = "/home/baumgartner/sgutwein84/container/utils/p_plan.dcm"

    for angle in [90]:
        for fz in [2]:#[2,3,4,5,6,7,8,9,10]:
            file = f"/home/baumgartner/sgutwein84/container/training_data/training_fields/MR-Linac_model_{fz}x{fz}.txt"
            binary_mask = create_binary_mask(file, plan_file, angle=angle)

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

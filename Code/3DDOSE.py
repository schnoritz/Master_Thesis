#%%
from skimage.transform import resize
from pydicom import dcmread
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap


def read_3ddose_file(filepath):

	with open(filepath, 'r') as fout:

		dim = np.array(fout.readline().split()).astype("int")
		for i in range(3):
			fout.readline()
		dose_volume = np.array(fout.readline().split()).astype("float")
		dose_volume = dose_volume.reshape(dim[0], dim[1], dim[2], order='F')
		dose_volume = dose_volume.transpose((1, 0, 2))

	return dose_volume


def get_ct_image(dose_files, image_files):

	ct_files = [file_ for file_ in sorted(os.listdir(dose_files)) if not file_.startswith(('._', '.', 'listfile'))]

	slice_position = []
	for file_ in ct_files:
		dat = dcmread(dose_files + file_, force=True)
		slice_position.append(str(dat.SliceLocation))

	files = os.listdir(image_files)
	files.remove('.DS_Store')
	filenames = []
	for file_ in files:
		ct = dcmread(image_files + file_)
		if str(ct.ImagePositionPatient[2]) in slice_position:
			filenames.append(file_)

	name = []
	position = []
	for file_ in filenames:
		dat = dcmread(image_files  + file_)
		name.append(file_)
		position.append(dat.ImagePositionPatient[2])

	file_list = pd.DataFrame(list(zip(name, position)), columns=['filename', 'slice_position'])
	file_list = file_list.sort_values(by=['slice_position'])
	file_list = file_list.reset_index(drop=True)

	ct_image = np.empty((512, 512, len(file_list)))
	for slice_ in range(len(file_list)):
		dat = dcmread("/Users/simongutwein/Downloads/DeepDosePC1/" + file_list['filename'][slice_])
		ct_image[:, :, slice_] = dat.pixel_array

	return ct_image


def upscale(dose3d, ct3d):

	target_width = ct3d.shape[0]
	target_height = int(target_width/dose3d.shape[1]*dose3d.shape[0])
	add = np.zeros((target_width-target_height, target_width, ct3d.shape[2]))
	resized = resize(dose3d, (target_height, target_width, ct3d.shape[2]))
	final_dose = np.concatenate((add, resized), axis=0)

	return final_dose

def get_colormap(cmap, threshold):

	my_cmap = cmap(np.arange(cmap.N))
	invis = np.zeros((1,threshold))
	vis = np.ones((1,256-threshold))
	alpha = np.concatenate((invis, vis), axis=1)
	my_cmap[:, -1] = alpha

	return ListedColormap(my_cmap)

if __name__ == "__main__":

	filepath_3ddose = "/Users/simongutwein/localfolder/EGSnrc/egs_home/dosxyznrc/p_225_5x5_0x0.3ddose"
	filepath_ct_dose = "/Users/simongutwein/localfolder/EGSnrc/egs_home/dosxyznrc/p/"
	filepath_ct_image = "/Users/simongutwein/Downloads/DeepDosePC1/"

	dose_3d = read_3ddose_file(filepath_3ddose)
	ct_3d = get_ct_image(filepath_ct_dose, filepath_ct_image)

	dose_3d_upscaled = upscale(dose_3d, ct_3d)

	with open('beam_xyz.npy', 'wb+') as fout:
		#can be read with np.load(fin)
		np.save(fout, dose_3d_upscaled)
		np.save(fout, ct_3d)
		
	dose_cmap = get_colormap(pl.cm.jet, 10)
	ct_cmap = get_colormap(pl.cm.bone, 7)

	dose_cmap_applied = dose_cmap(dose_3d_upscaled/dose_3d_upscaled.max())
	ct_cmap_applied = ct_cmap(ct_3d/ct_3d.max())

	for slice_ in range(ct_3d.shape[2]):
		plt.imshow(ct_cmap_applied[:, :, slice_])
		plt.imshow(dose_cmap_applied[:, :, slice_], alpha=0.8)
		plt.show()

	

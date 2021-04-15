
from IMPORTS import *
from PIL import Image
from pydicom import dcmread
import os
import pandas as pd
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

#File to read in .3ddose file and plot cross-, inplane and TDV

def dose_distribution_3D(filename):

	with open(filename, 'r') as fout:

		dim = np.array(fout.readline().split()).astype("int")
		x, y, z = np.array(fout.readline().split()).astype(
			"float"), np.array(fout.readline().split()).astype("float"), np.array(fout.readline().split()).astype("float")
		dose_dat = np.array(fout.readline().split()).astype("float")
		dose_dat = dose_dat.reshape(dim[0], dim[1], dim[2], order='F')
	
	return x, y, z, dose_dat


files = os.listdir("/Users/simongutwein/Downloads/DeepDosePC1/")
files.remove('.DS_Store')
pixel = np.empty((512,512,110))
info = []
number = []
for i in range(110): 
	ct = dcmread("/Users/simongutwein/Downloads/DeepDosePC1/" + files[i])
	files.append(files[i])
	number.append(ct.ImagePositionPatient[2])

lst = pd.DataFrame(list(zip(files, number)),columns=['file', 'id'])
lst = lst.sort_values(by=['id'])
lst = lst.reset_index()

for i in range(110):
	dat = dcmread("/Users/simongutwein/Downloads/DeepDosePC1/" + lst['file'][i])
	pixel[:,:,i] = dat.pixel_array

# input("Path:   ")
filename = "/Users/simongutwein/localfolder/EGSnrc/egs_home/dosxyznrc/phantom_file.3ddose"

x, y, z, dose_3D = dose_distribution_3D(filename)

# for i in range(len(z)-1):
# 	plt.imshow(dose_3D[:,:,i])
# 	plt.show()

#%%

cmap = pl.cm.jet

# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))

# Set alpha
zeros = np.zeros((1,20))
ones = np.ones((1,236))
arr = np.concatenate((zeros, ones), axis=1)
my_cmap[:, -1] = arr

# Create new colormap
cm = ListedColormap(my_cmap)

#%%

basewidth = 512
for i in range(dose_3D.shape[2]):
	img = Image.fromarray(np.array((dose_3D[:, :, i]).T/np.array(dose_3D).max()))
	img = cm(img.resize((basewidth, basewidth), Image.ANTIALIAS))
	plt.imshow(pixel[:, :, i], cmap='bone')
	plt.imshow(img, alpha = 0.5)
	plt.show()

# %%

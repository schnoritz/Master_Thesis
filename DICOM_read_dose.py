import pydicom
import matplotlib.pyplot as plt
import numpy as np

#read in dose file for specific field size and plot it optionally
plot = False

dose_path = "/Users/simongutwein/Downloads/Share_Simon/DICOMData/22X22/MRI_Phantom_Reference22X22_Dose.dcm"
ds = pydicom.read_file(dose_path,force=True)

plan_path = "/Users/simongutwein/Downloads/Share_Simon/DICOMData/22X22/MRI_Phantom_Reference22X22.dcm"
plan = pydicom.read_file(plan_path,force=True)

data= ds.pixel_array

if plot == True:
	print(ds.pixel_array[:,0,:])
	for i in range(ds.pixel_array.shape[1]):
		plt.show(block=False)
		plt.imshow(np.log(ds.pixel_array[:,i,:]))
		plt.draw()
		plt.pause(0.001)
		plt.clf()
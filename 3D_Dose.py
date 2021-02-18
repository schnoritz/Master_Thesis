import numpy as np
import matplotlib.pyplot as plt

def dose_distribution_3D(filename):

	file = open(filename)
	dim = np.array(list(map(int, file.readline().split())))
	z, y, x = np.array(list(map(float, file.readline().split()))), np.array(list(map(float, file.readline().split()))) ,np.array(list(map(float, file.readline().split())))
	dose_dat = np.array(list(map(float, file.readline().split())))
	dose_dat = dose_dat.reshape(dim[2], dim[1],  dim[0])

	return x, y, z, dose_dat

def plot_dose_distribution(dose_3D,x ,y ,z):
	
	plt.imshow(dose_3D[:,:,0])
	plt.show()
	pos = 50
	x_center, y_center = np.where(dose_3D[:,:,pos] == dose_3D[:,:,pos].max())
	x_center = x_center[0]
	y_center = y_center[0]

	TDV = dose_3D[x_center,y_center,:]
	IP = dose_3D[:,y_center,pos]
	CP = dose_3D[x_center,:,pos]

	plt.subplot(3, 1, 1)
	plt.plot(z[:-1],TDV[::-1])
	plt.subplot(3, 1, 2)
	plt.plot(x[:-1],IP)
	plt.subplot(3, 1, 3)
	plt.plot(y[:-1],CP)
	plt.show()
	
filename = "/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/water_phantom_2X2.3ddose"

x, y, z, dose_3D = dose_distribution_3D(filename)

plot_dose_distribution(dose_3D, x, y, z)
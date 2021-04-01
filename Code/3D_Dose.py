from imports import *
#File to read in .3ddose file and plot cross-, inplane and TDV

def dose_distribution_3D(filename):

	file = open(filename)
	dim = np.array(list(map(int, file.readline().split())))
	z, y, x = np.array(list(map(float, file.readline().split()))), np.array(list(map(float, file.readline().split()))) ,np.array(list(map(float, file.readline().split())))
	dose_dat = np.array(list(map(float, file.readline().split())))
	dose_dat = dose_dat.reshape(dim[2], dim[1],  dim[0])
	print(dose_dat.shape)
	return x, y, z, dose_dat

def plot_dose_distribution(dose_3D,x ,y ,z):
	
	pos = 90
	plt.imshow(dose_3D[:,pos,:])
	plt.show()
	x_center, y_center = dose_3D.shape[0]//2, dose_3D.shape[2]//2

	TDV = dose_3D[x_center,:, y_center]
	IP = dose_3D[:, pos, y_center]
	CP = dose_3D[x_center, pos, :]

	plt.subplot(3, 1, 1)
	plt.plot(z[:-1],TDV[::-1])
	plt.subplot(3, 1, 2)
	plt.plot(x[:-1],IP)
	plt.subplot(3, 1, 3)
	plt.plot(y[:-1],CP)
	plt.show()

	return
	

filename = "/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/MRI_Phantom_Reference2X2.dcm_1.3ddose"

x, y, z, dose_3D = dose_distribution_3D(filename)

# for i in range(dose_3D.shape[2]):
# 	plt.imshow(dose_3D[:,i,:])
# 	plt.show()
# 	print(i)

plot_dose_distribution(dose_3D, x, y, z)

from imports import *

size = 50
volume = np.ones((size,size,size))
num = int((size-1)/2)
d = 4
volume[num-d:num+d, num-d:num+d, num-d:num+d] = 0

origin_pos = [10,10,-1]

origin_pos_corrected = np.array(origin_pos+np.array([num,num,0]))

dist = [np.sqrt(np.sum(np.square([i,j,k]-origin_pos_corrected))) for i in range(volume.shape[0]) for j in range(volume.shape[1]) for k in range(volume.shape[2])]

dist_map = np.array(dist).reshape((size,size,size))

for i in range(volume.shape[0]):
	plt.imshow(dist_map[i,:,:])
	plt.draw()
	plt.pause(0.01)

radiological_depth = np.zeros(volume.shape)
for i in range(volume.shape[0]):
	for j in range(volume.shape[1]):
		for k in range(volume.shape[2]):
			vector = [i,j,k] - origin_pos_corrected
			x = np.round(np.linspace(i,origin_pos_corrected[0],vector[2]))
			y = np.round(np.linspace(j,origin_pos_corrected[1],vector[2]))
			z = np.round(np.linspace(k,origin_pos_corrected[2],vector[2]))

			x = np.delete(x, np.where(z < 0))
			y = np.delete(y, np.where(z < 0))
			z = np.delete(z, np.where(z < 0))
			
			voxels = np.array([[x[idx],y[idx],z[idx]] for idx in range(len(x))]).astype(int)
			mean_HU_value = np.sum([volume[voxels[i][0],voxels[i][1],voxels[i][2]] for i in range(len(voxels))])
			radiological_depth[i,j,k] = mean_HU_value*dist_map[i,j,k]

for i in range(volume.shape[0]):
	plt.imshow(radiological_depth[i,:,:])
	plt.draw()
	plt.pause(0.1)


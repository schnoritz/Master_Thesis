
#Quelle. https://mateuszbuda.github.io/2017/12/01/brainseg.html

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

file = open(
	"/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/training_data/MR-Linac_model_9x9_0x0.egsinp")

i=0

while True:

	i += 1
	if i == 10000:
		break

	t_line = file.readline()
	if "LEAFRADIUS, CIL" in t_line:
		info = ""
		for j in range(80): 
			info += file.readline()

	if "R0LEAF, Z0LEAF" in t_line:
		JAW_dat = file.readline()
		break

info = np.array(info.replace(',', '').split())
info = info.astype(float)
left_MLC = info[0::3]+41.5
left_offset = -5.4-left_MLC
right_MLC = info[1::3]- 41.5 #oder abs(info[1::3]- 41.5)
right_offset = 5.4-right_MLC
JAW_dat = np.array(JAW_dat.replace(',', '').split())
JAW_dat = JAW_dat.astype(float)
JAW = [JAW_dat[0]]
JAW.append(JAW_dat[1])
JAW = np.array(JAW)

ax = plt.barh(range(len(left_MLC)),left_offset,left=left_MLC)
plt.barh(range(len(left_MLC)),right_offset,left=right_MLC)
ax = plt.gca()
ax.add_patch(ptc.Rectangle((-5.4,-0.4),10.8,39.9+JAW[0],facecolor="r",alpha=0.5))
ax.add_patch(ptc.Rectangle((-5.4,39.5+JAW[1]),10.8,39.9-JAW[1]+0.4, facecolor="r",alpha=0.5))
fig = plt.gcf()
plt.show()
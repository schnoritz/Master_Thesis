#Quelle. https://mateuszbuda.github.io/2017/12/01/brainseg.html

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

file = open("/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/MR-Linac_model_22X22.egsinp")

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

#print(info,JAW_dat)

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
#plt.yticks(np.linspace(0,79,8),np.around(np.linspace(-28.6,28.6,8),decimals=2))
#plt.xticks(np.linspace(-5.4,5.4,10),np.around(np.linspace(-54,54,10),decimals=2))
ax = plt.gca()
ax.add_patch(ptc.Rectangle((-5.4,-0.4),10.8,39.9+JAW[0],facecolor="r",alpha=0.5))
ax.add_patch(ptc.Rectangle((-5.4,39.5+JAW[1]),10.8,39.9-JAW[1]+0.4, facecolor="r",alpha=0.5))
fig = plt.gcf()
plt.show()

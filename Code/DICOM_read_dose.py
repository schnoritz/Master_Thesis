import pydicom
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as ptc

#read in dose file for specific field size and plot it optionally
plot = False

dose_path = "/Users/simongutwein/Downloads/Share_Simon/DICOMData/22X22/MRI_Phantom_Reference22X22_Dose.dcm"
ds = pydicom.read_file(dose_path,force=True)

plan_path = "/Users/simongutwein/Downloads/Share_Simon/DICOMData/22X22/MRI_Phantom_Reference22X22.dcm"
plan = pydicom.read_file(plan_path,force=True)

print(plan)

#Maximale Öffnungsposition ist -110, geschlossen ist -2
# -> Angaben sind in mm -> -110 sind 11cm -> Bei beiden MLC bei -110 und +110 also Feldöffnunf von 220mm 
# -> Also 22cm
MLC = plan.BeamSequence[0].ControlPointSequence[0].BeamLimitingDevicePositionSequence[1].LeafJawPositions
MLC_bottom = np.array(MLC[:80])/10
MLC_Top = np.array(MLC[80:])/10
MLC_Top.astype(float)
MLC_bottom.astype(float)
print("TOP =", len(MLC_bottom), "\nBOTTOM =", len(MLC_Top))
print(MLC_bottom)

field = np.linspace(0,0.715*80,80)

#plt.plot(field,MLC_Top)
#plt.plot(field,MLC_bottom)
plt.bar(np.linspace(0,0.715*80,80),MLC_Top,color="w")
plt.bar(np.linspace(0,0.715*80,80),15-MLC_Top,width=0.6,bottom=MLC_Top)
plt.bar(np.linspace(0,0.715*80,80),MLC_bottom,color="w")
plt.bar(np.linspace(0,0.715*80,80),-15-MLC_bottom,width=0.6,bottom=MLC_bottom)

JAWS = plan.BeamSequence[0].ControlPointSequence[0].BeamLimitingDevicePositionSequence[0].LeafJawPositions
JAW_left = np.array(JAWS[0])
JAW_right = np.array(JAWS[1])
JAW_right.astype(float)
JAW_left.astype(float)
print(JAW_left, JAW_right)

#plt.vlines(57.2/2+JAW_left/10.0,-15,15,color="green")
#plt.vlines(57.2/2+JAW_right/10.0,-15,15,color="green")
ax = plt.gca()
ax.add_patch(ptc.Rectangle((0,-15),57.2/2+JAW_left/10.0,30,facecolor="g",alpha=0.5))
ax.add_patch(ptc.Rectangle((57.2/2+JAW_right/10.0,-15),57.2/2-JAW_right/10.0,30,facecolor="g",alpha=0.5))

plt.axis('equal')
plt.tight_layout()
plt.show()

data= ds.pixel_array

if plot == True:
	print(ds.pixel_array[:,0,:])
	for i in range(ds.pixel_array.shape[1]):
		plt.show(block=False)
		plt.imshow(np.log(ds.pixel_array[:,i,:]))
		plt.draw()
		plt.pause(0.001)
		plt.clf()
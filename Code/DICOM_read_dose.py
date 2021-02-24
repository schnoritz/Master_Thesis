import pydicom
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as ptc
import math

#read in dose file for specific field size and plot it optionally
plot = False

dose_path = "/Users/simongutwein/Downloads/Share_Simon/DICOMData/22X22/MRI_Phantom_Reference22X22_Dose.dcm"
ds = pydicom.read_file(dose_path,force=True)

plan_path = "/Users/simongutwein/Downloads/Share_Simon/DICOMData/15X15/MRI_Phantom_Reference15X15.dcm"
plan = pydicom.read_file(plan_path,force=True)

#print(plan)

#Maximale Öffnungsposition ist -110, geschlossen ist -2
# -> Angaben sind in mm -> -110 sind 11cm -> Bei beiden MLC bei -110 und +110 also Feldöffnunf von 220mm 
# -> Also 22cm
MLC = plan.BeamSequence[0].ControlPointSequence[0].BeamLimitingDevicePositionSequence[1].LeafJawPositions
MLC_bottom = np.array(MLC[:80])/10
MLC_Top = np.array(MLC[80:])/10
MLC_Top.astype(float)
MLC_bottom.astype(float)
MLC = np.array(MLC)
MLC.astype(float)
print("TOP =", len(MLC_bottom), "\nBOTTOM =", len(MLC_Top))
#print(MLC_bottom)

field = np.linspace(0,0.715*80,80)

#plt.plot(field,MLC_Top)
#plt.plot(field,MLC_bottom)
plt.bar(field,MLC_Top,color="w")
plt.bar(field,15-MLC_Top,width=0.6,bottom=MLC_Top)
plt.bar(field,MLC_bottom,color="w")
plt.bar(field,-15-MLC_bottom,width=0.6,bottom=MLC_bottom)

JAWS = plan.BeamSequence[0].ControlPointSequence[0].BeamLimitingDevicePositionSequence[0].LeafJawPositions
JAW_left = np.array(JAWS[0])
JAW_right = np.array(JAWS[1])
JAW_right.astype(float)
JAW_left.astype(float)
#print(JAW_left, JAW_right)

#plt.vlines(57.2/2+JAW_left/10.0,-15,15,color="green")
#plt.vlines(57.2/2+JAW_right/10.0,-15,15,color="green")
ax = plt.gca()
ax.add_patch(ptc.Rectangle((0,-15),57.2/2+JAW_left/10.0,30,facecolor="g",alpha=0.7))
ax.add_patch(ptc.Rectangle((57.2/2+JAW_right/10.0,-15),57.2/2-JAW_right/10.0,30,facecolor="g",alpha=0.7))

plt.axis('equal')
plt.tight_layout()
if plot == True:
	plt.show()

r = 41.5
ssd =143.5
cil = 35.77-0.09

MLC_new = []

for j in range(len(MLC)):
    if j<80:
        if MLC[j]<=0:
            MLC_new.append(((cil + math.sqrt(pow(r,2)-pow(math.cos(abs(MLC[j])*0.1/ssd)*r,2)))*abs(MLC[j])*0.1/ssd + math.cos(abs(MLC[j])*0.1/ssd)*r)*(-1))
        if MLC[j]>0:
            MLC_new.append((-(cil - math.sqrt(pow(r,2)-pow(math.cos(abs(MLC[j])*0.1/ssd)*r,2)))*abs(MLC[j])*0.1/ssd + math.cos(abs(MLC[j])*0.1/ssd)*r)*(-1))    
    
    if j>=80:
        if MLC[j]>=0:
            MLC_new.append((cil + math.sqrt(pow(r,2)-pow(math.cos(abs(MLC[j])*0.1/ssd)*r,2)))*abs(MLC[j])*0.1/ssd + math.cos(abs(MLC[j])*0.1/ssd)*r) 
        if MLC[j]<0:
            MLC_new.append(-(cil - math.sqrt(pow(r,2)-pow(math.cos(abs(MLC[j])*0.1/ssd)*r,2)))*abs(MLC[j])*0.1/ssd + math.cos(abs(MLC[j])*0.1/ssd)*r)     
 
cil_jaw = 44.35-0.09
r_jaw = 13.0

if JAW_left<=0:
    Jaw_new_left=((cil_jaw + math.sqrt(pow(r_jaw,2)-pow(math.cos(abs(JAW_left)*0.1/ssd)*r_jaw,2)))*abs(JAW_left)*0.1/ssd + math.cos(abs(JAW_left)*0.1/ssd)*r_jaw)*(-1)
if JAW_left>0:
    Jaw_new_left=(-(cil_jaw - math.sqrt(pow(r_jaw,2)-pow(math.cos(abs(JAW_left)*0.1/ssd)*r_jaw,2)))*abs(JAW_left)*0.1/ssd + math.cos(abs(JAW_left)*0.1/ssd)*r_jaw)*(-1)   
if JAW_right>=0:
    Jaw_new_right=(cil_jaw + math.sqrt(pow(r_jaw,2)-pow(math.cos(abs(JAW_right)*0.1/ssd)*r_jaw,2)))*abs(JAW_right)*0.1/ssd + math.cos(abs(JAW_right)*0.1/ssd)*r_jaw
if JAW_right<0:
    Jaw_new_right=-(cil_jaw - math.sqrt(pow(r_jaw,2)-pow(math.cos(abs(JAW_right)*0.1/ssd)*r_jaw,2)))*abs(JAW_right)*0.1/ssd + math.cos(abs(JAW_right)*0.1/ssd)*r_jaw   
  
MLC_new = np.array(MLC_new)

if plot == True:
	plt.figure()  
	plt.plot(MLC_new[:80]+41.5)
	plt.plot(MLC_new[80:]-41.5)
	plt.show()

print("MLC Positions Top MLC:",MLC_new[:80].shape, "\nMLC Positions Bottom MLC:", MLC_new[80:].shape,"\nNew Jaw-Position Left Jaw:", Jaw_new_left, "\nNew Jaw-Position Right Jaw:", Jaw_new_right)

data= ds.pixel_array

# if plot == True:
# 	print(ds.pixel_array[:,0,:])
# 	for i in range(ds.pixel_array.shape[1]):
# 		plt.show(block=False)
# 		plt.imshow(np.log(ds.pixel_array[:,i,:]))
# 		plt.draw()
# 		plt.pause(0.001)
# 		plt.clf()
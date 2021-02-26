import pydicom
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as ptc
import math

def calcNewMLC(old_MLC, radius=41.5, ssd=143.5, cil=35.77-0.09):
	
	new_MLC = []

	for j in range(len(old_MLC)):
		if j<80:
			if old_MLC[j]<=0:
				new_MLC.append(((cil + math.sqrt(pow(radius,2)-pow(math.cos(abs(old_MLC[j])*0.1/ssd)*radius,2)))*abs(old_MLC[j])*0.1/ssd + math.cos(abs(old_MLC[j])*0.1/ssd)*radius)*(-1))
			if old_MLC[j]>0:
				new_MLC.append((-(cil - math.sqrt(pow(radius,2)-pow(math.cos(abs(old_MLC[j])*0.1/ssd)*radius,2)))*abs(old_MLC[j])*0.1/ssd + math.cos(abs(old_MLC[j])*0.1/ssd)*radius)*(-1))    
		if j>=80:
			if old_MLC[j]>=0:
				new_MLC.append((cil + math.sqrt(pow(radius,2)-pow(math.cos(abs(old_MLC[j])*0.1/ssd)*radius,2)))*abs(old_MLC[j])*0.1/ssd + math.cos(abs(old_MLC[j])*0.1/ssd)*radius) 
			if old_MLC[j]<0:
				new_MLC.append(-(cil - math.sqrt(pow(radius,2)-pow(math.cos(abs(old_MLC[j])*0.1/ssd)*radius,2)))*abs(old_MLC[j])*0.1/ssd + math.cos(abs(old_MLC[j])*0.1/ssd)*radius)

	return np.array(new_MLC)

def calcNewJAW(old_JAWS, cil=44.35-0.09, radius=13.0, ssd=143.5):

	new_JAWS = np.zeros(2)

	if old_JAWS[0]<=0:
	    new_JAWS[0]=((cil + math.sqrt(pow(radius,2)-pow(math.cos(abs(old_JAWS[0])*0.1/ssd)*radius,2)))*abs(old_JAWS[0])*0.1/ssd + math.cos(abs(old_JAWS[0])*0.1/ssd)*radius)*(-1)
	if old_JAWS[0]>0:
	    new_JAWS[0]=(-(cil - math.sqrt(pow(radius,2)-pow(math.cos(abs(old_JAWS[0])*0.1/ssd)*radius,2)))*abs(old_JAWS[0])*0.1/ssd + math.cos(abs(old_JAWS[0])*0.1/ssd)*radius)*(-1)   
	if old_JAWS[1]>=0:
	    new_JAWS[1]=(cil + math.sqrt(pow(radius,2)-pow(math.cos(abs(old_JAWS[1])*0.1/ssd)*radius,2)))*abs(old_JAWS[1])*0.1/ssd + math.cos(abs(old_JAWS[1])*0.1/ssd)*radius
	if old_JAWS[1]<0:
	    new_JAWS[1]=-(cil - math.sqrt(pow(radius,2)-pow(math.cos(abs(old_JAWS[1])*0.1/ssd)*radius,2)))*abs(old_JAWS[1])*0.1/ssd + math.cos(abs(old_JAWS[1])*0.1/ssd)*radius   
	  
	return new_JAWS

#read in dose file for specific field size and plot it optionally
plot = True

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
JAWS = np.array(JAWS)
print(JAWS)

new_MLCs = calcNewMLC(MLC)
new_JAWs = calcNewJAW(JAWS)
#print(new_MLCs)
#print(new_JAWs)
#plt.vlines(57.2/2+JAW_left/10.0,-15,15,color="green")
#plt.vlines(57.2/2+JAW_right/10.0,-15,15,color="green")

ax = plt.gca()
ax.add_patch(ptc.Rectangle((0,-15),57.2/2+JAWS[0]/10.0,30,facecolor="g",alpha=0.7))
ax.add_patch(ptc.Rectangle((57.2/2+JAWS[1]/10.0,-15),57.2/2-JAWS[1]/10.0,30,facecolor="g",alpha=0.7))

plt.axis('equal')
plt.tight_layout()
if plot == True:
	plt.show()

#MLC_new = np.array(MLC_new)

if plot == True:
	plt.figure()  
	plt.plot(new_MLCs[:80]+41.5)
	plt.plot(new_MLCs[80:]-41.5)
	plt.show()

print("MLC Positions Top MLC:",new_MLCs[:80].shape, "\nMLC Positions Bottom MLC:", new_MLCs[80:].shape,"\nNew Jaw-Position Left Jaw:", new_JAWs[0], "\nNew Jaw-Position Right Jaw:", new_JAWs[1])

data= ds.pixel_array

# if plot == True:
# 	print(ds.pixel_array[:,0,:])
# 	for i in range(ds.pixel_array.shape[1]):
# 		plt.show(block=False)
# 		plt.imshow(np.log(ds.pixel_array[:,i,:]))
# 		plt.draw()
# 		plt.pause(0.001)
# 		plt.clf()
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as ptc
import math

def calcNewMLC(old_MLC, radius=41.5, ssd=143.5, cil=35.77-0.09):
	
	new_MLC = np.zeros((2,80))

	for j in range(2):
		for i in range(80):
			if j == 0:
				if old_MLC[j,i]<=0:
					new_MLC[j,i] = (((cil + math.sqrt(pow(radius,2)-pow(math.cos(abs(old_MLC[j,i])*0.1/ssd)*radius,2)))*abs(old_MLC[j,i])*0.1/ssd + math.cos(abs(old_MLC[j,i])*0.1/ssd)*radius)*(-1))
				if old_MLC[j,i]>0:
					new_MLC[j,i] = ((-(cil - math.sqrt(pow(radius,2)-pow(math.cos(abs(old_MLC[j,i])*0.1/ssd)*radius,2)))*abs(old_MLC[j,i])*0.1/ssd + math.cos(abs(old_MLC[j,i])*0.1/ssd)*radius)*(-1))    
			else:
				if old_MLC[j,i]>=0:
					new_MLC[j,i] = ((cil + math.sqrt(pow(radius,2)-pow(math.cos(abs(old_MLC[j,i])*0.1/ssd)*radius,2)))*abs(old_MLC[j,i])*0.1/ssd + math.cos(abs(old_MLC[j,i])*0.1/ssd)*radius) 
				if old_MLC[j,i]<0:
					new_MLC[j,i] = (-(cil - math.sqrt(pow(radius,2)-pow(math.cos(abs(old_MLC[j,i])*0.1/ssd)*radius,2)))*abs(old_MLC[j,i])*0.1/ssd + math.cos(abs(old_MLC[j,i])*0.1/ssd)*radius)

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

def plot_MLC_field(MLC_dat, JAWS_dat):

	MLC_dat = MLC_dat/10
	field = np.linspace(0,0.715*80,80)
	plt.bar(field,MLC_dat[1,:],color="w")
	plt.bar(field,15-MLC_dat[1,:],width=0.6,bottom=MLC_dat[1,:])
	plt.bar(field,MLC_dat[0,:],color="w")
	plt.bar(field,-15-MLC_dat[0,:],width=0.6,bottom=MLC_dat[0,:])

	ax = plt.gca()
	ax.add_patch(ptc.Rectangle((0,-15),57.2/2+JAWS_dat[0]/10.0,30,facecolor="g",alpha=0.7))
	ax.add_patch(ptc.Rectangle((57.2/2+JAWS_dat[1]/10.0,-15),57.2/2-JAWS_dat[1]/10.0,30,facecolor="g",alpha=0.7))

	plt.axis('equal')
	plt.tight_layout()
	plt.show()

	return

#read in dose file for specific field size and plot it optionally
plot = True

dose_path = "/Users/simongutwein/Downloads/Share_Simon/DICOMData/22X22/MRI_Phantom_Reference22X22_Dose.dcm"
ds = pydicom.read_file(dose_path,force=True)

plan_path = "/Users/simongutwein/Downloads/Share_Simon/DICOMData/10X10/MRI_Phantom_Reference10X10.dcm"
plan = pydicom.read_file(plan_path,force=True)

#print(plan)
#Maximale Öffnungsposition ist -110, geschlossen ist -2
# -> Angaben sind in mm -> -110 sind 11cm -> Bei beiden MLC bei -110 und +110 also Feldöffnunf von 220mm 
# -> Also 22cm
MLC = np.array(plan.BeamSequence[0].ControlPointSequence[0].BeamLimitingDevicePositionSequence[1].LeafJawPositions).astype(float)
MLC = MLC.reshape((2,80))
print("TOP =", len(MLC[0,:]), "\nBOTTOM =", len(MLC[1,:]))
print(MLC)

JAWS = plan.BeamSequence[0].ControlPointSequence[0].BeamLimitingDevicePositionSequence[0].LeafJawPositions
JAWS = np.array(JAWS)

new_MLCs = calcNewMLC(MLC)
new_JAWs = calcNewJAW(JAWS)
print(new_MLCs)
#print(new_JAWs)

if plot == True:
	plot_MLC_field(MLC, JAWS)

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
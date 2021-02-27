import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

def plot_MLC_field(MLC_dat, JAWS_dat):

	MLC_dat = MLC_dat/10
	field = np.linspace(0,0.715*80,80)
	plt.bar(field,MLC_dat[1,:],color="w")
	plt.bar(field,MLC_dat[0,:],color="w")
	plt.bar(field,15-MLC_dat[1,:],width=0.6,bottom=MLC_dat[1,:])
	plt.bar(field,-15-MLC_dat[0,:],width=0.6,bottom=MLC_dat[0,:])

	ax = plt.gca()
	ax.add_patch(ptc.Rectangle((0,-15),57.2/2+JAWS_dat[0]/10.0,30,facecolor="g",alpha=0.7))
	ax.add_patch(ptc.Rectangle((57.2/2+JAWS_dat[1]/10.0,-15),57.2/2-JAWS_dat[1]/10.0,30,facecolor="g",alpha=0.7))

	plt.axis('equal')
	plt.tight_layout()
	plt.show()

	plt.draw()
	plt.pause(0.001)

	return

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

	
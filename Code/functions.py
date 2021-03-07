from imports import *

def plot_mlc_field(MLC_dat, JAWS_dat):

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

	return

def calc_new_mlc(old_MLC, radius=41.5, ssd=143.5, cil=35.77-0.09):
	
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

def calc_new_jaw(old_JAWS, cil=44.35-0.09, radius=13.0, ssd=143.5):

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

def create_field_parameters(size=None, translation=None):
	"""creates parameters for the field, size and offset can be randomly generated. 
	if size and translation are given this code produces the wanted field parameters
	size and translation are tuples with x-dim and y-dim of field and translation
	"""
	if size is None:
		fieldsize = np.array([random.randint(2,57), random.randint(2,22)])

	else:
		fieldsize = np.array([size[0], size[1]])

	#print("FIELDSIZE:", fieldsize)

	if translation is None:
		max_x_translation = 28.5-fieldsize[0]/2
		max_y_translation = 11-fieldsize[1]/2
		#print("MAXMIMUM OFFSET VALUES:", max_x_translation, max_y_translation)
		#accounting for boundary conditions that field translation can't be so that field lies outside of maximum field
		#offset = [random.uniform(-max_x_translation, max_x_translation), random.uniform(-max_y_translation, max_y_translation)]
		offset = [random.randint(-np.floor(max_x_translation), np.floor(max_x_translation)), random.randint(-np.floor(max_y_translation), np.floor(max_y_translation))]
	else:
		offset = [translation[0],translation[1]]

	return fieldsize, offset

def calculate_mlc_positions(fieldsize, offset):
	
	MLC = np.zeros((2,80))
	dx, dy = offset[0], offset[1]
	central_leafes = [np.floor(dx/0.715).astype(int) + 40, np.floor(dx/0.715).astype(int) + 41]
	#print("CENTRAL_LEAFES:", central_leafes)

	#anzahl der leafes die noch hinzugefügt werden sollen, in dem Fall hier 4 Leafes mehr als benötigt werden.
	count_leafes = np.ceil(fieldsize[0]/0.715).astype(int)+2
	
	#Leaf anzahl immer gerade machen
	if count_leafes%2 != 0:
		count_leafes += 1
	#print("COUT_LEAFES:", count_leafes)

	MLC[0,:] += dy - 0.2 #Spalt in der Mitte erzeugen, 0,2mm breite
	MLC[1,:] += dy + 0.2

	if int(central_leafes[0]-count_leafes/2)-1 < 0 and int(central_leafes[1]+count_leafes/2) < 80:

		MLC[0,0:central_leafes[0]] -= fieldsize[1]/2
		MLC[1,0:central_leafes[0]] += fieldsize[1]/2
		MLC[0,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] -= fieldsize[1]/2
		MLC[1,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] += fieldsize[1]/2
	
	elif int(central_leafes[1]+count_leafes/2) >= 80 and int(central_leafes[0]-count_leafes/2)-1 > 0:
		
		MLC[0,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] -= fieldsize[1]/2
		MLC[1,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] += fieldsize[1]/2
		MLC[0,central_leafes[1]-1:80] -= fieldsize[1]/2
		MLC[1,central_leafes[1]-1:80] += fieldsize[1]/2
	
	elif int(central_leafes[1]+count_leafes/2) >= 80 and int(central_leafes[0]-count_leafes/2)-1 < 0:

		MLC[0,0:central_leafes[0]] -= fieldsize[1]/2
		MLC[1,0:central_leafes[0]] += fieldsize[1]/2
		MLC[0,central_leafes[1]-1:80] -= fieldsize[1]/2
		MLC[1,central_leafes[1]-1:80] += fieldsize[1]/2

	else:

		MLC[0,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] -= fieldsize[1]/2
		MLC[1,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] += fieldsize[1]/2
		MLC[0,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] -= fieldsize[1]/2
		MLC[1,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] += fieldsize[1]/2

	#JAW positionen berechnen
	JAWS = np.array([dx-fieldsize[0]/2, dx+fieldsize[0]/2])*10.0

	return MLC, JAWS

def read_template():

	f = open("/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/TEMPLATE.egsinp")
	template_text = f.read()
	template_text = template_text.split("\n")
	idx = [i for i, x in enumerate(template_text) if x=="###HIER ERSETZEN###"]
	return template_text, idx


	
def doses_of_dcm():

	sizes = ["2x2", "3x3", "5x5", "10x10", "15x15", "22x22", "40x22", "57x22"]

	for size in sizes:
		file_path = "/Users/simongutwein/Downloads/Share_Simon/DICOMData/" + size + "/MRI_Phantom_Reference" + size + "_Dose.dcm"
		ds = pydicom.read_file(file_path,force=True)
		print(size)
		dose = ds.pixel_array
		print(dose.shape)

		#dose dimensions = (101, 151, 219)
		#dimensionalität in cm bestimmen -> damit fielsize zu volumen gemapped werden kann 

def scatter_hist_2D_data(shapes):

	x_fieldsize = []
	y_fieldsize = []
	for i in shapes:
		x_fieldsize.append(i[0][0])
		y_fieldsize.append(i[0][1])

	x_translation = []
	y_translation = []
	for i in shapes:
		x_translation.append(i[1][0])
		y_translation.append(i[1][1])

	plot_scatt_hist(x_fieldsize,y_fieldsize)
	plot_scatt_hist(x_translation,y_translation)

def plot_scatt_hist(x,y):
	fig = plt.figure(figsize=(8,8))
	gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
	                  left=0.1, right=0.9, bottom=0.1, top=0.9,
	                  wspace=0.05, hspace=0.05)

	scat = fig.add_subplot(gs[1, 0])
	top = fig.add_subplot(gs[0, 0], sharex=scat)
	right = fig.add_subplot(gs[1, 1], sharey=scat)
	scat.scatter(x,y, marker="x", s=3)
	right.hist(y, orientation='horizontal', bins=len(np.unique(y)))
	top.hist(x, bins=len(np.unique(x)))
	top.tick_params(axis="x", labelbottom=False)
	right.tick_params(axis="y", labelleft=False)
	plt.show()


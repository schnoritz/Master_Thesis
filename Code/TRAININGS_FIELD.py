"""

idee: Feld ist definiert durch mittelpunkt und seine Dimensionen
JAWS: verschiebung der Koordinaten der JAWS um dx (dx ist hier die Verschiebung des Feldes), 
dann noch addieren bzw subtrahieren der Feldgröß

JAWS in Nullposition haben 0 und 0, für erine Verschiebung um zB 2 in negative x Richtung
und eine Feldgröße von 5x5 cm würden die JAWS folgende Werte haben:
Linkes JAW: 0-2-5 und rechtes JAW = 0-2+5
Also dx-Feldgröße und dx+Feldgröße

MLC: verschiebung um dy wird einfach auf leafes übertragen, evtl einfach das gesamte feld shiften
5 cm shiften wie viele Leafes sind das? 5%leafebreite? 5 / 0.715 = 6.993

2x2 = 5.72 -> 8 
10x10 = 12.87 -> 18 Leaves
15x15 = 18.59 -> 26 Leaves
22x22 = 25.74 -> 36 Leaves

Immer 5 leafes extra

0 bis 57 cm
Verscheibung in X-Richtung welches Leaf ist Zentral?
+dx dann ist rechte Seite Zentral, bei -dx ist linke Seite zentral:
dx -> dx/Leafbreite -> runden -> dann Feldgröße/leafbreite -> runden -> +4 
-> auf jeder seite davon so viele Leaves öffnen 

Beispiel: 4 verschiebung in x richtung bei 5x5er Feld
4/0.715 = 5.5944 -> runden: 6 - also sind 5 und 6 Zentral
5/0.715 = 6.993 -> runden auf gerade Zahl: 8 + 4 -> also 12 Leaves evlt immer auf gerade Leaf Zahl aufrunden. 

also hier leaf gesamt mit central leaf sind es 12 Leafes, also leafes 39 bis 45 und 46 bis 51

"""

"""
To-Do:
[✔] klären wie es aussieht mit den restlichen leafes sind die wichtig oder nicht
[✔] (Quasi fertig, function ist in functions.py) MLC positionen in template .egsnip file einspeichern
[✔] evtl. Output erstellen damit ich sehen kann welche Felder wie fot vorgekommen sind
[✔] JAW Positionen für .egsinp File wirken komisch. 
[✔] Leaves sind im Fled 0.2 zu weit ausgefahren
[✔] klasse für trainingsdata fertigstellen
[✔] batchweise erzeugung und speicherung
[✔] testen ob dimensions und translationskombination schon vorgekommen ist. 
[✔] möglichkeit bieten bei class auch ein festdefniniertes Feld zu erstellen
"""

from IMPORTS import *

class trainingData():
	'''
	fieldsize = desired fieldsize
	translation = desired translation 
	max_fieldsize = the biggest possible fieldsize
	max_generated_fieldsize = if fieldsize is chosen to be random, max_generated_fieldsize is the biggest fieldsize generated
	num_leafes = number of leafes
	'''

	def __init__(self, fieldsize=None, translation=None, max_fieldsize=None, max_generated_fieldsize=None, num_leafes=80, distribution="gaussian"):

		#check if max fieldsize is given or not, otherwise MR-Linac fieldsize is used
		if max_fieldsize is None:
			self.max_fieldsize = (57, 22)
		else:
			self.max_fieldsize =  max_fieldsize #[x_max, y_max]

		if fieldsize is not None and (fieldsize[0] > self.max_fieldsize[0] or fieldsize[1] > self.max_fieldsize[1]):
			raise ValueError("fieldsize is bigger than max_fieldsize")
		#check if max_generated_fieldsize is passeed, if not max_generated_fieldsize is set to max_fieldsize
		if max_generated_fieldsize is None:
			self.max_generated_fieldsize = self.max_fieldsize
		else:
			self.max_generated_fieldsize = max_generated_fieldsize

		#check if max_generated_fieldsize bigger than max_fieldsize
		if self.max_generated_fieldsize[0] > self.max_fieldsize[0] or self.max_generated_fieldsize[1] > self.max_fieldsize[1]:
			raise ValueError("Passed max_generated_fieldsize value bigger than max_fieldsize.")

		#parameter assignment
		self.num_leafes = num_leafes
		self.leaf_width = self.max_fieldsize[0]/self.num_leafes
		self.fieldsize, self.translation = self.create_field_parameters(fieldsize, translation, distribution)
		self.MLC_iso = self.create_mlc_positions()
		self.JAW_iso = self.create_jaw_positions()
		self.MLC_egsinp = self.calculate_new_mlc()
		self.JAW_egsinp = self.calculate_new_jaw()

	def create_field_parameters(self, fieldsize=None, translation=None, distribution="gaussian"):

		#if fielsize and translation are passed, check if they fit the max_fieldsize
		if fieldsize is not None and translation is not None:
			if fieldsize[0]/2 + abs(translation[0]) > self.max_fieldsize[0]/2 or fieldsize[1]/2 + abs(translation[1]) > self.max_fieldsize[1]/2:
				raise ValueError("fieldsize and translation dont fit inside maximum_fieldsize")

		#if fieldsize and translation is not passed use full spectrum of fieldsizes and translations
		if fieldsize is None and translation is None:

			if distribution == "gaussian":

				fieldsize = self.gaussian_fieldsize(self.max_generated_fieldsize)

				#check for boundary conditions
				max_x_translation = self.max_fieldsize[0]/2 - fieldsize[0]/2
				max_y_translation = self.max_fieldsize[1]/2 - fieldsize[1]/2
				translation = self.gaussian_translation(max_x_translation, max_y_translation)

			elif distribution == "random":

				fieldsize = ([random.randint(2,self.max_generated_fieldsize[0]), random.randint(2,self.max_generated_fieldsize[1])])
				
				#check for boundary conditions
				max_x_translation = self.max_fieldsize[0]/2 - fieldsize[0]/2
				max_y_translation = self.max_fieldsize[1]/2 - fieldsize[1]/2
				translation = [random.randint(-np.floor(max_x_translation), np.floor(max_x_translation)), random.randint(-np.floor(max_y_translation), np.floor(max_y_translation))]
				#translation = [random.uniform(-max_x_translation, max_x_translation), random.uniform(-max_y_translation, max_y_translation)]

			else:
				raise ValueError(f"{distribution} is not a viable distribution parameter. Choose from \"gaussian\" or \"random\"")
		
		#check if one of paramters is given
		if fieldsize is None and translation is not None:

			if distribution == "gaussian":
				fieldsize = self.gaussian_fieldsize(self.max_generated_fieldsize)

			elif distribution == "random":
				fieldsize = ([random.randint(2,int(self.max_generated_fieldsize[0]/2)-translation[0]), random.randint(2,int(self.max_generated_fieldsize[1]/2)-translation[1])])

			else:
				raise ValueError(f"{distribution} is not a viable distribution parameter. Choose from \"gaussian\" or \"random\"")

		if fieldsize is not None and translation is None:

			max_x_translation = self.max_fieldsize[0]/2 - fieldsize[0]/2
			max_y_translation = self.max_fieldsize[1]/2 - fieldsize[1]/2

			if distribution == "gaussian":
				translation = self.gaussian_translation(max_x_translation, max_y_translation)

			elif distribution == "random":
				translation = [random.randint(-np.floor(max_x_translation), np.floor(max_x_translation)), random.randint(-np.floor(max_y_translation), np.floor(max_y_translation))]

			else:
				raise ValueError(f"{distribution} is not a viable distribution parameter. Choose from \"gaussian\" or \"random\"")


		return fieldsize, translation

	@staticmethod
	def gaussian_fieldsize(max_generated_fieldsize):
		returnable = False

		while returnable == False:
			fieldsize = [round(abs(random.gauss(0,max_generated_fieldsize[0]/3))) ,round(abs(random.gauss(0,max_generated_fieldsize[1]/3)))]

			if fieldsize[0] >= 2 and fieldsize[1] >= 2 and fieldsize[0] <= max_generated_fieldsize[0] and fieldsize[1] <= max_generated_fieldsize[1]:
				returnable = True
	
		return list(map(int, fieldsize))

	@staticmethod
	def gaussian_translation(max_x_translation, max_y_translation):
		
		returnable = False

		while returnable == False:
			translation = [np.floor(random.gauss(0,max_x_translation/3)), np.floor(random.gauss(0,max_y_translation/3))]

			if abs(translation[0]) <= max_x_translation and abs(translation[1]) <= max_y_translation:
				returnable = True
	
		return list(map(int, translation))

	def create_mlc_positions(self):

		MLC = np.zeros((2,self.num_leafes))
		dx, dy = self.translation[0], self.translation[1]
		field_extend = np.array([dx-float(self.fieldsize[0]/2), dx+float(self.fieldsize[0]/2)])
		min_leafes = np.ceil((self.fieldsize[0]/2)/0.7)+2

		if self.translation[0] > 0:
			if self.translation[0] > -0.3 and self.translation[0] < 0.3:
				num_left, num_left = min_leafes, min_leafes	
			else:
				if np.round((abs(self.translation[0])-0.3)%0.7,4) == 0:
					num_left = min_leafes-1-np.ceil(np.round((abs(self.translation[0])-0.3)/0.7,4))
				else:
					num_left = min_leafes-np.ceil(np.round((abs(self.translation[0])-0.3)/0.7,4))
				if self.translation[0] < 0.4:
					num_right = min_leafes
				else:
					if np.round((abs(self.translation[0])-0.4) % 0.7,4) == 0:
						num_right = min_leafes +1 +np.ceil(np.round((abs(self.translation[0])-0.4)/0.7,4))
					else: 
						num_right = min_leafes+np.ceil(np.round((abs(self.translation[0])-0.4)/0.7,4))
		else:
			if self.translation[0] > -0.3 and self.translation[0] < 0.3:
				num_left, num_right = min_leafes, min_leafes
			else:
				if np.round((abs(self.translation[0])-0.3) % 0.7, 4) == 0:
					num_right = min_leafes-1-np.ceil(np.round((abs(self.translation[0])-0.3)/0.7,4))
				else:
					num_right = min_leafes-np.ceil(np.round((abs(self.translation[0])-0.3)/0.7,4))

				if self.translation[0] < 0.4 and self.translation[0] > -0.4:
					num_left = min_leafes
				else:
					if np.round((abs(self.translation[0])-0.4) % 0.7, 4) == 0:
						num_left = min_leafes +1 +np.ceil(np.round((abs(self.translation[0])-0.4)/0.7,4))
					else:
						num_left = min_leafes+np.ceil(np.round((abs(self.translation[0])-0.4)/0.7,4))

		open_leafes = np.array([int(self.num_leafes/2)-num_left+1, int(self.num_leafes/2)+num_right]).astype('int32')

		# #create small gap 
		MLC[0, :] += dy - 0.2
		MLC[1, :] += dy + 0.2

		#check for boundary conditions
		return self.check_boundary_conditions(MLC, open_leafes)

	def create_jaw_positions(self):

		#calculate the new positions defined by the fieldsize
		return np.array([self.translation[0] - self.fieldsize[0]/2, self.translation[0] + self.fieldsize[0]/2])

	def calculate_new_mlc(self, radius=41.5, ssd=143.5, cil=35.77-0.09):
		
		MLC_egsinp = np.zeros((2, self.num_leafes))

		#calculate new MLC Positions for egsinp 
		for j in range(2):
			for i in range(self.num_leafes):
				if j == 0:
					if self.MLC_iso[j,i] <= 0:
						MLC_egsinp[j,i] = (((cil + math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.MLC_iso[j,i]*10)*0.1/ssd)*radius, 2)))*abs(self.MLC_iso[j,i]*10)*0.1/ssd + math.cos(abs(self.MLC_iso[j,i]*10)*0.1/ssd)*radius)*(-1))
					if self.MLC_iso[j,i]*10 > 0:
						MLC_egsinp[j,i] = ((-(cil - math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.MLC_iso[j,i]*10)*0.1/ssd)*radius, 2)))*abs(self.MLC_iso[j,i]*10)*0.1/ssd + math.cos(abs(self.MLC_iso[j,i]*10)*0.1/ssd)*radius)*(-1))    
				else:
					if self.MLC_iso[j,i]*10 >= 0:
						MLC_egsinp[j,i] = ((cil + math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.MLC_iso[j,i]*10)*0.1/ssd)*radius, 2)))*abs(self.MLC_iso[j,i]*10)*0.1/ssd + math.cos(abs(self.MLC_iso[j,i]*10)*0.1/ssd)*radius) 
					if self.MLC_iso[j,i]*10 < 0:
						MLC_egsinp[j,i] = (-(cil - math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.MLC_iso[j,i]*10)*0.1/ssd)*radius, 2)))*abs(self.MLC_iso[j,i]*10)*0.1/ssd + math.cos(abs(self.MLC_iso[j,i]*10)*0.1/ssd)*radius)

		return MLC_egsinp

	def calculate_new_jaw(self, cil=44.35-0.09, radius=13.0, ssd=143.5):
		
		new_JAWS = np.zeros(2)

		#calculate new JAW Positions for egsinp 
		if self.JAW_iso[0] <= 0:
			new_JAWS[0] = ((cil + math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.JAW_iso[0]*10)*0.1/ssd)*radius, 2)))*abs(self.JAW_iso[0]*10)*0.1/ssd + math.cos(abs(self.JAW_iso[0]*10)*0.1/ssd)*radius)*(-1)
		if self.JAW_iso[0] > 0:
			new_JAWS[0] = (-(cil - math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.JAW_iso[0]*10)*0.1/ssd)*radius, 2)))*abs(self.JAW_iso[0]*10)*0.1/ssd + math.cos(abs(self.JAW_iso[0]*10)*0.1/ssd)*radius)*(-1)   
		if self.JAW_iso[1] >= 0:
			new_JAWS[1] = (cil + math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.JAW_iso[1]*10)*0.1/ssd)*radius, 2)))*abs(self.JAW_iso[1]*10)*0.1/ssd + math.cos(abs(self.JAW_iso[1]*10)*0.1/ssd)*radius
		if self.JAW_iso[1] < 0:
			new_JAWS[1] =- (cil - math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.JAW_iso[1]*10)*0.1/ssd)*radius, 2)))*abs(self.JAW_iso[1]*10)*0.1/ssd + math.cos(abs(self.JAW_iso[1]*10)*0.1/ssd)*radius   

		return new_JAWS

	def check_boundary_conditions(self, MLC, open_leafes):

		

		#check for boundary conditions to move MLC accordingly
		# if int(central_leafes[0] - count_leafes/2) - 1 < 0 and int(central_leafes[1] + count_leafes/2) < self.num_leafes:
		# 	MLC[0, 0 : central_leafes[0]] -= self.fieldsize[1]/2 - 0.2
		# 	MLC[1, 0 : central_leafes[0]] += self.fieldsize[1]/2 - 0.2
		# 	MLC[0, central_leafes[1] - 1 : int(central_leafes[1] + count_leafes/2)] -= self.fieldsize[1]/2 - 0.2
		# 	MLC[1, central_leafes[1] - 1 : int(central_leafes[1] + count_leafes/2)] += self.fieldsize[1]/2 - 0.2
		

		# elif int(central_leafes[1] + count_leafes/2) >= self.num_leafes and int(central_leafes[0] - count_leafes/2) - 1 > 0:
			
		# 	MLC[0, int(central_leafes[0] - count_leafes/2) - 1 : central_leafes[0]] -= self.fieldsize[1]/2 - 0.2
		# 	MLC[1, int(central_leafes[0] - count_leafes/2) - 1 : central_leafes[0]] += self.fieldsize[1]/2 - 0.2
		# 	MLC[0, central_leafes[1] - 1 : self.num_leafes] -= self.fieldsize[1]/2 - 0.2
		# 	MLC[1, central_leafes[1] - 1 : self.num_leafes] += self.fieldsize[1]/2 - 0.2
		

		# elif int(central_leafes[1] + count_leafes/2) >= self.num_leafes and int(central_leafes[0] - count_leafes/2) - 1 < 0:
			
		# 	MLC[0, 0 : central_leafes[0]] -= self.fieldsize[1]/2 - 0.2
		# 	MLC[1, 0 : central_leafes[0]] += self.fieldsize[1]/2 - 0.2
		# 	MLC[0, central_leafes[1] - 1 : self.num_leafes] -= self.fieldsize[1]/2 - 0.2
		# 	MLC[1, central_leafes[1] - 1 : self.num_leafes] += self.fieldsize[1]/2 - 0.2

		# else:

		# 	MLC[0, int(central_leafes[0] - count_leafes/2) - 1 : central_leafes[0]] -= self.fieldsize[1]/2 - 0.2
		# 	MLC[1, int(central_leafes[0] - count_leafes/2) - 1 : central_leafes[0]] += self.fieldsize[1]/2 - 0.2
		# 	MLC[0, central_leafes[1] - 1 : int(central_leafes[1] + count_leafes/2)] -= self.fieldsize[1]/2 - 0.2
		# 	MLC[1, central_leafes[1] - 1 : int(central_leafes[1] + count_leafes/2)] += self.fieldsize[1]/2 - 0.2

		MLC[0,open_leafes[0]-1:open_leafes[1]] = self.translation[1]-self.fieldsize[1]/2
		MLC[1, open_leafes[0]-1:open_leafes[1]] = self.translation[1]+self.fieldsize[1]/2

		return MLC	

	def plot_mlc(self):

		MLC = self.MLC_iso
		field = np.linspace(-self.max_fieldsize[0]/2, self.max_fieldsize[0]/2, self.num_leafes)
		plt.bar(field,MLC[1, :], color="w")
		plt.bar(field,MLC[0, :], color="w")
		plt.bar(field, self.max_fieldsize[1]/2 + 3 - MLC[1,:], width=self.leaf_width -0.1, bottom=MLC[1,:], color="chocolate")
		plt.bar(field, -self.max_fieldsize[1]/2 - 3 -MLC[0,:], width=self.leaf_width -0.1, bottom=MLC[0,:], color="sandybrown")

		ax = plt.gca()
		ax.add_patch(ptc.Rectangle((-self.max_fieldsize[0]/2, -self.max_fieldsize[1]/2 - 3), self.max_fieldsize[0]/2 + self.JAW_iso[0], self.max_fieldsize[1] + 6, facecolor="midnightblue", alpha=0.6))
		ax.add_patch(ptc.Rectangle((self.JAW_iso[1], -self.max_fieldsize[1]/2 - 3), self.max_fieldsize[0]/2 - self.JAW_iso[1], self.max_fieldsize[1] + 6, facecolor="midnightblue", alpha=0.6))

		plt.hlines([self.max_fieldsize[1]/2, -self.max_fieldsize[1]/2], xmin=-self.max_fieldsize[0]/2, xmax=self.max_fieldsize[0]/2, colors="black")
		plt.vlines([-self.max_fieldsize[0]/2, self.max_fieldsize[0]/2], ymin=-self.max_fieldsize[1]/2, ymax=self.max_fieldsize[1]/2, colors="black")
		plt.vlines([0, 0], ymin=-14, ymax=14, colors="black", linestyles='dashed', lw=1)
		plt.hlines([0, 0], xmin=-28.5, xmax=28.5, colors="black", linestyles='dashed', lw=1)


		plt.text(-self.max_fieldsize[0]/2, -self.max_fieldsize[1]/2 - 5, f"Fieldsize: [{self.fieldsize[0]} X {self.fieldsize[1]}]")
		plt.text(-self.max_fieldsize[0]/2, -self.max_fieldsize[1]/2 - 7, f"Offset: [{self.translation[0]} X {self.translation[1]}]")
		#plt.annotate(text='', xy=(self.max_fieldsize[0]/2+self.translation[0],self.translation[1]+self.fieldsize[1]/2), xytext=(self.max_fieldsize[0]/2+self.translation[0], self.translation[1]-self.fieldsize[1]/2), arrowprops=dict(arrowstyle='<|-|>'))
		#plt.annotate(text='', xy=(self.max_fieldsize[0]/2+self.translation[0]-self.fieldsize[0]/2 ,self.translation[1]), xytext=(self.max_fieldsize[0]/2+self.translation[0]+self.fieldsize[0]/2, self.translation[1]), arrowprops=dict(arrowstyle='<|-|>'))
		plt.plot(self.translation[0], self.translation[1], markersize=5, marker="x", color="black")

		plt.yticks(np.arange(-self.max_fieldsize[1]/2, self.max_fieldsize[1]/2 + 0.1, step=self.max_fieldsize[1]/10))
		plt.xticks(np.arange(-self.max_fieldsize[0]/2, self.max_fieldsize[0]/2 + 0.1, step=self.max_fieldsize[0]/10))
		plt.axis('equal')
		plt.tight_layout()

		plt.show()
	
		return

	def create_egsinp_text(self, template_text, idx):

		template = template_text[:]

		JAW_text = [", ".join([f"{self.JAW_egsinp[0]:.4f}",f"{self.JAW_egsinp[1]:.4f}", "2"])]
		MLC_text = [[f"{self.MLC_egsinp[0,i]:.4f}",f"{self.MLC_egsinp[1,i]:.4f}", "1"] for i in range(self.num_leafes)]
		MLC_text = [", ".join(i) for i in MLC_text]

		template.insert(idx[1], JAW_text[0])

		for i in range(self.num_leafes):
			template.insert(idx[0]+i+1, MLC_text[i])

		final_text = []
		
		for line in template:
			if line.strip("\n") !=  "###HIER ERSETZEN###":
				final_text.append(line)

		return final_text

	def create_egs_file(self, path):

		field_path = f"{path}/MR-Linac_model_{self.fieldsize[0]}x{self.fieldsize[1]}_{self.translation[0]}x{self.translation[1]}.egsinp" 
		if not Path(field_path).is_file():
			with open(field_path,"x") as f:
				f.truncate(0)
				for line in self.egsinp_text:
					f.write("%s\n" % line)


"""############################################################################################################################################################
																PROGRAMM START
############################################################################################################################################################"""			

batch_size = 10000
shapes = []
template, idx = read_template()
path = "/Users/simongutwein/Documents/GitHub/Master_Thesis/Data/training_data"

field = trainingData(fieldsize=(10,10),translation=(7,-3))
#field.plot_mlc()
#field.egsinp_text = field.create_egsinp_text(template, idx)
#field.create_egs_file(path)

while len(shapes) < batch_size:

	field = trainingData(distribution="gaussian")
	if (field.fieldsize, field.translation) in shapes: 
		continue

	field.egsinp_text = field.create_egsinp_text(template, idx)
	#field.create_egs_file(path)
	shapes.append((field.fieldsize, field.translation))
	#pprint.pprint(field.__dict__)
	#field.plot_mlc()

scatter_hist_2D_data(shapes) 


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
[ ] klären wie es aussieht mit den restlichen leafes sind die wichtig oder nicht
[-] (Quasi fertig, function ist in functions.py) MLC positionen in template .egsnip file einspeichern
[ ] evtl. Output erstellen damit ich sehen kann welche Felder wie fot vorgekommen sind
[ ] JAW Positionen für .egsinp File wirken komisch. 
[✔] Leaves sind im Fled 0.2 zu weit ausgefahren
[✔] klasse für trainingsdata fertigstellen
[✔] batchweise erzeugung und speicherung
[✔] testen ob dimensions und translationskombination schon vorgekommen ist. 
[✔] möglichkeit bieten bei class auch ein festdefniniertes Feld zu erstellen
"""
from imports import *

class trainingData():
	'''
	fieldsize = desired fieldsize
	translation = desired translation 
	max_fieldsize = the biggest possible fieldsize
	max_generated_fieldsize = if fieldsize is chosen to be random, max_generated_fieldsize is the biggest fieldsize generated
	num_leafes = number of leafes
	'''

	def __init__(self, fieldsize=None, translation=None, max_fieldsize=None, max_generated_fieldsize=None, num_leafes=80):

		#check if max fieldsize is given or not, otherwise MR-Linac fieldsize is used
		if max_fieldsize is None:
			self.max_fieldsize = (57, 22)
		else:
			self.max_fieldsize =  max_fieldsize #[x_max, y_max]

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
		self.fieldsize, self.translation = self.create_field_parameters(fieldsize, translation)
		self.MLC_iso = self.create_mlc_positions()
		self.JAW_iso = self.create_jaw_positions()
		self.MLC_egsinp = self.calculate_new_mlc()
		self.JAW_egsinp = self.calculate_new_jaw()

	def create_field_parameters(self, fieldsize=None, translation=None):

		#if fielsize and translation are passed, check if they fit the max_fieldsize
		if fieldsize is not None and translation is not None:
			if fieldsize[0]/2 + abs(translation[0]) > self.max_fieldsize[0]/2 or fieldsize[1]/2 + abs(translation[1]) > self.max_fieldsize[1]/2:
				raise ValueError("fieldsize and translation dont fit inside maximum_fieldsize")

		#if fieldsize and translation is not passed use full spectrum of fieldsizes and translations
		if fieldsize is None and translation is None:
			fieldsize = ([random.randint(2,self.max_generated_fieldsize[0]), random.randint(2,self.max_generated_fieldsize[1])])

			#check for boundary conditions
			max_x_translation = self.max_fieldsize[0]/2 - fieldsize[0]/2
			max_y_translation = self.max_fieldsize[1]/2 - fieldsize[1]/2
			translation = [random.randint(-np.floor(max_x_translation), np.floor(max_x_translation)), random.randint(-np.floor(max_y_translation), np.floor(max_y_translation))]
			#translation = [random.uniform(-max_x_translation, max_x_translation), random.uniform(-max_y_translation, max_y_translation)]

		#check if one of paramters is given
		if fieldsize is None and translation is not None:
			fieldsize = ([random.randint(2,int(self.max_generated_fieldsize[0]/2)-translation[0]), random.randint(2,int(self.max_generated_fieldsize[1]/2)-translation[1])])

		if fieldsize is not None and translation is None:
			max_x_translation = self.max_fieldsize[0]/2 - fieldsize[0]/2
			max_y_translation = self.max_fieldsize[1]/2 - fieldsize[1]/2
			translation = [random.randint(-np.floor(max_x_translation), np.floor(max_x_translation)), random.randint(-np.floor(max_y_translation), np.floor(max_y_translation))]

		return fieldsize, translation

	def create_mlc_positions(self):

		MLC = np.zeros((2,self.num_leafes))
		dx, dy = self.translation[0], self.translation[1]
		#calculate the central leaves
		central_leafes = [np.floor(dx/self.leaf_width).astype(int) + int(self.num_leafes/2), np.floor(dx/self.leaf_width).astype(int) + int(self.num_leafes/2) + 1]
		print(central_leafes)
		#calculate the numbers of leafes which need to be added to the central leafes + 4 extra leafes
		count_leafes = np.ceil(self.fieldsize[0]/self.leaf_width).astype(int) + 2
		print(count_leafes)
		#make number of leafes even
		if count_leafes%2 != 0:
			count_leafes += 1

		#create small gap 
		MLC[0, :] += dy - 0.2
		MLC[1, :] += dy + 0.2

		#check for boundary conditions
		return self.check_boundary_conditions(MLC, central_leafes, count_leafes)

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
						MLC_egsinp[j,i] = (((cil + math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.MLC_iso[j,i])*0.1/ssd)*radius, 2)))*abs(self.MLC_iso[j,i])*0.1/ssd + math.cos(abs(self.MLC_iso[j,i])*0.1/ssd)*radius)*(-1))
					if self.MLC_iso[j,i] > 0:
						MLC_egsinp[j,i] = ((-(cil - math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.MLC_iso[j,i])*0.1/ssd)*radius, 2)))*abs(self.MLC_iso[j,i])*0.1/ssd + math.cos(abs(self.MLC_iso[j,i])*0.1/ssd)*radius)*(-1))    
				else:
					if self.MLC_iso[j,i] >= 0:
						MLC_egsinp[j,i] = ((cil + math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.MLC_iso[j,i])*0.1/ssd)*radius, 2)))*abs(self.MLC_iso[j,i])*0.1/ssd + math.cos(abs(self.MLC_iso[j,i])*0.1/ssd)*radius) 
					if self.MLC_iso[j,i] < 0:
						MLC_egsinp[j,i] = (-(cil - math.sqrt(pow(radius, 2)-pow(math.cos(abs(self.MLC_iso[j,i])*0.1/ssd)*radius, 2)))*abs(self.MLC_iso[j,i])*0.1/ssd + math.cos(abs(self.MLC_iso[j,i])*0.1/ssd)*radius)

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

	def check_boundary_conditions(self, MLC, central_leafes, count_leafes):

		#check for boundary conditions to move MLC accordingly
		if int(central_leafes[0] - count_leafes/2) - 1 < 0 and int(central_leafes[1] + count_leafes/2) < self.num_leafes:
			MLC[0, 0 : central_leafes[0]] -= self.fieldsize[1]/2 - 0.2
			MLC[1, 0 : central_leafes[0]] += self.fieldsize[1]/2 - 0.2
			MLC[0, central_leafes[1] - 1 : int(central_leafes[1] + count_leafes/2)] -= self.fieldsize[1]/2 - 0.2
			MLC[1, central_leafes[1] - 1 : int(central_leafes[1] + count_leafes/2)] += self.fieldsize[1]/2 - 0.2
		

		elif int(central_leafes[1] + count_leafes/2) >= self.num_leafes and int(central_leafes[0] - count_leafes/2) - 1 > 0:
			
			MLC[0, int(central_leafes[0] - count_leafes/2) - 1 : central_leafes[0]] -= self.fieldsize[1]/2 - 0.2
			MLC[1, int(central_leafes[0] - count_leafes/2) - 1 : central_leafes[0]] += self.fieldsize[1]/2 - 0.2
			MLC[0, central_leafes[1] - 1 : self.num_leafes] -= self.fieldsize[1]/2 - 0.2
			MLC[1, central_leafes[1] - 1 : self.num_leafes] += self.fieldsize[1]/2 - 0.2
		

		elif int(central_leafes[1] + count_leafes/2) >= self.num_leafes and int(central_leafes[0] - count_leafes/2) - 1 < 0:
			
			MLC[0, 0 : central_leafes[0]] -= self.fieldsize[1]/2 - 0.2
			MLC[1, 0 : central_leafes[0]] += self.fieldsize[1]/2 - 0.2
			MLC[0, central_leafes[1] - 1 : self.num_leafes] -= self.fieldsize[1]/2 - 0.2
			MLC[1, central_leafes[1] - 1 : self.num_leafes] += self.fieldsize[1]/2 - 0.2

		else:

			MLC[0, int(central_leafes[0] - count_leafes/2) - 1 : central_leafes[0]] -= self.fieldsize[1]/2 - 0.2
			MLC[1, int(central_leafes[0] - count_leafes/2) - 1 : central_leafes[0]] += self.fieldsize[1]/2 - 0.2
			MLC[0, central_leafes[1] - 1 : int(central_leafes[1] + count_leafes/2)] -= self.fieldsize[1]/2 - 0.2
			MLC[1, central_leafes[1] - 1 : int(central_leafes[1] + count_leafes/2)] += self.fieldsize[1]/2 - 0.2

		return MLC



"""############################################################################################################################################################
																PROGRAMM START
############################################################################################################################################################"""			

#print(dat.translation, dat.fieldsize, dat.MLC_iso, dat.JAW_iso)
batch_size = 1

plot = False

shapes, MLCs, JAWs = [], [], []

template, idx = read_template()

while len(MLCs) < batch_size:

	field = trainingData()

	if (field.fieldsize, field.translation) in shapes: 
		continue

	#print(template)
	#field.plot_mlc()
	field.egsinp_text = create_egsinp_text(field, template[:], idx)
 
	shapes.append((field.fieldsize, field.translation))
	MLCs.append(field.MLC_iso)
	JAWs.append(field.JAW_iso)

#print(shapes)
#print(len(shapes), np.array(MLCs).shape, np.array(JAWs).shape)

# occ = [];
# for i in shapes:
# 	if i[0] == [2,2]:
# 		occ.append(i)
# print(occ)

field = trainingData()
field.egsinp_text = create_egsinp_text(field, template, idx)
pprint.pprint(field.__dict__)
field.plot_mlc()


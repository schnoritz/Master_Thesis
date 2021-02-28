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
[✔] batchweise erzeugung und speicherung
[✔] testen ob dimensions und translationskombination schon vorgekommen ist. 
[ ] klären wie es aussieht mit den restlichen leafes sind die wichtig oder nicht
[ ] MLC positionen in template .egsnip file einspeichern
[✔] klasse für trainingsdata fertigstellen
[✔] möglichkeit bieten bei class auch ein festdefniniertes Feld zu erstellen
[ ] evtl. Output erstellen damit ich sehen kann welche Felder wie fot vorgekommen sind
[✔] Leaves sind im Fled 0.2 zu weit ausgefahren
"""
from imports import *

class trainingData():

	def __init__(self, fieldsize=None, translation=None):
		self.max_fieldsize = [57, 22]
		self.num_leafes = 80
		self.leaf_width = 0.715
		self.fieldsize, self.translation = self.create_field_parameters(fieldsize, translation)
		self.MLC_iso = self.calculate_mlc()
		self.JAW_iso = self.calculate_jaw()

	def create_field_parameters(self, fieldsize=None, translation=None):

		if fieldsize is None:
			fieldsize = ([random.randint(2,self.max_fieldsize[0]), random.randint(2,self.max_fieldsize[1])])

		#print("FIELDSIZE:", fieldsize)

		if translation is None:
			max_x_translation = self.max_fieldsize[0]/2-fieldsize[0]/2
			max_y_translation = self.max_fieldsize[1]/2-fieldsize[1]/2
			#print("MAXMIMUM OFFSET VALUES:", max_x_translation, max_y_translation)
			#accounting for boundary conditions that field translation can't be so that field lies outside of maximum field
			#offset = [random.uniform(-max_x_translation, max_x_translation), random.uniform(-max_y_translation, max_y_translation)]
			translation = [random.randint(-np.floor(max_x_translation), np.floor(max_x_translation)), random.randint(-np.floor(max_y_translation), np.floor(max_y_translation))]

		return fieldsize, translation

	def calculate_mlc(self):

		MLC = np.zeros((2,self.num_leafes))
		dx, dy = self.translation[0], self.translation[1]
		central_leafes = [np.floor(dx/self.leaf_width).astype(int) + int(self.num_leafes/2), np.floor(dx/self.leaf_width).astype(int) + int(self.num_leafes/2) + 1]
		#print("CENTRAL_LEAFES:", central_leafes)

		#anzahl der leafes die noch hinzugefügt werden sollen, in dem Fall hier 4 Leafes mehr als benötigt werden.
		count_leafes = np.ceil(self.fieldsize[0]/self.leaf_width).astype(int)+2
		
		#Leaf anzahl immer gerade machen
		if count_leafes%2 != 0:
			count_leafes += 1
		#print("COUT_LEAFES:", count_leafes)

		MLC[0,:] += dy - 0.2 #Spalt in der Mitte erzeugen, 0,2mm breite
		MLC[1,:] += dy + 0.2

		if int(central_leafes[0]-count_leafes/2)-1 < 0 and int(central_leafes[1]+count_leafes/2) < self.num_leafes:

			MLC[0,0:central_leafes[0]] -= self.fieldsize[1]/2-0.2
			MLC[1,0:central_leafes[0]] += self.fieldsize[1]/2-0.2
			MLC[0,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] -= self.fieldsize[1]/2-0.2
			MLC[1,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] += self.fieldsize[1]/2-0.2
		

		elif int(central_leafes[1]+count_leafes/2) >= self.num_leafes and int(central_leafes[0]-count_leafes/2)-1 > 0:
			
			MLC[0,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] -= self.fieldsize[1]/2-0.2
			MLC[1,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] += self.fieldsize[1]/2-0.2
			MLC[0,central_leafes[1]-1:self.num_leafes] -= self.fieldsize[1]/2-0.2
			MLC[1,central_leafes[1]-1:self.num_leafes] += self.fieldsize[1]/2-0.2
		

		elif int(central_leafes[1]+count_leafes/2) >= self.num_leafes and int(central_leafes[0]-count_leafes/2)-1 < 0:
			
			MLC[0,0:central_leafes[0]] -= self.fieldsize[1]/2-0.2
			MLC[1,0:central_leafes[0]] += self.fieldsize[1]/2-0.2
			MLC[0,central_leafes[1]-1:self.num_leafes] -= self.fieldsize[1]/2-0.2
			MLC[1,central_leafes[1]-1:self.num_leafes] += self.fieldsize[1]/2-0.2

		else:

			MLC[0,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] -= self.fieldsize[1]/2-0.2
			MLC[1,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] += self.fieldsize[1]/2-0.2
			MLC[0,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] -= self.fieldsize[1]/2-0.2
			MLC[1,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] += self.fieldsize[1]/2-0.2

		return MLC

	def calculate_jaw(self):

		return np.array([self.translation[0]-self.fieldsize[0]/2, self.translation[0]+self.fieldsize[0]/2])

	def plot_mlc(self):

		MLC = self.MLC_iso
		field = np.linspace(-self.max_fieldsize[0]/2,self.max_fieldsize[0]/2,self.num_leafes)
		plt.bar(field,MLC[1,:],color="w")
		plt.bar(field,MLC[0,:],color="w")
		plt.bar(field,15-MLC[1,:],width=0.6,bottom=MLC[1,:], color="chocolate")
		plt.bar(field,-15-MLC[0,:],width=0.6,bottom=MLC[0,:], color="sandybrown")

		ax = plt.gca()
		ax.add_patch(ptc.Rectangle((-self.max_fieldsize[0]/2,-15),self.max_fieldsize[0]/2+self.JAW_iso[0],30,facecolor="midnightblue",alpha=0.6))
		ax.add_patch(ptc.Rectangle((self.JAW_iso[1],-15),self.max_fieldsize[0]/2-self.JAW_iso[1],30,facecolor="midnightblue",alpha=0.6))

		plt.hlines([self.max_fieldsize[1]/2, -self.max_fieldsize[1]/2], xmin=-self.max_fieldsize[0]/2, xmax=self.max_fieldsize[0]/2, colors="black")
		plt.vlines([-self.max_fieldsize[0]/2, self.max_fieldsize[0]/2], ymin=-self.max_fieldsize[1]/2, ymax=self.max_fieldsize[1]/2, colors="black")

		plt.text(-self.max_fieldsize[0]/2, -17, f"Fieldsize: [{self.fieldsize[0]} X {self.fieldsize[1]}]")
		plt.text(-self.max_fieldsize[0]/2, -19, f"Offset: [{self.translation[0]} X {self.translation[1]}]")
		#plt.annotate(text='', xy=(self.max_fieldsize[0]/2+self.translation[0],self.translation[1]+self.fieldsize[1]/2), xytext=(self.max_fieldsize[0]/2+self.translation[0], self.translation[1]-self.fieldsize[1]/2), arrowprops=dict(arrowstyle='<|-|>'))
		#plt.annotate(text='', xy=(self.max_fieldsize[0]/2+self.translation[0]-self.fieldsize[0]/2 ,self.translation[1]), xytext=(self.max_fieldsize[0]/2+self.translation[0]+self.fieldsize[0]/2, self.translation[1]), arrowprops=dict(arrowstyle='<|-|>'))
		plt.plot(self.translation[0], self.translation[1], markersize=5, marker="x", color="black")

		plt.yticks(np.arange(-14, 15, step=2))
		plt.xticks(np.arange(-30, 31, step=5))
		plt.axis('equal')
		plt.tight_layout()

		plt.show()
	
		return

"""############################################################################################################################################################
																PROGRAMM START
############################################################################################################################################################"""			

#print(dat.translation, dat.fieldsize, dat.MLC_iso, dat.JAW_iso)
batch_size = 10

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
print(len(shapes), np.array(MLCs).shape, np.array(JAWs).shape)

# occ = [];
# for i in shapes:
# 	if i[0] == [10,10]:
# 		occ.append(i)
# #print(occ)

field = trainingData()
field.egsinp_text = create_egsinp_text(field, template[:], idx)
pprint.pprint(field.MLC_iso)
pprint.pprint(field.JAW_iso)
pprint.pprint(field.egsinp_text)
field.plot_mlc()

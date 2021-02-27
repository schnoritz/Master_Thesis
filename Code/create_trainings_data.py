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
[ ] batchweise erzeugung und speicherung
[✔] testen ob dimensions und translationskombination schon vorgekommen ist. 
[ ] klären wie es aussieht mit den restlichen leaves sind die wichtig oder nicht
[ ] neue MLC positionen in template .egsnip file einspeichern
[ ] klasse für trainingsdata fertigstellen

"""
from imports import *


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

def calculate_MLC_positions(fieldsize, offset):
	
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

class training_data():
	def __init__(self):
		self.fieldsize, self.translation = self.createFieldParameters()
		self.MLC_iso = self.calculateMLC()
		self.JAW_iso = self.calculateJAW()

	def calculateMLC(self):
		pass

	def calculateJAW(self):
		pass

	def createFieldParameters(self):
		fieldsize = ([random.randint(2,57), random.randint(2,22)])
		max_x_translation = 28.5-fieldsize[0]/2
		max_y_translation = 11-fieldsize[1]/2
		offset = [random.randint(-np.floor(max_x_translation), np.floor(max_x_translation)), random.randint(-np.floor(max_y_translation), np.floor(max_y_translation))]

		return fieldsize, offset


"""############################################################################################################################################################
																PROGRAMM START
############################################################################################################################################################"""			

#dat = training_data()

batch_size = 100

plot = False

shapes = []
MLCs = []
JAWs = []
while len(MLCs) < batch_size:

	fieldsize = ([random.randint(2,57), random.randint(2,22)])
	max_x_translation = 28.5-fieldsize[0]/2
	max_y_translation = 11-fieldsize[1]/2
	offset = [random.randint(-np.floor(max_x_translation), np.floor(max_x_translation)), random.randint(-np.floor(max_y_translation), np.floor(max_y_translation))]

	if (fieldsize, offset) in shapes: 
		continue

	shapes.append((fieldsize, offset))
	MLC, JAW = calculate_MLC_positions(fieldsize, offset)

	MLCs.append(MLC)
	JAWs.append(JAW)

#print(shapes)
print(len(shapes), np.array(MLCs).shape, np.array(JAWs).shape)

#size_parameters, translation_parameters = (54, 14), (0,0) #define size parameters
size_parameters, translation_parameters = None, None
size, position = create_field_parameters(size=size_parameters, translation=translation_parameters)
#print("FIELDSIZE:", size, "TRANSLATIONAL POSITION:", position)

#MLC und JAW berechnen
MLC, JAWS = calculate_MLC_positions(size, position)

#print("MLCs:", MLC, "\nJAWs:", JAWS)

if plot == True:
	plot_MLC_field(MLC*10, JAWS)

new_MLC = calcNewMLC(MLC)
new_JAW = calcNewJAW(JAWS)
#print(new_MLC, new_JAW)

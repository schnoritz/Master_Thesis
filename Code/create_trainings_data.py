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

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from test import *


def create_field_parameters(size=None, translation=None):
	"""creates parameters for the field, size and offset can be randomly generated. 
	if size and translation are given this code produces the wanted field parameters
	size and translation are tuples with x-dim and y-dim of field and translation
	"""
	if size is None:
		fieldsize = np.array([random.randint(2,57), random.randint(2,22)])

	else:
		fieldsize = np.array([size[0], size[0]])


	if translation is None:
		#accounting for boundary conditions that field translation can't be so that field lies outside of maximum field
		offset = [random.randint(-27+np.ceil(fieldsize[0]/2), 27-np.ceil(fieldsize[0]/2)), random.randint(-11+np.ceil(fieldsize[1]/2), 11-np.ceil(fieldsize[1]/2))]
	else:
		offset = [translation[0],translation[1]]

	return fieldsize, offset

def calculate_MLC_positions(fieldsize, offset):
	
	#braucht noch funktionalität an den rändern, und hat weirdes problem wenn feld in der Mitte oder so dass mittlerer Bin tiefer

	MLC = np.zeros((2,80))
	print(fieldsize, offset)
	dx, dy = offset[0], offset[1]
	central_leafes = [np.floor(dx/0.715).astype(int) + 40, np.ceil(dx/0.715).astype(int) + 40] #hier noch testen ob central leaves beide die selbe zahl haben
	print(central_leafes)

	#anzahl der leafes die noch hinzugefügt werden sollen, in dem Fall hier 4 Leafes mehr als benötigt werden.
	count_leafes = np.ceil(fieldsize[0]/0.715).astype(int)+2
	
	#Leaf anzahl immer gerade machen
	if count_leafes%2 != 0:
		count_leafes += 1
	print(count_leafes)

	MLC[0,:] += dy - 0.2 #Spalt in der Mitte erzeugen, 0,2mm breite
	MLC[1,:] += dy + 0.2
	MLC[0,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] -= fieldsize[1]/2 #MLC positionen in array speichern 
	MLC[0,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] -= fieldsize[1]/2 #MLC ist dabei ein 2x80 Array mit MLC[0] = untete Leafes
	MLC[1,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] += fieldsize[1]/2 #und MLC[1] = obere Leafes
	MLC[1,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] += fieldsize[1]/2

	#JAW positionen berechnen
	JAWS = np.array([dx-fieldsize[0]/2, dx+fieldsize[0]/2])*10.0

	return MLC, JAWS

# size_parameters, translation_parameters = (10, 10), (0,0) #define size parameters
size_parameters, translation_parameters = None, None
size, position = create_field_parameters(size=size_parameters)

#MLC und JAW berechnen
MLC, JAWS = calculate_MLC_positions(size, position)

print(MLC, JAWS)

plot_MLC_field(MLC*10, JAWS)


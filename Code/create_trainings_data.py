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

def create_field_parameters():

	fieldsize = np.array([2, 2])
	offset = [random.randint(-11,11), random.randint(-27,27)]

	return fieldsize, offset

def calculate_MLC_positions(fieldsize, offset):
	
	#braucht noch funktionalität an den rändern, und hat weirdes problem wenn feld in der Mitte oder so dass mittlerer Bin tiefer

	MLC = np.zeros((2,80))
	print(fieldsize, offset)
	dy, dx = offset[0], offset[1]
	central_leafes = [np.floor(dx/0.715).astype(int) + 40, np.ceil(dx/0.715).astype(int) + 40]
	print(central_leafes)

	count_leafes = np.ceil(fieldsize[0]/0.715).astype(int)+2
	
	if count_leafes%2 != 0:
		count_leafes += 1
	print(count_leafes)

	MLC[0,:] += dy - 0.2
	MLC[1,:] += dy + 0.2
	MLC[0,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] -= fieldsize[1]/2
	MLC[0,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] -= fieldsize[1]/2
	MLC[1,int(central_leafes[0]-count_leafes/2)-1:central_leafes[0]] += fieldsize[1]/2
	MLC[1,central_leafes[1]-1:int(central_leafes[1]+count_leafes/2)] += fieldsize[1]/2

	JAWS = np.array([dx-fieldsize[0]/2, dx+fieldsize[0]/2])*10.0

	return MLC, JAWS



size, position = create_field_parameters()
MLC, JAWS = calculate_MLC_positions(size, position)

print(MLC, JAWS)

plot_MLC_field(MLC*10, JAWS)


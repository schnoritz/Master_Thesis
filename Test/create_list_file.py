 
import dicom
import os 
import numpy as np
import matplotlib.pyplot as plt
from dicom.sequence import Sequence


CT_name = input("Please enter CT filename WITH PATH (without suffix and image+number):   ")
CT_name = CT_name.rsplit('/', 1)[-1]

number=input("How many slices? ")
number=int(number)

list_file_name = input("Select List File")

fout = open(list_file_name, 'w')

for i in range(number):
    fout.write(CT_name + "_image" + ("%05i" % i) + ".DCM" + "\n")
    
fout.close()
    

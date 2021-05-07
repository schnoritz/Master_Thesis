import os

dir = "/Users/simongutwein/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc/output"

files = [x for x in os.listdir(dir) if not x.startswith(".")]

command = []
for line in files:
    command.append(f"scp tu_zxoys08@login1.nemo.uni-freiburg.de:/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc/output/{line} /home/baumgartner/sgutwein84/training_data/3ddose\n")

with open("/home/baumgartner/sgutwein84/copy_output.sh", "w") as fout:
    fout.writelines(command)

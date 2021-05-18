import os
from datetime import date

d = date.today()
if d.month < 10:
    month = "0" + str(d.month)
else:
    month = str(d.month)

with open("/home/baumgartner/sgutwein84/copy_output.sh", "w") as fout:

    fout.write(
        f"scp -r tu_zxoys08@login1.nemo.uni-freiburg.de:/home/tu/tu_tu/tu_zxoys08/EGSnrc/egs_home/dosxyznrc/output /home/baumgartner/sgutwein84/container/output_{d.year}{month}{d.day}"
    )
    #fout.writelines(command)

import os

answer = input("Type 'CANCEL' to cancel all jobs: ")

if answer == "CANCEL":
    stream = os.popen("squeue -u sgutwein84")
    out = stream.read().split("\n")
    for line in out:
        if line:
            id = line.split()[0]
            if id != "JOBID":
                os.system(f"scancel {id}")
                print(f"{id} canceled!")

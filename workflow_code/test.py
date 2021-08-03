import matplotlib.pyplot as plt
import torch
from pt_3ddose import dose_to_pt
from pt_ct import convert_ct_array
import numpy as np
from pymedphys import gamma
import os


if __name__ == "__main__":

    dir = "/mnt/qb/baumgartner/sgutwein84/training_prostate/"
    files = [dir + x for x in os.listdir(dir) if not x.startswith(".")]
    files = [f"{dir}p{i}_0/" for i in range(41)]
    print(files)
    files.remove("/mnt/qb/baumgartner/sgutwein84/training_prostate/p6_0/")

    for file in files:
        print(file)
        masks = torch.load(f"{file}/training_data.pt")
        print(masks.shape)
        # target = torch.load(f"{files}/target_data.pt")
        # print(target.shape)

    # hostname = "134.2.168.52"
    # username = "sgutwein84"
    # password = "Derzauberkoenig1!"
    # with paramiko.SSHClient() as client:
    #     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #     client.connect(hostname=hostname, username=username, password=password)
    #     _, stdout, _ = client.exec_command(
    #         f"squeue -u sgutwein84"
    #     )

    #     jobs = []
    #     for line in stdout:
    #         if line:
    #             jobs.append(line.split()[0])

    #     print(jobs)
    #     jobs.remove("JOBID")

    #     for job in jobs:
    #         _, stdout, _ = client.exec_command(
    #             f"scancel {job}"
    #         )

    # cts = ["/home/baumgartner/sgutwein84/container/output_prostate/ct/" + x for x in os.listdir(
    #     "/home/baumgartner/sgutwein84/container/output_prostate/ct/") if not x.startswith(".")]

    # fig, ax = plt.subplots(8, 5, figsize=(30, 48))

    # for num, ct in enumerate(cts):
    #     ct_dat = convert_ct_array(ct)

    #     ax[num % 8, num // 8].imshow(ct_dat[:, 256, :])
    #     ax[num % 8, num // 8].set_title(ct.split("/")[-1])
    #     print("done ", ct.split("/")[-1])

    # plt.savefig("/home/baumgartner/sgutwein84/container/test/patients.png")

    # files = ["/home/baumgartner/sgutwein84/p0_0_1E07_new.3ddose",
    #          "/home/baumgartner/sgutwein84/p0_0_1E07_old.3ddose"]
    # ct_path = "/home/baumgartner/sgutwein84/container/output_prostate/ct/p0"

    # dose = []
    # for dose_path in files:
    #     dose.append(dose_to_pt(dose_path, ct_path))

    # gamma_options = {
    #     'dose_percent_threshold': 1,
    #     'distance_mm_threshold': 1,
    #     'lower_percent_dose_cutoff': 1,
    #     'interp_fraction': 5,  # Should be 10 or more for more accurate results
    #     'max_gamma': 1.01,
    #     'ram_available': 2**37,
    #     'quiet': False,
    #     'local_gamma': False,
    #     'random_subset': 100000
    # }

    # coords = (np.arange(0, 1.17*dose[0].shape[0], 1.17), np.arange(
    #     0, 1.17*dose[0].shape[1], 1.17), np.arange(0, 3*dose[0].shape[2], 3))

    # gamma_val = gamma(
    #     coords, np.array(dose[0]),
    #     coords, np.array(dose[1]),
    #     **gamma_options)

    # dat = ~np.isnan(gamma_val)
    # dat2 = ~np.isnan(gamma_val[gamma_val <= 1])
    # all = np.count_nonzero(dat)
    # true = np.count_nonzero(dat2)

    # print(np.round((true/all)*100, 4))

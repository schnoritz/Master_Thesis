import matplotlib.pyplot as plt
import torch


if __name__ == "__main__":

    masks = torch.load(
        "/home/baumgartner/sgutwein84/container/training_prostate/p0_0/training_data.pt")
    target = torch.load(
        "/home/baumgartner/sgutwein84/container/training_prostate/p0_0/target_data.pt")

    plt.imshow(masks[0, :, :, 37])
    plt.imshow(target[0, :, :, 37], alpha=0.4)
    plt.imshow(masks[1, :, :, 37], alpha=0.4)
    plt.savefig(
        "/home/baumgartner/sgutwein84/container/training_prostate/p0_0/test1.png")

    plt.imshow(masks[0, 256, :, :], cmap="bone")
    plt.imshow(target[0, 256, :, :], alpha=0.6)
    plt.savefig(
        "/home/baumgartner/sgutwein84/container/training_prostate/p0_0/test2.png")

    # data_path = "/home/baumgartner/sgutwein84/container/training_data_prostate/p12_34/"

    # masks = torch.load(data_path + "training_data.pt")
    # target = torch.load(data_path + "target_data.pt")

    # fig, ax = plt.subplots(1, 6, figsize=(10, 60))
    # ax[0].imshow(masks[0, :, :, 38])
    # ax[1].imshow(masks[1, :, :, 38])
    # ax[2].imshow(masks[2, :, :, 38])
    # ax[3].imshow(masks[3, :, :, 38])
    # ax[4].imshow(masks[4, :, :, 38])
    # ax[5].imshow(target[0, :, :, 38])
    # plt.savefig("/home/baumgartner/sgutwein84/container/masks.png")

    # hostname = "134.2.168.52"
    # username = "sgutwein84"
    # password = "Derzauberkoenig1!"
    # client = paramiko.SSHClient()
    # client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # client.connect(hostname=hostname, username=username, password=password)
    # z, stdout, stderr = client.exec_command(
    #     f"top"
    # )

    # for i in stdout:
    #     print(i)

    # for i in stderr:
    #     print(i)

    # jobs.remove("354827")
    # for ID in jobs:
    #     print(ID)
    #     _, stdout, _ = client.exec_command(
    #         f"scancel {ID}"
    #     )

    # client.close()
    # workbook = xlsxwriter.Workbook(
    #     '/Users/simongutwein/Studium/Masterarbeit/segments.xlsx')
    # worksheet = workbook.add_worksheet()

    # # print(len([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 130.0, 150.0, 170.0,
    # # 190.0, 220.0, 230.0, 260.0, 280.0, 300.0, 320.0, 340.0, 350.0]))

    # dir = "/home/baumgartner/sgutwein84/container/output_prostate/"

    # segs = [x for x in os.listdir(
    #     dir) if not "ct" in x and not "egsphant" in x and not x.startswith(".")]

    # patients = np.array([x.split("_")[0] for x in segs])

    # print(len(np.unique(patients)))

    # for patient in np.unique(patients):
    #     count = [x for x in segs if patient + "_" in x]
    #     print(f"{patient}: {len(count)}")

    # egsinp_files = [f"{dir}{x}/{x}.egsinp" for x in segs]
    # beam_files = [f"{dir}{x}/beam_config_{x}.txt" for x in segs]

    # segs_infos = []
    # num = 0
    # for egsinp, beam, seg in zip(egsinp_files, beam_files, segs):

    #     with open(egsinp) as fin:
    #         lines = fin.readlines()
    #         angle = float(lines[5].split(",")[6]) - 270
    #         iso_shift = lines[5].split(",")[2:5]

    #     with open(beam) as fin:
    #         lines = fin.readlines()
    #         lines = [x.split(",")[:2] for x in lines]
    #         area = np.round(
    #             np.array([float(x[1]) - float(x[0]) for x in lines]).sum(), 1)

    #     worksheet.write(num, 0, seg)     # Writes an int
    #     worksheet.write(num, 1, angle)  # Writes a float
    #     worksheet.write(num, 2, ",".join(iso_shift))  # Writes a string
    #     worksheet.write(num, 3, area)     # Writes None
    #     num += 1

    #     segs_infos.append(
    #         {
    #             "segment": seg,
    #             "angle": angle,
    #             "iso_shift": iso_shift,
    #             "area": np.round(area, 1)
    #         }
    #     )

    #     # print(f"Segment    : {seg}")
    #     # print(f"Angle      : {angle}")
    #     # print(f"ISO-Shift  : {iso_shift}")
    #     # print(f"Area       : {np.round(area,1)}\n\n")

    # workbook.close()

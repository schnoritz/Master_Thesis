from pydicom import dcmread, uid
import numpy as np
import os

if __name__ == "__main__":

    # hallo ivan
    # hier ist das Skript für die DVH analyse
    # du benmötigst hier folgendes:
    #   1. deine Dosisverteilung 1 (zb. von normalem CT) -> hier zb dose_dat_1
    #   2. deine Dosisverteilung 2 (zb. von pseudo CT) -> hier exemplarisch auch dose_dat_1 nur mit noise zum testen
    #       WICHTIG!: müssen beide die selbe shape haben: zb (512, 512, 110)
    #   3. dein Anfangspunkt deines Files also image position patient -> aus dose file gelesen (origin)
    #   4. dein structure file
    #   5. Pfad in dem du dein Bild und deine Excel gespeichert haben möchtest

    pwd = os.getcwd() + "/"
    print(pwd)

    dose_file_dat = dcmread(pwd + "pt0_dose.dcm", force=True)
    dose_file_dat.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
    dose_dat_1 = dose_file_dat.pixel_array/1000
    dose_dat_2 = np.copy(dose_dat_1) + np.random.randn(*dose_dat_1.shape)
    dose_dat_1 = np.transpose(dose_dat_1, [1, 2, 0])
    dose_dat_2 = np.transpose(dose_dat_2, [1, 2, 0])
    print(dose_dat_1.max(), dose_dat_2.max())

    origin = np.array(dose_file_dat.ImagePositionPatient)
    px_sp = dose_file_dat.PixelSpacing
    # px_sp.append(dose_file_dat.SliceThickness)
    px_sp.append(3)
    px_sp = np.array(px_sp)
    print(px_sp)

    structure_file = pwd + "pt0_strctr.dcm"

    structures = analyse_structures(structure_file, origin, px_sp, dose_dat_1.shape, dose_dat_1, dose_dat_2)

    #import matplotlib.pyplot as plt
    # ctv = structures[13]
    # for i in range(ctv["binary_mask"].shape[2]):
    #     plt.imshow(ctv["binary_mask"][:,:,i], interpolation=None)
    #     plt.imshow(dose_dat_1[:, :, i], interpolation=None, alpha=0.5)
    #     plt.show()

    plot_dvh(structures, pwd + "dvh.png", diff=False)
    dvh_values_to_xlsx(structures, pwd + "dvh_data.xlsx")

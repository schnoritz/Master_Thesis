import os


if __name__ == "__main__":

    output_path = "/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/output/"
    target_name = input("Enter the desired folder name: ")

    while os.path.isdir(f"/work/ws/nemo/tu_zxoys08-EGS-0/{target_name}"):
        print("Folder already exists! Choose different name!")
        target_name = input("Enter the desired folder name: ")

    segments = [x for x in os.listdir(output_path) if not x.startswith(".")]
    patients = [x.split("_")[0] for x in segments]
    patients = list(dict.fromkeys(patients))

    os.mkdir(f"/work/ws/nemo/tu_zxoys08-EGS-0/{target_name}")
    os.mkdir(f"/work/ws/nemo/tu_zxoys08-EGS-0/{target_name}/ct")

    for segment in segments:
        os.rename(output_path + segment, f"/work/ws/nemo/tu_zxoys08-EGS-0/{target_name}/{segment}")
        pass

    for patient in patients:
        os.rename(f"/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/{patient}", f"/work/ws/nemo/tu_zxoys08-EGS-0/{target_name}/ct/{patient}")

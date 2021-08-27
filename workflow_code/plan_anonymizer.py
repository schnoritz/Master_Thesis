
from pydicom import dcmread, uid
import os
from pprint import pprint
from glob import glob


def create_anonymized_data(path):

    # get patient name and fix forwardslash for paths
    if path[-1] != "/":
        patient_name = path.split("/")[-1]
        path += "/"
    else:
        patient_name = path.split("/")[-2]

    # get paths for all files in folder
    all_files = [path + x for x in os.listdir(
        path) if not x.startswith(".") and x.lower().endswith(".dcm") and not "Sets" in x]

    # define fileds to anonymize
    fields_to_anonymize = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "OtherPatientIDs",
        "OtherPatientNames",
        "PatientsBirthName",
        "PatientsAddress",
        "PatientsMothersBirthName",
        "CountryOfResidence",
        "RegionOfResidence",
        "PatientsTelephoneNumbers",
        "StudyID",
        "CurrentPatientLocation",
        "PatientsInstitutionResidence",
        "DateTime",
        "Date",
        "Time",
        "PersonName",
        "StudyDate",
        "StudyTime "
    ]
    # StudyID evtl. hinzufügen

    # create folder for anonymized trainingsdata
    anonymized_folder = path + patient_name + "/"
    if not os.path.isdir(anonymized_folder):
        os.system(f"mkdir {anonymized_folder}")

    # anonyimize fields and save new dicom file
    for file in all_files:
        filename = file.split("/")[-1]
        dat = dcmread(file, force=True)
        for field in fields_to_anonymize:
            if field in dat:
                setattr(dat, field, "ANONYM")

        dat.save_as(f"{anonymized_folder}{filename}")

    # get paths for all ct images
    ct_files = [anonymized_folder + x for x in os.listdir(
        anonymized_folder) if not x.startswith(".") and x.lower().endswith(".dcm") and "image" in x]

    # create list of dict to sort the ct images depending on slice location
    ct_dict = []
    for file in ct_files:
        ct_dict.append({
            "filename": file,
            "location": dcmread(file, force=True).SliceLocation
        })

    # sort dict after slice location
    ct_dict = sorted(ct_dict, key=lambda d: d['location'])

    pix = []
    i = 0
    # change name of files depending on slice location, so that smallest slice location has index 0
    for file in ct_dict:
        dat = dcmread(file['filename'], force=True)
        dat.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
        pix.append(dat.pixel_array)
        os.rename(file['filename'], anonymized_folder +
                  f"CT_image{str(i)}.dcm")
        i += 1

    # rename dose and plan file to fit naming convention
    dose_file = [anonymized_folder + x for x in os.listdir(
        anonymized_folder) if not x.startswith(".") and "Dose" in x]
    plan_file = [anonymized_folder + x for x in os.listdir(
        anonymized_folder) if not x.startswith(".") and not "Dose" in x and not "image" in x]

    os.rename(dose_file[0], "/".join(dose_file[0].split("/")
                                     [:-1]) + f"/{patient_name}_dose.dcm")
    os.rename(plan_file[0], "/".join(plan_file[0].split("/")
                                     [:-1]) + f"/{patient_name}_plan.dcm")


if __name__ == "__main__":

    # enter path to directory with patient ct, dose and plan files:
    path = "/Users/simongutwein/Studium/Masterarbeit/DATA/"
    files = [
        path + x for x in os.listdir(path) if not x.startswith(".") and not "xlsx" in x]
    for file in files:
        create_anonymized_data(file)
        print(file, " done!")

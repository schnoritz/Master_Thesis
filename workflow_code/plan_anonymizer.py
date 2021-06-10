
from pydicom import dcmread, uid
import matplotlib.pyplot as plt
import os

patient_folder = "/Users/simongutwein/Studium/Masterarbeit/DATA/Rehak,Alois/"

all_files = [patient_folder + x for x in os.listdir(
    patient_folder) if not x.startswith(".") and x.lower().endswith(".dcm")]

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

anonymized_folder = patient_folder + "anonymized"
if not os.path.isdir(anonymized_folder):
    os.system(f"mkdir {anonymized_folder}")

for file in all_files:
    filename = file.split("/")[-1]
    dat = dcmread(file, force=True)
    for field in fields_to_anonymize:
        if field in dat:
            setattr(dat, field, "ANONYM")

    dat.save_as(f"{anonymized_folder}/{filename}")


# wichtig wen TransferSyntaxUID nicht gesettet
file = "/Users/simongutwein/Studium/Masterarbeit/DATA/Rehak,Alois/anonymized/0004221866_CT1_image00126.DCM"
dat = dcmread(file, force=True)
dat.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian

pix = dat.pixel_array
plt.imshow(pix)
plt.show()


# for i in range(pix.shape[0]):
#     plt.imshow(pix[i, :, :])
#     plt.show()
#     plt.close()

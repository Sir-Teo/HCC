import os
import csv
import pydicom
from tqdm import tqdm
import random

# -------------------------- Configuration -------------------------- #

# Define the root directory containing DICOM files
dicom_root = '/gpfs/data/mankowskilab/HCC_Recurrence/dicom'  # Update if necessary

# Output CSV file path
output_csv_path = 'patient_labels.csv'

# Define the axial orientation tuple as strings
AXIAL_ORIENTATION = ('1', '0', '0', '0', '1', '0')

# Define the DICOM file extension
DICOM_EXTENSION = '.dcm'

# -------------------------- Functions -------------------------- #

def extract_patient_ids(dicom_root):
    """
    Extract all Patient IDs from the DICOM root directory.
    Assumes each subdirectory in dicom_root is a Patient_id.
    """
    try:
        patient_ids = [
            d for d in os.listdir(dicom_root)
            if os.path.isdir(os.path.join(dicom_root, d))
        ]
        return patient_ids
    except Exception as e:
        print(f"Error accessing DICOM root directory '{dicom_root}': {e}")
        return []

def has_axial_image(patient_dir):
    """
    Check if the patient has at least one axial image based on ImageOrientationPatient.
    """
    try:
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if file.lower().endswith(DICOM_EXTENSION):
                    dicom_path = os.path.join(root, file)
                    try:
                        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                        orientation = ds.get('ImageOrientationPatient', None)
                        if orientation:
                            # Convert orientation values to strings for comparison
                            orientation_tuple = tuple(str(x) for x in orientation)
                            if orientation_tuple == AXIAL_ORIENTATION:
                                return True
                        else:
                            print(f"Warning: 'ImageOrientationPatient' not found in {dicom_path}.")
                    except Exception as e:
                        print(f"Error reading DICOM file '{dicom_path}': {e}")
        return False
    except Exception as e:
        print(f"Error traversing patient directory '{patient_dir}': {e}")
        return False

def assign_random_label():
    """
    Assign a random label: 0 or 1.
    """
    return random.choice([0, 1])

def generate_patient_labels_csv(dicom_root, output_csv):
    """
    Generate a CSV file with Patient_id and randomly assigned Label for all patients.
    """
    patient_ids = extract_patient_ids(dicom_root)
    total_patients = len(patient_ids)
    print(f"Total patients found: {total_patients}")

    if total_patients == 0:
        print("No patients found. Exiting.")
        return

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Patient_id', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for patient_id in tqdm(patient_ids, desc="Processing Patients"):
            label = assign_random_label()
            writer.writerow({'Patient_id': patient_id, 'Label': label})

    print(f"CSV file '{output_csv}' has been generated successfully.")


# -------------------------- Main Execution -------------------------- #

def main():
    generate_patient_labels_csv(dicom_root, output_csv_path)

if __name__ == "__main__":
    main()

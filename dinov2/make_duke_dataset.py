import os
import shutil
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split

# Function to convert DICOM to JPEG
def convert_dicom_to_jpeg(dicom_path, jpeg_path):
    try:
        # Read the DICOM file
        dicom = pydicom.dcmread(dicom_path)
        
        # Extract pixel array from the DICOM file
        image_data = dicom.pixel_array
        
        # Normalize image data to 0-255 if needed
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
        image_data = image_data.astype(np.uint8)
        
        # Convert to PIL Image and save as JPEG
        image = Image.fromarray(image_data)
        image.save(jpeg_path, 'JPEG')
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")

# Load the CSV file
df = pd.read_csv('/gpfs/data/mankowskilab/HCC/data/Series_Classification/SeriesClassificationKey.csv')

# Define the root directory for the organized dataset
root_dir = '/gpfs/data/mankowskilab/HCC/data/Series_Classification'
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')
test_dir = os.path.join(root_dir, 'test')
extra_dir = os.path.join(root_dir, 'extra')

# Create necessary directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(extra_dir, exist_ok=True)

# Split the data into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['Label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Label'], random_state=42)

# Function to organize dataset by copying and converting image files into appropriate directories
def organize_dataset(subset_df, subset_dir, subset_type):
    image_count = {label: 0 for label in subset_df['Label'].unique()}

    for label in subset_df['Label'].unique():
        label_dir = os.path.join(subset_dir, label)
        os.makedirs(label_dir, exist_ok=True)

    total_images = 0
    for _, row in subset_df.iterrows():
        dlds_id = str(row['DLDS']).zfill(4)
        series_num = str(row['Series'])
        label = row['Label']
        label_dir = os.path.join(subset_dir, label)

        for file_index in range(1, 51):  # Adjust as needed for the number of files
            source_file = f'/gpfs/data/mankowskilab/HCC/data/Series_Classification/{dlds_id}/{series_num}/{str(file_index).zfill(4)}.dicom'
            image_count[label] += 1
            destination_file = os.path.join(label_dir, f'{label}_{image_count[label]}.JPEG')
            
            if os.path.exists(source_file):
                convert_dicom_to_jpeg(source_file, destination_file)
                total_images += 1
            else:
                pass
                #print(f"Warning: {source_file} does not exist.")
    
    print(f"Total images in {subset_type}: {total_images}")

# Organize training, validation, and test datasets
organize_dataset(train_df, train_dir, 'TRAIN')
organize_dataset(val_df, val_dir, 'VAL')
organize_dataset(test_df, test_dir, 'TEST')

# Create a labels.txt file listing all classes with indices and names in the ImageNet format
labels_path = os.path.join(root_dir, 'labels.txt')
unique_labels = sorted(df['Label'].unique())

with open(labels_path, 'w') as f:
    for idx, label in enumerate(unique_labels):
        # Write each label in the format: {index: 'label_name'}
        label_entry = f"{label}, {label}\n"
        f.write(label_entry)

print("labels.txt generated in ImageNet label format.")

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import monai.transforms as mtf
import random
import csv

# Define constants
DESIRED_NUM_IMAGES = 32  # Number of images to sample per patient
RANDOM_SEED = 42  # For reproducibility
random.seed(RANDOM_SEED)

# Define the input and output directories
input_image_dir = '/media/user1/HD8TB/External_Validation/Ultrasound_Ext_Fan_SAM'
input_report_file = '/media/user1/HD8TB/External_Validation/Ext_Val.csv'
output_dir = '/media/user1/HD8TB/External_Validation/Ext_Preprocessed'
skipped_patients_file = os.path.join(output_dir, 'skipped_patients.csv')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the reports CSV file
reports_df = pd.read_csv(input_report_file)
# Create a dictionary for faster lookup
reports_dict = dict(zip(reports_df['XNATSessionID'], reports_df['FINDINGS']))

# Define the image transformation pipeline
transform = mtf.Compose([
    mtf.CropForeground(),  # Removes unnecessary background
    mtf.Resize(spatial_size=[256, 256], mode="bilinear")  # Resizes images to a fixed size
])

def sample_images(image_files, num_images=DESIRED_NUM_IMAGES):
    """
    Sample a fixed number of images from the sequence.
    """
    # Skip first and last 5 images if sequence is long enough
    sequence = image_files[5:-5] if len(image_files) > 50 else image_files
    
    if len(sequence) < 2:
        return None  # Not enough images after adjustments
    
    # Sample images to meet the desired number
    if len(sequence) > num_images:
        # Uniform sampling
        step = len(sequence) / num_images
        selected_images = [sequence[int(i * step)] for i in range(num_images)]
    else:
        # Cycle sequentially until we reach desired number
        selected_images = sequence.copy()
        while len(selected_images) < num_images:
            selected_images.append(sequence[len(selected_images) % len(sequence)])
    
    return selected_images

def process_patient_folder(folder_name):
    try:
        input_folder = os.path.join(input_image_dir, folder_name)
        output_folder = os.path.join(output_dir, folder_name)
        
        # Skip if no report available
        if folder_name not in reports_dict:
            return {'patient': folder_name, 'reason': 'no_report'}
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Save the report
        report_path = os.path.join(output_folder, 'text.txt')
        with open(report_path, 'w') as f:
            f.write(reports_dict[folder_name])
        
        # Get all image files
        image_files = [f for f in os.listdir(input_folder) 
                      if f.endswith('.jpg') and f.startswith('cropped_')]
        
        if len(image_files) == 0:
            return {'patient': folder_name, 'reason': 'no_images'}
        
        def extract_sequence_number(filename):
            try:
                return int(filename.split('-')[-1].split('.')[0])
            except (IndexError, ValueError):
                print(f"Invalid filename format: {filename}")
                return 0  # 保证排序继续但可能打乱顺序
            
        # Sort images based on their sequence number
        image_files.sort(key=extract_sequence_number)
        
        # Sample images
        selected_images = sample_images(image_files)
        if selected_images is None:
            return {'patient': folder_name, 'reason': 'insufficient_images'}
        
        images_stack = []
        for image_file in selected_images:
            image_path = os.path.join(input_folder, image_file)
            try:
                # Load and process image
                img = Image.open(image_path)
                img = img.convert("L")  # Convert to grayscale
                img_array = np.array(img)
                
                # Normalize pixel values to [0, 1]
                img_array = img_array.astype(np.float32) / 255.0
                
                # Add channel dimension
                img_array = img_array[None, ...]
                
                # Apply transformations
                img_trans = transform(img_array)
                
                images_stack.append(img_trans)
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
        
        if images_stack:
            # Stack all images into a single array and save
            full_sequence = np.concatenate(images_stack, axis=0)
            sequence_path = os.path.join(output_folder, 'img.npy')
            np.save(sequence_path, full_sequence)
            
            return None  # Success
            
    except Exception as e:
        return {'patient': folder_name, 'reason': f'processing_error: {str(e)}'}

# Get all patient folders
patient_folders = [folder for folder in os.listdir(input_image_dir) 
                  if os.path.isdir(os.path.join(input_image_dir, folder))]

# Process all folders and collect skipped patients
skipped_patients = []
with Pool(processes=6) as pool:
    with tqdm(total=len(patient_folders), desc="Processing") as pbar:
        for result in pool.imap_unordered(process_patient_folder, patient_folders):
            if result is not None:  # If patient was skipped
                skipped_patients.append(result)
            pbar.update(1)

# Write skipped patients to CSV
with open(skipped_patients_file, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['patient', 'reason'])
    writer.writeheader()
    writer.writerows(skipped_patients)

print(f"\nProcessing complete. Skipped patients logged to {skipped_patients_file}")
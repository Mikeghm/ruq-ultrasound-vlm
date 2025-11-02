import os
import zipfile
import pydicom
from PIL import Image
import numpy as np

def dicom_to_jpg(dicom_file, output_jpg):
    # 读取DICOM文件
    dicom = pydicom.dcmread(dicom_file)
    
    # 获取像素数组（图像数据）
    img = dicom.pixel_array
    
    # 归一化图像
    if img.dtype != np.uint8:
        img = img.astype(float)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
    
    # 创建PIL图像
    img_pil = Image.fromarray(img)
    
    # 添加黑色水印到图像顶部10%
    width, height = img_pil.size
    watermark_height = int(height * 0.1)
    watermark = Image.new('RGB', (width, watermark_height), color='black')
    img_pil.paste(watermark, (0, 0))
    
    # 保存为JPG
    img_pil.save(output_jpg)

def process_zip_file(zip_path, output_base_dir):
    # Extract the zip file name without extension
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    
    # Create output directory for this zip file
    output_dir = os.path.join(output_base_dir, zip_name)
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all contents to a temporary directory
        temp_dir = os.path.join(output_dir, 'temp')
        zip_ref.extractall(temp_dir)
        
        # Process all DICOM files in the temporary directory
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_path = os.path.join(root, file)
                    jpg_filename = os.path.splitext(file)[0] + ".jpg"
                    jpg_path = os.path.join(output_dir, jpg_filename)
                    
                    try:
                        dicom_to_jpg(dicom_path, jpg_path)
                        print(f"Converted {file} to {jpg_filename}")
                    except Exception as e:
                        print(f"Error converting {file}: {str(e)}")
        
        # Remove the temporary directory
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)

def batch_process_zip_files(input_dir, output_base_dir):
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process all zip files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.zip'):
            zip_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            process_zip_file(zip_path, output_base_dir)
            print(f"Finished processing {filename}")

# Example usage
input_directory = "/media/user1/myHD20TB/Ultrasound_XNAT"
output_directory = "/media/user1/myHD20TB/preprocessed/original_images/Ultrasound_XNAT"
batch_process_zip_files(input_directory, output_directory)
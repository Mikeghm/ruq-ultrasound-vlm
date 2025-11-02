Image preprocessing process:
1. First run batch-dicom2jpg.py to convert dicom to a jpg image
2. Run Batch_Crop_San_SArea_SAM.py again to crop the fan-shaped area of the ultrasound image
3. Then run Black_image_Demove.cy to remove all black ultrasound images
4. Run M3D_Input_Sataset_Creation.py to create the dataset format required for the M3D model
5. Run add-dimension.ipynb to increase dimensions so that the model can accept input
6. Finally, run Dataset_CSV_Creation.py to create the dataset description file (in JSON format) required for the M3D model
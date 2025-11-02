import cv2
import os
from pathlib import Path
import numpy as np

def is_black_image(image_path, threshold=20):
    """
    判断图像是否是全黑的。
    
    参数：
    - image_path: Path，图像文件路径对象
    - threshold: int，像素值均值的阈值，低于该值认为是黑图
    
    返回：
    - bool: 如果是黑图返回True，否则返回False
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False  # 读取失败，跳过
    mean_pixel_value = np.mean(img)
    return mean_pixel_value < threshold

def remove_black_images(root_folder):
    """
    递归遍历文件夹，删除所有全黑图片。
    
    参数：
    - root_folder: str，包含患者文件夹的根目录路径
    """
    root_path = Path(root_folder)
    # 获取所有子文件夹中的图片文件（支持常见格式）
    image_files = [f for f in root_path.rglob('*') 
                  if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    
    deleted_count = 0
    for img_path in image_files:
        if is_black_image(img_path):
            try:
                print(f"删除全黑图片: {img_path.relative_to(root_path)}")
                os.remove(img_path)
                deleted_count += 1
            except Exception as e:
                print(f"删除失败 {img_path}: {str(e)}")
            
    print(f"清理完成，共删除 {deleted_count} 张全黑图片")

if __name__ == "__main__":
    input_folder = "/media/user1/HD8TB/External_Validation/Ultrasound_Ext_Fan_SAM"  # 根目录
    remove_black_images(input_folder)
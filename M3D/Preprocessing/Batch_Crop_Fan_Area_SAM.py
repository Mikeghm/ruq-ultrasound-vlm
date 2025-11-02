import cv2
import numpy as np
import os
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def crop_ultrasound_fan(img):
    """
    Crops the fan-shaped area from an ultrasound image with tighter borders.
    Args:
        img (numpy.ndarray): Input ultrasound image
    Returns:
        numpy.ndarray: Cropped image
    """
    if img is None:
        raise ValueError("Invalid image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a black mask of the same size as the image
    mask = np.zeros_like(gray)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Define the fan shape parameters with tighter margins
    bottom_center = (width // 2, height - 30)
    fan_width = width * 0.8
    fan_height = height * 0.8
    
    # Create points for the fan shape with tighter margins
    fan_points = np.array([
        [bottom_center[0], bottom_center[1]],
        [int(bottom_center[0] - fan_width/2), bottom_center[1]],
        [int(width/2 - fan_width/3), int(height - fan_height)],
        [int(width/2 + fan_width/3), int(height - fan_height)],
        [int(bottom_center[0] + fan_width/2), bottom_center[1]],
    ], np.int32)
    
    # Fill the fan shape in the mask
    cv2.fillPoly(mask, [fan_points], 255)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Find the bounding rectangle of the fan with tighter margins
    x_min = max(0, int(bottom_center[0] - fan_width/2))
    x_max = min(width, int(bottom_center[0] + fan_width/2))
    y_min = max(0, int(height - fan_height))
    y_max = min(height, bottom_center[1])
    
    # Add a small margin
    margin = 10
    x_min = max(0, x_min - margin)
    x_max = min(width, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(height, y_max + margin)
    
    # Crop the image
    cropped = result[y_min:y_max, x_min:x_max]
    
    # Remove empty borders
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray_cropped)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = cropped[y:y+h, x:x+w]
    
    return cropped

class SAMSegmenter:
    """SAM模型分割处理器"""
    def __init__(self, device="cuda"):
        # 配置参数
        self.model_type = "vit_h"  # 使用大模型
        self.checkpoint_path = "/media/user1/HD8TB/Preprocessing_Pipeline/Images/sam_vit_h_4b8939.pth"
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # 初始化模型
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=24,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            min_mask_region_area=200,
        )

    def generate_mask(self, image):
        """生成扇形区域掩模"""
        # 转换颜色空间
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 生成掩模
        masks = self.mask_generator.generate(rgb_image)
        
        # 筛选最佳掩模
        valid_masks = [m for m in masks if self._is_valid_sector(m, image.shape)]
        if not valid_masks:
            return None
        best_mask = max(valid_masks, key=lambda x: x['stability_score'])
        return best_mask['segmentation'].astype(np.uint8)

    def _is_valid_sector(self, mask, img_shape):
        """验证掩模是否符合扇形区域特征"""
        h, w = img_shape[:2]
        bbox = mask['bbox']
        
        # 面积过滤
        area_ratio = mask['area'] / (h * w)
        if not 0.1 < area_ratio < 0.7:
            return False
        
        # 位置过滤（中心在上半部）
        cy = bbox[1] + bbox[3]/2
        if cy > h * 0.6:
            return False
        
        # 形状紧凑度
        contours, _ = cv2.findContours(mask['segmentation'].astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return False
        roundness = 4 * np.pi * cv2.contourArea(cnt) / (perimeter ** 2)
        return roundness > 0.4

class UltrasoundProcessor:
    """超声图像处理管道"""
    def __init__(self):
        self.sam = SAMSegmenter()
        self.fallback_enabled = True  # 启用传统方法回退

    def process_image(self, img_path, output_path):
        """处理单个图像"""
        try:
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图像: {img_path}")
                return False

            # 使用SAM生成掩模
            mask = self.sam.generate_mask(img)
            
            # 回退机制
            if mask is None and self.fallback_enabled:
                print(f"SAM检测失败，使用传统方法: {img_path.name}")
                return self._process_with_fallback(img, output_path)

            # 应用掩模
            cropped = self._apply_mask(img, mask)
            
            # 后处理
            final_img = self._postprocess(cropped)
            
            # 保存结果
            cv2.imwrite(str(output_path), final_img)
            return True
            
        except Exception as e:
            print(f"处理失败 {img_path.name}: {str(e)}")
            return False

    def _apply_mask(self, img, mask):
        """应用掩模并裁剪"""
        # 生成掩模图像
        masked = cv2.bitwise_and(img, img, mask=mask)
        
        # 计算有效区域
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        
        # 扩展边界
        margin = 0
        x = max(0, int(x - w * margin))
        y = max(0, int(y - h * margin))
        w = min(img.shape[1]-x, int(w * (1 + 2*margin)))
        h = min(img.shape[0]-y, int(h * (1 + 2*margin)))
        
        return masked[y:y+h, x:x+w]

    def _postprocess(self, img):
        """后处理流程"""
        # 添加水印
        img = self._add_watermark(img)
        # 移除空白边界
        return self._remove_empty_borders(img)

    def _add_watermark(self, img):
        """添加黑色水印（保留原实现）"""
        height = img.shape[0]
        watermark_height = int(height * 0.05)
        img[-watermark_height:, :] = 0
        return img

    def _remove_empty_borders(self, img):
        """移除空白边界（保留原实现）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w]
        return img

    def _process_with_fallback(self, img, output_path):
        """传统方法回退（保留原实现）"""
        try:
            cropped = crop_ultrasound_fan(img)
            cv2.imwrite(str(output_path), cropped)
            return True
        except Exception as e:
            print(f"传统方法失败: {str(e)}")
            return False

# 保留原有的批处理函数（仅修改process_patient_folder）
def process_patient_folder(patient_folder, base_output_folder, processor):
    """
    处理患者文件夹（更新版）
    """
    # 创建输出目录
    output_folder = base_output_folder / patient_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    # 获取图像文件
    image_files = [f for f in patient_folder.iterdir() 
                  if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    
    # 并行处理配置（可选）
    for img_path in image_files:
        output_path = output_folder / f"cropped_{img_path.name}"
        if not output_path.exists():
            success = processor.process_image(img_path, output_path)
            status = "成功" if success else "失败"
            print(f"处理 {img_path.name} => {status}")
        else:
            print(f"跳过已存在文件: {img_path.name}")

def main():
    # 初始化处理器
    processor = UltrasoundProcessor()
    
    # 输入输出配置
    base_input = Path("/media/user1/HD8TB/External_Validation/Ultrasound_Ext")
    base_output = Path("/media/user1/HD8TB/External_Validation/Ultrasound_Ext_Fan_SAM")
    
    # 创建输出目录
    base_output.mkdir(parents=True, exist_ok=True)
    
    # 获取患者文件夹
    patient_folders = [f for f in base_input.iterdir() 
                      if f.is_dir() and '-' in f.name]
    
    # 处理每个患者
    for folder in patient_folders:
        print(f"\n正在处理患者: {folder.name}")
        process_patient_folder(folder, base_output, processor)

if __name__ == "__main__":    
    main()
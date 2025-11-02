import os
import json
from pathlib import Path

def create_ultrasound_dataset_json(data_root, mode="full"):
    """Create dataset JSON file with mode selection (full/val-only)"""
    data_dir = Path(data_root)
    
    all_patients = [d for d in data_dir.iterdir() if d.is_dir()]
    
    def create_entries(patients):
        entries = []
        for patient_dir in patients:
            img_path = patient_dir / "img.npy"
            txt_path = patient_dir / "text.txt"
            if img_path.exists() and txt_path.exists():
                entries.append({
                    "image": str(img_path.relative_to(data_dir)),
                    "text": str(txt_path.relative_to(data_dir))
                })
        return entries
    
    if mode == "full":
        # 90/10 划分验证测试集
        train_split = int(len(all_patients) * 0.9)
        dataset = {
            "train": create_entries(all_patients[:train_split]),
            "validation": create_entries(all_patients[train_split:]),
            "test": create_entries(all_patients[train_split:])  # 验证测试共用
        }
    elif mode == "val-only":
        # 全部作为验证集
        dataset = {
            "train": [],
            "validation": create_entries(all_patients),
            "test": create_entries(all_patients)
        }
    else:
        raise ValueError("Invalid mode. Choose 'full' or 'val-only'")

    output_path = data_dir / "ultrasound_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    mode_info = " (validation only)" if mode == "val-only" else ""
    print(f"Created dataset{mode_info} at: {output_path}")
    print(f"Total samples: {len(dataset['validation'])} validation samples")
    return str(output_path)

if __name__ == "__main__":
    # 生成完整数据集
    # create_ultrasound_dataset_json("/path/to/dataset")
    
    # 仅生成验证集
    create_ultrasound_dataset_json("/media/user1/HD8TB/External_Validation/Ext_Preprocessed", mode="val-only")

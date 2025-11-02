import os
import zipfile
from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm
import pandas as pd
import monai.transforms as mtf
import json
import argparse


# -----------------------------
# Step 1: DICOM -> JPG (+watermark)
# -----------------------------
def dicom_to_jpg(dicom_file: Path, output_jpg: Path):
    dicom = pydicom.dcmread(str(dicom_file))
    img = dicom.pixel_array

    # Normalize to uint8
    if img.dtype != np.uint8:
        img = img.astype(float)
        imin, imax = float(img.min()), float(img.max())
        if imax > imin:
            img = (img - imin) / (imax - imin)
        else:
            img = img * 0.0
        img = (img * 255).astype(np.uint8)

    img_pil = Image.fromarray(img)

    # Add black watermark at top 10% (de-identification)
    width, height = img_pil.size
    watermark_height = int(height * 0.1)
    if watermark_height > 0:
        watermark = Image.new("RGB", (width, watermark_height), color="black")
        img_pil = img_pil.convert("RGB")
        img_pil.paste(watermark, (0, 0))

    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(str(output_jpg))


def process_zip_to_jpg(zip_path: Path, output_base_dir: Path):
    zip_name = zip_path.stem
    output_dir = output_base_dir / zip_name
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(temp_dir))

    # Convert all DICOM files in temp_dir
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_path = Path(root) / file
                jpg_filename = f"{Path(file).stem}.jpg"
                jpg_path = output_dir / jpg_filename
                try:
                    dicom_to_jpg(dicom_path, jpg_path)
                except Exception as e:
                    print(f"Error converting {dicom_path}: {e}")

    # Clean temp_dir
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for name in files:
            try:
                Path(root, name).unlink()
            except Exception:
                pass
        for name in dirs:
            try:
                Path(root, name).rmdir()
            except Exception:
                pass
    try:
        temp_dir.rmdir()
    except Exception:
        pass


def batch_zip_to_jpg(input_dir: Path, output_base_dir: Path):
    output_base_dir.mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.zip'):
            zip_path = input_dir / filename
            print(f"[DICOM->JPG] Processing {filename}...")
            process_zip_to_jpg(zip_path, output_base_dir)
            print(f"[DICOM->JPG] Finished {filename}")


# -----------------------------
# Step 2: Fan crop without SAM
# -----------------------------
def crop_ultrasound_fan(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    h, w = gray.shape

    bottom_center = (w // 2, h - 30)
    fan_width = w * 0.8
    fan_height = h * 0.8

    fan_points = np.array([
        [bottom_center[0], bottom_center[1]],
        [int(bottom_center[0] - fan_width / 2), bottom_center[1]],
        [int(w / 2 - fan_width / 3), int(h - fan_height)],
        [int(w / 2 + fan_width / 3), int(h - fan_height)],
        [int(bottom_center[0] + fan_width / 2), bottom_center[1]],
    ], np.int32)

    cv2.fillPoly(mask, [fan_points], 255)
    result = cv2.bitwise_and(img, img, mask=mask)

    x_min = max(0, int(bottom_center[0] - fan_width / 2))
    x_max = min(w, int(bottom_center[0] + fan_width / 2))
    y_min = max(0, int(h - fan_height))
    y_max = min(h, bottom_center[1])

    margin = 10
    x_min = max(0, x_min - margin)
    x_max = min(w, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(h, y_max + margin)

    cropped = result[y_min:y_max, x_min:x_max]

    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray_cropped)
    if coords is not None:
        x, y, ww, hh = cv2.boundingRect(coords)
        cropped = cropped[y:y + hh, x:x + ww]

    return cropped


def crop_folder_no_sam(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    for img_path in image_files:
        out_path = output_dir / f"cropped_{img_path.name}"
        if out_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[CROP] Failed to read {img_path}")
            continue
        try:
            cropped = crop_ultrasound_fan(img)
            cv2.imwrite(str(out_path), cropped)
        except Exception as e:
            print(f"[CROP] Error on {img_path.name}: {e}")


def batch_crop_no_sam(base_input: Path, base_output: Path):
    base_output.mkdir(parents=True, exist_ok=True)
    patient_folders = [f for f in base_input.iterdir() if f.is_dir()]
    for folder in patient_folders:
        print(f"[CROP] Processing {folder.name}")
        crop_folder_no_sam(folder, base_output / folder.name)


# -----------------------------
# Step 3: Remove black images
# -----------------------------
def is_black_image(image_path: Path, threshold: int = 20) -> bool:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    mean_pixel_value = float(np.mean(img))
    return mean_pixel_value < threshold


def remove_black_images(root_folder: Path) -> int:
    root_folder = Path(root_folder)
    image_files = [f for f in root_folder.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    deleted = 0
    for img_path in image_files:
        if is_black_image(img_path):
            try:
                img_path.unlink()
                deleted += 1
            except Exception as e:
                print(f"[CLEAN] Failed deleting {img_path}: {e}")
    print(f"[CLEAN] Removed {deleted} black images")
    return deleted


# -----------------------------
# Step 4: Build M3D input (npy + text)
# -----------------------------
DESIRED_NUM_IMAGES = 32


def sample_images(image_files, num_images=DESIRED_NUM_IMAGES):
    sequence = image_files[5:-5] if len(image_files) > 50 else image_files
    if len(sequence) < 2:
        return None
    if len(sequence) > num_images:
        step = len(sequence) / num_images
        return [sequence[int(i * step)] for i in range(num_images)]
    selected = sequence.copy()
    while len(selected) < num_images:
        selected.append(sequence[len(selected) % len(sequence)])
    return selected


transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=[256, 256], mode="bilinear"),
])


def extract_sequence_number(filename: str) -> int:
    try:
        return int(filename.split('-')[-1].split('.')[0])
    except Exception:
        return 0


def process_patient_to_npy(folder_name: str,
                           input_image_dir: Path,
                           output_dir: Path,
                           reports_dict: dict):
    try:
        input_folder = input_image_dir / folder_name
        output_folder = output_dir / folder_name

        if folder_name not in reports_dict:
            return {'patient': folder_name, 'reason': 'no_report'}

        output_folder.mkdir(parents=True, exist_ok=True)

        # Save report text
        (output_folder / 'text.txt').write_text(str(reports_dict[folder_name]))

        image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') and f.startswith('cropped_')]
        if len(image_files) == 0:
            return {'patient': folder_name, 'reason': 'no_images'}

        image_files.sort(key=extract_sequence_number)
        selected_images = sample_images(image_files)
        if selected_images is None:
            return {'patient': folder_name, 'reason': 'insufficient_images'}

        images_stack = []
        for image_file in selected_images:
            image_path = input_folder / image_file
            try:
                img = Image.open(image_path).convert('L')
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = img_array[None, ...]  # add channel dim
                img_trans = transform(img_array)
                images_stack.append(img_trans)
            except Exception as e:
                print(f"[M3D] Error processing {image_path}: {e}")

        if images_stack:
            full_sequence = np.concatenate(images_stack, axis=0)
            # Add leading dimension as in add_dimension.ipynb
            full_sequence = np.expand_dims(full_sequence, axis=0)
            np.save(output_folder / 'img.npy', full_sequence)
            return None
        else:
            return {'patient': folder_name, 'reason': 'no_valid_images'}

    except Exception as e:
        return {'patient': folder_name, 'reason': f'processing_error: {str(e)}'}


def _process_patient_to_npy_star(args):
    folder_name, cropped_dir, output_dir, reports_dict = args
    return process_patient_to_npy(folder_name, cropped_dir, output_dir, reports_dict)


def build_m3d_inputs(cropped_dir: Path,
                      reports_csv: Path,
                      output_dir: Path,
                      session_col: str = 'XNATSessionID',
                      text_col: str = 'FINDINGS',
                      workers: int = 6,
                      skipped_csv: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_df = pd.read_csv(reports_csv)
    reports_dict = dict(zip(reports_df[session_col], reports_df[text_col]))

    patient_folders = [d.name for d in cropped_dir.iterdir() if d.is_dir()]
    skipped = []

    with Pool(processes=workers) as pool:
        with tqdm(total=len(patient_folders), desc="M3D") as pbar:
            for result in pool.imap_unordered(
                _process_patient_to_npy_star,
                [(name, cropped_dir, output_dir, reports_dict) for name in patient_folders]
            ):
                if result is not None:
                    skipped.append(result)
                pbar.update(1)

    if skipped_csv is not None:
        import csv as _csv
        with open(skipped_csv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=['patient', 'reason'])
            writer.writeheader()
            writer.writerows(skipped)
    print(f"[M3D] Complete. Skipped: {len(skipped)}")


# -----------------------------
# Step 5: Create dataset JSON
# -----------------------------
def create_ultrasound_dataset_json(data_root: Path, mode: str = "val-only") -> Path:
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
        train_split = int(len(all_patients) * 0.9)
        dataset = {
            "train": create_entries(all_patients[:train_split]),
            "validation": create_entries(all_patients[train_split:]),
            "test": create_entries(all_patients[train_split:]),
        }
    elif mode == "val-only":
        dataset = {
            "train": [],
            "validation": create_entries(all_patients),
            "test": create_entries(all_patients),
        }
    else:
        raise ValueError("Invalid mode. Choose 'full' or 'val-only'")

    output_path = data_dir / "ultrasound_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"[JSON] Created at: {output_path}")
    print(f"[JSON] Validation samples: {len(dataset['validation'])}")
    return output_path


# -----------------------------
# Orchestrator / CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="End-to-end preprocessing pipeline without SAM")
    parser.add_argument('--zips_dir', type=Path, required=True, help='Input directory containing zip files')
    parser.add_argument('--jpg_out', type=Path, required=True, help='Output directory for extracted JPGs per session')
    parser.add_argument('--cropped_out', type=Path, required=True, help='Output directory for cropped JPGs per session')
    parser.add_argument('--reports_csv', type=Path, required=True, help='CSV with reports text')
    parser.add_argument('--preproc_out', type=Path, required=True, help='Output directory for M3D npy+text')
    parser.add_argument('--dataset_mode', type=str, default='val-only', choices=['full', 'val-only'])
    parser.add_argument('--session_col', type=str, default='XNATSessionID')
    parser.add_argument('--text_col', type=str, default='FINDINGS')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--skip_dicom', action='store_true', help='Skip DICOM->JPG step if already done')
    parser.add_argument('--skip_crop', action='store_true', help='Skip cropping step if already done')
    parser.add_argument('--skip_clean', action='store_true', help='Skip black image removal step')
    parser.add_argument('--skip_m3d', action='store_true', help='Skip M3D npy generation')
    parser.add_argument('--skip_json', action='store_true', help='Skip JSON creation')
    args = parser.parse_args()

    if not args.skip_dicom:
        batch_zip_to_jpg(args.zips_dir, args.jpg_out)
    else:
        print('[SKIP] DICOM->JPG')

    if not args.skip_crop:
        batch_crop_no_sam(args.jpg_out, args.cropped_out)
    else:
        print('[SKIP] Cropping')

    if not args.skip_clean:
        remove_black_images(args.cropped_out)
    else:
        print('[SKIP] Black image removal')

    if not args.skip_m3d:
        skipped_csv = args.preproc_out / 'skipped_patients.csv'
        build_m3d_inputs(
            cropped_dir=args.cropped_out,
            reports_csv=args.reports_csv,
            output_dir=args.preproc_out,
            session_col=args.session_col,
            text_col=args.text_col,
            workers=args.workers,
            skipped_csv=skipped_csv,
        )
    else:
        print('[SKIP] M3D input build')

    if not args.skip_json:
        create_ultrasound_dataset_json(args.preproc_out, mode=args.dataset_mode)
    else:
        print('[SKIP] Dataset JSON')


if __name__ == '__main__':
    main()

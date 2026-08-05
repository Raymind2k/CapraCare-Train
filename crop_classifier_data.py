"""
CapraCare - Crop YOLO bounding boxes into per-class folders.
Reads CapraCare-1/ (Roboflow YOLOv8 export) and creates CapraCare_Classification/
with images organized by disease class for classification training.
"""

from pathlib import Path
from PIL import Image
import yaml

INPUT_DIR = Path("CapraCare-1")
OUTPUT_DIR = Path("CapraCare_Classification")
PADDING = 0.10


def load_class_names(data_yaml):
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']


def crop_with_padding(img, x_center, y_center, width, height, pad_ratio=0.1):
    W, H = img.size
    cx, cy = x_center * W, y_center * H
    w, h = width * W, height * H
    w *= (1 + pad_ratio)
    h *= (1 + pad_ratio)
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(W, int(cx + w / 2))
    y2 = min(H, int(cy + h / 2))
    return img.crop((x1, y1, x2, y2))


def process_split(split_dir, class_names, output_dir):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"  Skipping {split_dir.name}: missing images/ or labels/")
        return 0

    count = 0
    img_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))

    for img_path in img_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])
            if class_id >= len(class_names):
                continue
            class_name = class_names[class_id]
            class_out_dir = output_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)
            try:
                crop = crop_with_padding(img, x_center, y_center, w, h, PADDING)
                if crop.size[0] < 20 or crop.size[1] < 20:
                    continue
                out_name = f"{img_path.stem}_crop{idx}.jpg"
                crop.save(class_out_dir / out_name, quality=95)
                count += 1
            except Exception as e:
                print(f"  Crop failed for {img_path.name}: {e}")
    return count


def main():
    print("=" * 70)
    print("CapraCare - Crop YOLO bboxes for classification")
    print("=" * 70)

    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}")
        print("   Download dataset from Roboflow first and rename folder to CapraCare-1")
        return

    data_yaml = INPUT_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"data.yaml not found in {INPUT_DIR}")
        return

    class_names = load_class_names(data_yaml)
    print(f"\nClasses found: {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for split_name in ["train", "valid", "test"]:
        split_dir = INPUT_DIR / split_name
        if split_dir.exists():
            print(f"\nProcessing {split_name}...")
            count = process_split(split_dir, class_names, OUTPUT_DIR)
            print(f"  Cropped {count} images from {split_name}")
            total += count

    print("\n" + "=" * 70)
    print(f"Cropping complete! Total: {total} cropped images")
    out_path = str(OUTPUT_DIR.absolute())
    print(f"   Output: {out_path}")
    print("=" * 70)

    print("\nPer-class counts:")
    for class_dir in sorted(OUTPUT_DIR.iterdir()):
        if class_dir.is_dir():
            n = len(list(class_dir.glob("*.jpg")))
            print(f"  {class_dir.name:35s}: {n}")


if __name__ == '__main__':
    main()
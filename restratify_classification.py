"""
CapraCare - Restratify cropped dataset into clean 70/15/15 train/val/test.
Uses stratified sampling per class with fixed seed for reproducibility.
"""

import random
import shutil
from pathlib import Path

INPUT_DIR = Path("CapraCare_Classification")
OUTPUT_DIR = Path("CapraCare_Classification_Final")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def main():
    print("=" * 70)
    print("CapraCare - Stratified 70/15/15 split")
    print("=" * 70)
    
    if not INPUT_DIR.exists():
        print(f"❌ Input not found: {INPUT_DIR}")
        print("   Run crop_classifier_data.py first.")
        return
    
    random.seed(SEED)
    
    # Clean output dir
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)
    
    class_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    
    total_train, total_val, total_test = 0, 0, 0
    
    print(f"\n{'Class':<35} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    print("-" * 70)
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ])
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val
        
        # Ensure at least 1 in val and test if class has >=3 images
        if n_total >= 3:
            if n_val == 0:
                n_val = 1
                n_train -= 1
            if n_test == 0:
                n_test = 1
                n_train -= 1
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        for split_name, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_class_dir = OUTPUT_DIR / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for img in img_list:
                shutil.copy2(img, split_class_dir / img.name)
        
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)
        
        print(f"{class_name:<35} {n_total:>6} {len(train_imgs):>6} {len(val_imgs):>5} {len(test_imgs):>5}")
    
    print("-" * 70)
    print(f"{'TOTAL':<35} {total_train+total_val+total_test:>6} {total_train:>6} {total_val:>5} {total_test:>5}")
    
    print(f"\n✅ Split complete. Output: {OUTPUT_DIR.absolute()}")


if __name__ == '__main__':
    main()
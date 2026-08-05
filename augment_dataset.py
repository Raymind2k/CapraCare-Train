"""
CapraCare - Custom Augmentation Script
Replicates the Roboflow augmentation pipeline locally so we can document
and explain exactly what was done. Applies augmentations to training set
only (val/test are copied unchanged to prevent data leakage).

Augmentations applied:
  - Horizontal flip (50% probability)
  - Rotation: ±15 degrees
  - Brightness: ±20%
  - Exposure (contrast): ±15%
  - Gaussian blur: 0 to 1.5 pixels
  - Gaussian noise: up to 2% of pixel intensity
  - Output multiplier: 3x (each image -> 3 augmented copies)
"""

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ============================================================
# Configuration
# ============================================================
# Input: the stratified split we already created
INPUT_DIR = Path("CapraCare_Classification_Final")

# Output: new augmented dataset
OUTPUT_DIR = Path("CapraCare_Classification_Augmented")

# Augmentation parameters (matching Roboflow settings)
AUG_MULTIPLIER = 3              # each train image -> 3 augmented copies
HFLIP_PROB = 0.5                # 50% chance of horizontal flip
ROTATION_DEGREES = 15           # ±15 degrees
BRIGHTNESS_RANGE = 0.20         # ±20%
EXPOSURE_RANGE = 0.15           # ±15% (PIL contrast)
BLUR_MAX_RADIUS = 1.5           # max gaussian blur radius in pixels
NOISE_STD = 0.02                # gaussian noise std as fraction of 255

# Reproducibility
RANDOM_SEED = 42


# ============================================================
# Augmentation functions
# ============================================================
def random_horizontal_flip(img):
    """Flip image horizontally with HFLIP_PROB probability."""
    if random.random() < HFLIP_PROB:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_rotation(img):
    """Rotate by random angle in [-ROTATION_DEGREES, +ROTATION_DEGREES]."""
    angle = random.uniform(-ROTATION_DEGREES, ROTATION_DEGREES)
    # Resample with bilinear, fill empty corners with edge color
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=None)


def random_brightness(img):
    """Adjust brightness by ±BRIGHTNESS_RANGE."""
    factor = 1.0 + random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
    return ImageEnhance.Brightness(img).enhance(factor)


def random_exposure(img):
    """Adjust contrast (closest PIL equivalent to 'exposure') by ±EXPOSURE_RANGE."""
    factor = 1.0 + random.uniform(-EXPOSURE_RANGE, EXPOSURE_RANGE)
    return ImageEnhance.Contrast(img).enhance(factor)


def random_blur(img):
    """Apply Gaussian blur with random radius in [0, BLUR_MAX_RADIUS]."""
    radius = random.uniform(0, BLUR_MAX_RADIUS)
    if radius > 0.1:  # skip imperceptible blurs
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def random_noise(img):
    """Add Gaussian noise to the image."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, NOISE_STD * 255, arr.shape)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_augmentations(img):
    """Apply the full augmentation pipeline to one image."""
    img = random_horizontal_flip(img)
    img = random_rotation(img)
    img = random_brightness(img)
    img = random_exposure(img)
    img = random_blur(img)
    img = random_noise(img)
    return img


# ============================================================
# Main pipeline
# ============================================================
def augment_training_set(train_in, train_out):
    """Apply AUG_MULTIPLIER augmentations to each training image."""
    total_originals = 0
    total_augmented = 0
    
    class_dirs = [d for d in train_in.iterdir() if d.is_dir()]
    print(f"\n[Train] Augmenting {len(class_dirs)} classes...")
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        out_class_dir = train_out / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        
        images = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ])
        
        for img_path in images:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  ⚠️  Skipping {img_path.name}: {e}")
                continue
            
            # Save original
            out_orig = out_class_dir / f"{img_path.stem}_orig{img_path.suffix}"
            img.save(out_orig, quality=95)
            total_originals += 1
            
            # Generate AUG_MULTIPLIER augmented versions
            for i in range(AUG_MULTIPLIER):
                aug_img = apply_augmentations(img)
                out_aug = out_class_dir / f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                aug_img.save(out_aug, quality=95)
                total_augmented += 1
        
        n_originals = len(images)
        n_total = n_originals * (1 + AUG_MULTIPLIER)
        print(f"  {class_name:35s}: {n_originals:3d} -> {n_total:4d}")
    
    print(f"\n  Total: {total_originals} originals + {total_augmented} augmented "
          f"= {total_originals + total_augmented} training images")


def copy_split_unchanged(split_in, split_out, split_name):
    """Copy val or test split unchanged (NO augmentation - prevents leakage)."""
    if not split_in.exists():
        print(f"\n[{split_name}] No source directory at {split_in}, skipping.")
        return
    
    class_dirs = [d for d in split_in.iterdir() if d.is_dir()]
    print(f"\n[{split_name}] Copying {len(class_dirs)} classes (NO augmentation)...")
    
    total = 0
    for class_dir in sorted(class_dirs):
        out_class_dir = split_out / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ]
        
        for img_path in images:
            shutil.copy2(img_path, out_class_dir / img_path.name)
            total += 1
        
        print(f"  {class_dir.name:35s}: {len(images):3d} images copied")
    
    print(f"  Total: {total} images copied unchanged")


def main():
    print("=" * 70)
    print("CapraCare - Custom Augmentation Pipeline")
    print("=" * 70)
    print(f"Input:  {INPUT_DIR.absolute()}")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print(f"\nAugmentation settings:")
    print(f"  Horizontal flip probability: {HFLIP_PROB}")
    print(f"  Rotation range:              ±{ROTATION_DEGREES}°")
    print(f"  Brightness range:            ±{BRIGHTNESS_RANGE*100:.0f}%")
    print(f"  Exposure (contrast) range:   ±{EXPOSURE_RANGE*100:.0f}%")
    print(f"  Gaussian blur max radius:    {BLUR_MAX_RADIUS} px")
    print(f"  Gaussian noise std:          {NOISE_STD*100:.0f}% of 255")
    print(f"  Augmentation multiplier:     {AUG_MULTIPLIER}x")
    print(f"  Random seed:                 {RANDOM_SEED}")
    
    # Reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Validate input
    if not INPUT_DIR.exists():
        print(f"\n❌ Input directory not found: {INPUT_DIR}")
        print("   Make sure you've run restratify_classification.py first.")
        return
    
    # Check for existing output
    if OUTPUT_DIR.exists():
        print(f"\n⚠️  Output directory already exists: {OUTPUT_DIR}")
        response = input("   Delete and recreate? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(OUTPUT_DIR)
            print("   Deleted.")
        else:
            print("   Aborting.")
            return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    augment_training_set(INPUT_DIR / "train", OUTPUT_DIR / "train")
    copy_split_unchanged(INPUT_DIR / "val", OUTPUT_DIR / "val", "Val")
    copy_split_unchanged(INPUT_DIR / "test", OUTPUT_DIR / "test", "Test")
    
    print("\n" + "=" * 70)
    print("✅ Augmentation complete!")
    print("=" * 70)
    print(f"\nNew dataset saved to: {OUTPUT_DIR.absolute()}")
    print("\nFolder structure:")
    print(f"  {OUTPUT_DIR.name}/")
    print(f"    train/   (with augmentations, {AUG_MULTIPLIER+1}x original size)")
    print(f"    val/     (unchanged - never augmented)")
    print(f"    test/    (unchanged - never augmented)")
    print("\nNote: val and test are NOT augmented to prevent data leakage.")
    print("Only the training set gets augmented.")


if __name__ == '__main__':
    main()
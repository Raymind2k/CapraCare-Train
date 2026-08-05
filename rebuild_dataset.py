from pathlib import Path
import shutil
import random
from sklearn.model_selection import train_test_split

random.seed(42)

SRC = Path(r"D:\CapraCare\CapraCare-1\classification_dataset\train")
DST = Path(r"D:\CapraCare\CapraCare-1\classification_dataset_v2")

for cls_dir in SRC.iterdir():

    if not cls_dir.is_dir():
        continue

    images = list(cls_dir.glob("*"))

    if len(images) < 3:
        print(f"Skipping {cls_dir.name}")
        continue

    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    valid_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=42
    )

    for split, img_list in {
        "train": train_imgs,
        "valid": valid_imgs,
        "test": test_imgs
    }.items():

        out_dir = DST / split / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img in img_list:
            shutil.copy2(img, out_dir / img.name)

print("Done!")
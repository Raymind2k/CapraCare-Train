from pathlib import Path
import shutil

ROOT = Path(r"D:\CapraCare\CapraCare-1")

classes = [
    "parasitic_mange",
    "parasitic_myiasis",
    "skin_infection_abscess",
    "skin_infection_dermatitis",
    "skin_infection_dermatophilosis",
    "viral_orf",
    "viral_ppr",
    "wounds_footrot"
]

OUT_ROOT = ROOT / "classification_dataset"

for split in ["train", "valid", "test"]:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    print(f"Processing {split}...")

    for cls in classes:
        (OUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

    count = 0

    for label_file in label_dir.glob("*.txt"):

        with open(label_file, "r") as f:
            line = f.readline().strip()

        if not line:
            continue

        class_id = int(line.split()[0])
        class_name = classes[class_id]

        image_file = image_dir / (label_file.stem + ".jpg")

        if not image_file.exists():
            image_file = image_dir / (label_file.stem + ".png")

        if image_file.exists():
            shutil.copy2(
                image_file,
                OUT_ROOT / split / class_name / image_file.name
            )
            count += 1

    print(f"{split}: {count} images copied")

print("\nConversion complete!")
print(f"Dataset saved to: {OUT_ROOT}")
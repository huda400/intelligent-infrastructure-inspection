from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = PROJECT_ROOT / "data" / "yolo_crack"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_stems(split):
    images_dir = YOLO_DIR / "images" / split

    return {
        p.stem
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


train = get_stems("train")
val = get_stems("val")
test = get_stems("test")

train_val = train | val

train_test_overlap = train & test
val_test_overlap = val & test
train_val_test_overlap = train_val & test

print("=" * 60)
print("YOLO DATASET LEAKAGE CHECK")
print("=" * 60)

print(f"Train images: {len(train)}")
print(f"Val images:   {len(val)}")
print(f"Test images:  {len(test)}")

print(f"\nTrain ↔ Test overlap:     {len(train_test_overlap)}")
print(f"Val ↔ Test overlap:       {len(val_test_overlap)}")
print(f"Train+Val ↔ Test overlap: {len(train_val_test_overlap)}")

if train_val_test_overlap:
    print("\n❌ LEAKAGE DETECTED!")
    for name in sorted(train_val_test_overlap):
        print(f"  {name}")
else:
    print("\n🎉 NO DATA LEAKAGE!")
    print("Train/Val and Test are completely separated.")
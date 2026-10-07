from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASKS_DIR = PROJECT_ROOT / "data" / "dataset" / "masks"


# =========================
# Analyze All Masks
# =========================

mask_files = list(MASKS_DIR.glob("*.jpg"))

print("=" * 50)
print("FULL MASK PIXEL ANALYSIS")
print("=" * 50)

print(f"\nTotal masks: {len(mask_files)}")


empty_masks = 0
unique_values = set()

pixel_counts = Counter()

min_values = []
max_values = []


for i, mask_path in enumerate(mask_files):

    mask = np.array(
        Image.open(mask_path).convert("L")
    )

    unique_values.update(np.unique(mask).tolist())

    pixel_counts.update(mask.flatten().tolist())

    min_values.append(mask.min())
    max_values.append(mask.max())

    if mask.max() == 0:
        empty_masks += 1


# =========================
# Results
# =========================

print("\n--- Results ---")

print(f"Empty masks: {empty_masks}")

print(f"Minimum pixel value across masks: {min(min_values)}")

print(f"Maximum pixel value across masks: {max(max_values)}")

print(f"Total unique pixel values: {len(unique_values)}")


print("\nUnique pixel values:")

print(sorted(unique_values))


# =========================
# Pixel Distribution
# =========================

print("\n--- Most Common Pixel Values ---")

for value, count in pixel_counts.most_common(20):

    print(f"Pixel {value:3d}: {count:,}")


# =========================
# Binary Threshold Analysis
# =========================

print("\n--- Threshold Analysis ---")

for threshold in [1, 10, 50, 100, 127, 128, 200, 240]:

    crack_pixels = 0
    total_pixels = 0

    for mask_path in mask_files:

        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        crack_pixels += np.sum(mask >= threshold)
        total_pixels += mask.size

    percentage = (crack_pixels / total_pixels) * 100

    print(
        f"Threshold {threshold:3d}: "
        f"{percentage:.4f}% crack pixels"
    )
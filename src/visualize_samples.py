"""Visualize sample images, masks, and segmentation overlays."""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_THRESHOLD = 127
NUM_SAMPLES = 6
RANDOM_SEED = 42


def main() -> None:
    """Display random image/mask/overlay samples."""
    image_files = sorted(
        path
        for path in IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    random.seed(RANDOM_SEED)
    samples = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

    for image_path in samples:
        mask_path = MASKS_DIR / image_path.name

        if not mask_path.exists():
            print(f"Skipping {image_path.name}: mask not found.")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        binary_mask = mask >= MASK_THRESHOLD

        overlay = image.copy()
        overlay[binary_mask] = [255, 0, 0]

        blended = (
            0.6 * image + 0.4 * overlay
        ).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        axes[2].imshow(blended)
        axes[2].set_title("Mask Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

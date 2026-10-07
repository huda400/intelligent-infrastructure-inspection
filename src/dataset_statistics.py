from pathlib import Path
from collections import Counter
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset"

IMAGES_PATH = DATASET_PATH / "images"
MASKS_PATH = DATASET_PATH / "masks"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(folder):
    """Return all JPG image files inside a folder."""
    return sorted(folder.glob("*.jpg"))


def calculate_crack_ratio(mask):
    """
    Calculate the percentage of pixels considered as crack.

    The dataset masks contain values close to:
    0   -> background
    255 -> crack

    We use a threshold of 128 to convert the mask
    conceptually into binary foreground/background.
    """

    binary_mask = mask >= 128

    crack_pixels = np.count_nonzero(binary_mask)
    total_pixels = mask.size

    if total_pixels == 0:
        return 0.0

    return crack_pixels / total_pixels * 100


def count_connected_components(mask):
    """
    Count connected crack regions in a mask.
    """

    binary_mask = (mask >= 128).astype(np.uint8)

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    # Label 0 is the background.
    return max(0, num_labels - 1)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 60)
    print("CRACK SEGMENTATION DATASET CHARACTERIZATION")
    print("=" * 60)

    print(f"\nDataset path:")
    print(DATASET_PATH)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    image_files = get_image_files(IMAGES_PATH)
    mask_files = get_image_files(MASKS_PATH)

    print("\n--- Dataset Size ---")

    print(f"Total images: {len(image_files)}")
    print(f"Total masks:  {len(mask_files)}")

    # --------------------------------------------------------
    # Prepare mask lookup
    # --------------------------------------------------------

    mask_lookup = {
        mask.name: mask
        for mask in mask_files
    }

    # --------------------------------------------------------
    # Statistics containers
    # --------------------------------------------------------

    crack_ratios = []
    connected_components = []

    positive_images = 0
    negative_images = 0

    image_dimensions = Counter()
    mask_dimensions = Counter()

    # --------------------------------------------------------
    # Analyze images and masks
    # --------------------------------------------------------

    for image_path in image_files:

        mask_path = mask_lookup.get(image_path.name)

        if mask_path is None:
            continue

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        # Image dimensions
        height, width = image.shape[:2]
        image_dimensions[(width, height)] += 1

        # Mask dimensions
        mask_height, mask_width = mask.shape[:2]
        mask_dimensions[(mask_width, mask_height)] += 1

        # Crack percentage
        crack_ratio = calculate_crack_ratio(mask)

        crack_ratios.append(crack_ratio)

        # Positive / Negative classification
        if crack_ratio == 0:
            negative_images += 1
        else:
            positive_images += 1

        # Connected components
        components = count_connected_components(mask)
        connected_components.append(components)

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n--- Positive / Negative Distribution ---")

    total_analyzed = positive_images + negative_images

    print(f"Analyzed images:  {total_analyzed}")
    print(f"Positive images:  {positive_images}")
    print(f"Negative images:  {negative_images}")

    if total_analyzed > 0:

        positive_ratio = positive_images / total_analyzed * 100
        negative_ratio = negative_images / total_analyzed * 100

        print(f"Positive ratio:   {positive_ratio:.2f}%")
        print(f"Negative ratio:   {negative_ratio:.2f}%")

    # --------------------------------------------------------
    # Crack area statistics
    # --------------------------------------------------------

    print("\n--- Crack Area Distribution ---")

    if crack_ratios:

        crack_ratios_np = np.array(crack_ratios)

        print(f"Minimum crack area:  {np.min(crack_ratios_np):.4f}%")
        print(f"Maximum crack area:  {np.max(crack_ratios_np):.4f}%")
        print(f"Mean crack area:     {np.mean(crack_ratios_np):.4f}%")
        print(f"Median crack area:   {np.median(crack_ratios_np):.4f}%")
        print(f"Std deviation:       {np.std(crack_ratios_np):.4f}%")

        print("\nPercentiles:")

        print(f"25th percentile:     {np.percentile(crack_ratios_np, 25):.4f}%")
        print(f"50th percentile:     {np.percentile(crack_ratios_np, 50):.4f}%")
        print(f"75th percentile:     {np.percentile(crack_ratios_np, 75):.4f}%")
        print(f"90th percentile:     {np.percentile(crack_ratios_np, 90):.4f}%")
        print(f"95th percentile:     {np.percentile(crack_ratios_np, 95):.4f}%")

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    print("\n--- Connected Crack Regions ---")

    if connected_components:

        components_np = np.array(connected_components)

        print(f"Minimum components:  {np.min(components_np)}")
        print(f"Maximum components:  {np.max(components_np)}")
        print(f"Mean components:     {np.mean(components_np):.2f}")
        print(f"Median components:   {np.median(components_np):.2f}")

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    print("\n--- Image Dimensions ---")

    for dimension, count in image_dimensions.most_common():
        print(f"{dimension}: {count}")

    # --------------------------------------------------------
    # Mask dimensions
    # --------------------------------------------------------

    print("\n--- Mask Dimensions ---")

    for dimension, count in mask_dimensions.most_common():
        print(f"{dimension}: {count}")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET CHARACTERIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""Audit the source crack segmentation dataset."""

from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_files(directory: Path) -> list[Path]:
    """Return supported files from a directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_names(directory: Path) -> set[str]:
    """Return supported filenames from a directory."""
    return {path.name for path in get_files(directory)}


def get_extensions(directory: Path) -> Counter:
    """Count file extensions in a directory."""
    return Counter(path.suffix.lower() for path in get_files(directory))


def check_dimensions(
    image_files: list[Path],
    mask_lookup: dict[str, Path],
) -> tuple[Counter, Counter, int]:
    """Check image/mask readability and dimensions."""
    image_dimensions = Counter()
    mask_dimensions = Counter()
    unreadable = 0

    for image_path in image_files:
        mask_path = mask_lookup.get(image_path.name)

        try:
            with Image.open(image_path) as image:
                image_dimensions[image.size] += 1

            if mask_path is not None:
                with Image.open(mask_path) as mask:
                    mask_dimensions[mask.size] += 1

        except (OSError, ValueError):
            unreadable += 1

    return image_dimensions, mask_dimensions, unreadable


def main() -> None:
    """Run the complete source dataset audit."""
    images_dir = DATASET_DIR / "images"
    masks_dir = DATASET_DIR / "masks"

    train_images_dir = DATASET_DIR / "train" / "images"
    train_masks_dir = DATASET_DIR / "train" / "masks"

    test_images_dir = DATASET_DIR / "test" / "images"
    test_masks_dir = DATASET_DIR / "test" / "masks"

    all_images = get_names(images_dir)
    all_masks = get_names(masks_dir)

    train_images = get_names(train_images_dir)
    train_masks = get_names(train_masks_dir)

    test_images = get_names(test_images_dir)
    test_masks = get_names(test_masks_dir)

    image_files = get_files(images_dir)
    mask_lookup = {
        path.name: path
        for path in get_files(masks_dir)
    }

    print("=" * 60)
    print("CRACK SEGMENTATION DATASET AUDIT")
    print("=" * 60)

    print(f"\nDataset: {DATASET_DIR}")

    print("\n--- Dataset Size ---")
    print(f"Images: {len(all_images)}")
    print(f"Masks:  {len(all_masks)}")

    print("\n--- Train / Test Split ---")
    print(f"Train images: {len(train_images)}")
    print(f"Train masks:  {len(train_masks)}")
    print(f"Test images: {len(test_images)}")
    print(f"Test masks:  {len(test_masks)}")

    print("\n--- File Extensions ---")
    print(f"Images: {get_extensions(images_dir)}")
    print(f"Masks:  {get_extensions(masks_dir)}")

    print("\n--- Image / Mask Matching ---")
    images_without_masks = all_images - all_masks
    masks_without_images = all_masks - all_images

    print(f"Images without masks: {len(images_without_masks)}")
    print(f"Masks without images: {len(masks_without_images)}")

    print("\n--- Split Membership ---")
    print(
        f"Train images found in main image set: "
        f"{len(train_images & all_images)}"
    )
    print(
        f"Test images found in main image set: "
        f"{len(test_images & all_images)}"
    )
    print(
        f"Train masks found in main mask set: "
        f"{len(train_masks & all_masks)}"
    )
    print(
        f"Test masks found in main mask set: "
        f"{len(test_masks & all_masks)}"
    )

    print("\n--- Files Outside Train/Test Split ---")
    print(
        f"Images not assigned to train/test: "
        f"{len(all_images - train_images - test_images)}"
    )
    print(
        f"Masks not assigned to train/test: "
        f"{len(all_masks - train_masks - test_masks)}"
    )

    print("\n--- Source Dataset Consistency ---")
    print(
        f"Train images without local split masks: "
        f"{len(train_images - train_masks)}"
    )
    print(
        f"Test images without local split masks: "
        f"{len(test_images - test_masks)}"
    )

    image_dimensions, mask_dimensions, unreadable = check_dimensions(
        image_files,
        mask_lookup,
    )

    print("\n--- Image / Mask Dimensions ---")
    print(f"Image dimensions: {dict(image_dimensions)}")
    print(f"Mask dimensions:  {dict(mask_dimensions)}")
    print(f"Unreadable files: {unreadable}")

    print("\n--- Audit Summary ---")

    if images_without_masks or masks_without_images:
        print("⚠️ Global image/mask mismatch detected.")
    else:
        print("✓ Global image/mask filenames match.")

    if train_images - train_masks:
        print(
            "⚠️ Train split contains images whose masks are stored "
            "outside the local train/masks directory."
        )
    else:
        print("✓ Train split image/mask structure is complete.")

    if test_images - test_masks:
        print("⚠️ Test split contains images without local masks.")
    else:
        print("✓ Test split image/mask structure is complete.")

    if unreadable:
        print("⚠️ Unreadable files detected.")
    else:
        print("✓ All checked files are readable.")

    print("\n" + "=" * 60)
    print("DATASET AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

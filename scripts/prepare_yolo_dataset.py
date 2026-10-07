from pathlib import Path
import random
import shutil

import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_crack"

TRAIN_IMAGES_DIR = DATASET_DIR / "train" / "images"
TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"
GLOBAL_IMAGES_DIR = DATASET_DIR / "images"
GLOBAL_MASKS_DIR = DATASET_DIR / "masks"

YOLO_IMAGES_DIR = OUTPUT_DIR / "images"
YOLO_LABELS_DIR = OUTPUT_DIR / "labels"


# ============================================================
# CONFIGURATION
# ============================================================

VAL_RATIO = 0.20
RANDOM_SEED = 42

CLASS_ID = 0
CLASS_NAME = "crack"

MASK_THRESHOLD = 127

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# FILE HELPERS
# ============================================================

def get_image_files(directory: Path) -> list[Path]:
    """Return all supported image files in a directory."""

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_mask_lookup(directory: Path) -> dict[str, Path]:
    """Build image filename -> mask path lookup."""

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Mask directory not found: {directory}"
        )

    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }


# ============================================================
# SPLITTING
# ============================================================

def split_train_validation(
    image_paths: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split images into train and validation sets."""

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    shuffled = image_paths.copy()

    random.Random(seed).shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)

    val_images = shuffled[:val_size]
    train_images = shuffled[val_size:]

    return train_images, val_images


# ============================================================
# NONCRACK SELECTION
# ============================================================

def get_noncrack_images(
    images_dir: Path,
    mask_lookup: dict[str, Path],
    excluded_names: set[str],
) -> list[Path]:
    """
    Find confirmed noncrack images.

    Conditions:
    1. Filename starts with 'noncrack_'
    2. Image has a corresponding mask
    3. Mask contains no crack pixels
    4. Image is NOT part of the locked test set
    """

    noncrack_images = []

    for image_path in get_image_files(images_dir):

        # Only use explicitly named noncrack samples.
        if not image_path.name.startswith("noncrack_"):
            continue

        # Never allow test images into training/validation.
        if image_path.name in excluded_names:
            continue

        mask_path = mask_lookup.get(image_path.name)

        if mask_path is None:
            continue

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            continue

        # Confirm that the mask is completely empty.
        if not (mask > MASK_THRESHOLD).any():
            noncrack_images.append(image_path)

    return sorted(noncrack_images)


# ============================================================
# YOLO MASK CONVERSION
# ============================================================

def mask_to_yolo_labels(
    mask_path: Path,
    image_width: int,
    image_height: int,
) -> list[str]:
    """
    Convert a binary segmentation mask into YOLO segmentation labels.
    """

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise OSError(
            f"Could not read mask: {mask_path}"
        )

    binary_mask = (
        mask > MASK_THRESHOLD
    ).astype("uint8") * 255

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    labels = []

    for contour in contours:

        if len(contour) < 3:
            continue

        points = contour.reshape(-1, 2)

        normalized_points = []

        for x, y in points:

            x_normalized = x / image_width
            y_normalized = y / image_height

            normalized_points.extend(
                [
                    x_normalized,
                    y_normalized,
                ]
            )

        # Need at least 3 points = 6 coordinates.
        if len(normalized_points) < 6:
            continue

        values = [
            str(CLASS_ID),
            *[
                f"{value:.6f}"
                for value in normalized_points
            ],
        ]

        labels.append(" ".join(values))

    return labels


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_sample(
    image_path: Path,
    mask_lookup: dict[str, Path],
    output_images_dir: Path,
    output_labels_dir: Path,
) -> None:
    """Copy image and create its YOLO segmentation label."""

    output_image_path = (
        output_images_dir / image_path.name
    )

    output_label_path = (
        output_labels_dir / f"{image_path.stem}.txt"
    )

    # Copy image.
    shutil.copy2(
        image_path,
        output_image_path,
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise OSError(
            f"Could not read image: {image_path}"
        )

    image_height, image_width = image.shape[:2]

    mask_path = mask_lookup.get(
        image_path.name
    )

    if mask_path is None:
        raise FileNotFoundError(
            f"No mask found for image: {image_path.name}"
        )

    labels = mask_to_yolo_labels(
        mask_path,
        image_width,
        image_height,
    )

    output_label_path.write_text(
        "\n".join(labels),
        encoding="utf-8",
    )


# ============================================================
# DATASET SAFETY CHECKS
# ============================================================

def check_no_overlap(
    train_images: list[Path],
    val_images: list[Path],
    test_images: list[Path],
) -> None:
    """
    Make sure train, validation, and test are completely separate.
    """

    train_names = {
        path.name for path in train_images
    }

    val_names = {
        path.name for path in val_images
    }

    test_names = {
        path.name for path in test_images
    }

    train_val_overlap = (
        train_names & val_names
    )

    train_test_overlap = (
        train_names & test_names
    )

    val_test_overlap = (
        val_names & test_names
    )

    if train_val_overlap:
        raise RuntimeError(
            "DATA LEAKAGE: Train and Validation overlap detected:\n"
            + "\n".join(
                sorted(train_val_overlap)
            )
        )

    if train_test_overlap:
        raise RuntimeError(
            "DATA LEAKAGE: Train and Test overlap detected:\n"
            + "\n".join(
                sorted(train_test_overlap)
            )
        )

    if val_test_overlap:
        raise RuntimeError(
            "DATA LEAKAGE: Validation and Test overlap detected:\n"
            + "\n".join(
                sorted(val_test_overlap)
            )
        )

    print("\nDataset separation check: PASS")


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def prepare_output_directories() -> None:
    """Create a clean YOLO output directory."""

    if OUTPUT_DIR.exists():
        print(
            "\nRemoving previous YOLO output directory..."
        )

        shutil.rmtree(OUTPUT_DIR)

    for split in [
        "train",
        "val",
        "test",
    ]:

        (
            YOLO_IMAGES_DIR / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            YOLO_LABELS_DIR / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# DATA.YAML
# ============================================================

def create_data_yaml() -> None:
    """Create YOLO dataset configuration."""

    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
"""

    (
        OUTPUT_DIR / "data.yaml"
    ).write_text(
        yaml_content,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("YOLO DATASET PREPARATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate source directories
    # --------------------------------------------------------

    required_directories = [
        TRAIN_IMAGES_DIR,
        TEST_IMAGES_DIR,
        GLOBAL_IMAGES_DIR,
        GLOBAL_MASKS_DIR,
    ]

    for directory in required_directories:

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required directory not found: {directory}"
            )

    # --------------------------------------------------------
    # Build mask lookup
    # --------------------------------------------------------

    mask_lookup = build_mask_lookup(
        GLOBAL_MASKS_DIR
    )

    print(
        f"Global masks: {len(mask_lookup)}"
    )

    # --------------------------------------------------------
    # LOCK TEST SET FIRST
    # --------------------------------------------------------

    test_images = get_image_files(
        TEST_IMAGES_DIR
    )

    test_names = {
        path.name
        for path in test_images
    }

    print(
        f"Locked test images: {len(test_images)}"
    )

    # --------------------------------------------------------
    # POSITIVE IMAGES
    # --------------------------------------------------------

    positive_images = get_image_files(
        TRAIN_IMAGES_DIR
    )

    print(
        f"Original positive images: "
        f"{len(positive_images)}"
    )

    # Safety check:
    # original positive images must not overlap test.
    positive_test_overlap = {
        path.name
        for path in positive_images
    } & test_names

    if positive_test_overlap:

        raise RuntimeError(
            "DATA LEAKAGE: Original positive training "
            "images overlap with test images."
        )

    # --------------------------------------------------------
    # NONCRACK IMAGES
    # --------------------------------------------------------

    noncrack_images = get_noncrack_images(
        GLOBAL_IMAGES_DIR,
        mask_lookup,
        excluded_names=test_names,
    )

    print(
        f"Confirmed noncrack images "
        f"available for Train/Val: "
        f"{len(noncrack_images)}"
    )

    # --------------------------------------------------------
    # SPLIT POSITIVE IMAGES
    # --------------------------------------------------------

    positive_train, positive_val = (
        split_train_validation(
            positive_images,
            VAL_RATIO,
            RANDOM_SEED,
        )
    )

    # --------------------------------------------------------
    # SPLIT NONCRACK IMAGES
    # --------------------------------------------------------

    noncrack_train, noncrack_val = (
        split_train_validation(
            noncrack_images,
            VAL_RATIO,
            RANDOM_SEED,
        )
    )

    print(
        f"Positive train:       "
        f"{len(positive_train)}"
    )

    print(
        f"Positive validation:  "
        f"{len(positive_val)}"
    )

    print(
        f"Noncrack train:       "
        f"{len(noncrack_train)}"
    )

    print(
        f"Noncrack validation:  "
        f"{len(noncrack_val)}"
    )

    # --------------------------------------------------------
    # COMBINE TRAIN / VAL
    # --------------------------------------------------------

    train_images = (
        positive_train
        + noncrack_train
    )

    val_images = (
        positive_val
        + noncrack_val
    )

    # Shuffle combined datasets.
    random.Random(
        RANDOM_SEED
    ).shuffle(train_images)

    random.Random(
        RANDOM_SEED
    ).shuffle(val_images)

    # --------------------------------------------------------
    # FINAL LEAKAGE CHECK BEFORE WRITING
    # --------------------------------------------------------

    check_no_overlap(
        train_images,
        val_images,
        test_images,
    )

    # --------------------------------------------------------
    # CREATE OUTPUT
    # --------------------------------------------------------

    prepare_output_directories()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nProcessing training images..."
    )

    for image_path in train_images:

        process_sample(
            image_path,
            mask_lookup,
            YOLO_IMAGES_DIR / "train",
            YOLO_LABELS_DIR / "train",
        )

    print(
        f"Training samples created: "
        f"{len(train_images)}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\nProcessing validation images..."
    )

    for image_path in val_images:

        process_sample(
            image_path,
            mask_lookup,
            YOLO_IMAGES_DIR / "val",
            YOLO_LABELS_DIR / "val",
        )

    print(
        f"Validation samples created: "
        f"{len(val_images)}"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print(
        "\nProcessing test images..."
    )

    for image_path in test_images:

        process_sample(
            image_path,
            mask_lookup,
            YOLO_IMAGES_DIR / "test",
            YOLO_LABELS_DIR / "test",
        )

    print(
        f"Test samples created: "
        f"{len(test_images)}"
    )

    # --------------------------------------------------------
    # DATA.YAML
    # --------------------------------------------------------

    create_data_yaml()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("YOLO DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Train:       {len(train_images)}"
    )

    print(
        f"Validation:  {len(val_images)}"
    )

    print(
        f"Test:        {len(test_images)}"
    )

    print(
        f"Output:      {OUTPUT_DIR}"
    )

    print(
        f"YAML:        {OUTPUT_DIR / 'data.yaml'}"
    )

    print(
        "\n🔒 Test set remained completely isolated."
    )
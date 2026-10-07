from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = PROJECT_ROOT / "data" / "yolo_crack"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_images(directory: Path):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_split(split: str):
    images_dir = YOLO_DIR / "images" / split
    labels_dir = YOLO_DIR / "labels" / split

    images = get_images(images_dir)
    labels = sorted(labels_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    extra_labels = label_stems - image_stems

    invalid_lines = []
    out_of_range = []
    empty_labels = 0
    object_count = 0

    for label_path in labels:
        text = label_path.read_text(encoding="utf-8").strip()

        # Empty label = no crack object
        if not text:
            empty_labels += 1
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            parts = line.split()

            # class + at least 3 (x,y) points
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                invalid_lines.append(
                    f"{label_path.name}: line {line_number}"
                )
                continue

            try:
                class_id = int(parts[0])
                coordinates = [float(x) for x in parts[1:]]
            except ValueError:
                invalid_lines.append(
                    f"{label_path.name}: line {line_number}"
                )
                continue

            if class_id != 0:
                invalid_lines.append(
                    f"{label_path.name}: invalid class {class_id}"
                )

            if any(x < 0 or x > 1 for x in coordinates):
                out_of_range.append(
                    f"{label_path.name}: line {line_number}"
                )

            object_count += 1

    print(f"\n{'=' * 60}")
    print(f"VALIDATING: {split.upper()}")
    print(f"{'=' * 60}")

    print(f"Images:              {len(images)}")
    print(f"Labels:              {len(labels)}")
    print(f"Empty labels:        {empty_labels}")
    print(f"Objects:             {object_count}")
    print(f"Missing labels:      {len(missing_labels)}")
    print(f"Extra labels:        {len(extra_labels)}")
    print(f"Invalid lines:       {len(invalid_lines)}")
    print(f"Out-of-range coords: {len(out_of_range)}")

    if missing_labels:
        print("\nMissing labels:")
        for name in sorted(missing_labels)[:10]:
            print(f"  {name}")

    if extra_labels:
        print("\nExtra labels:")
        for name in sorted(extra_labels)[:10]:
            print(f"  {name}")

    if invalid_lines:
        print("\nInvalid label lines:")
        for item in invalid_lines[:10]:
            print(f"  {item}")

    if out_of_range:
        print("\nOut-of-range coordinates:")
        for item in out_of_range[:10]:
            print(f"  {item}")

    return (
        len(images) == len(labels)
        and not missing_labels
        and not extra_labels
        and not invalid_lines
        and not out_of_range
    )


def main():
    print("=" * 60)
    print("YOLO DATASET VALIDATION")
    print("=" * 60)

    if not YOLO_DIR.exists():
        raise FileNotFoundError(f"YOLO dataset not found: {YOLO_DIR}")

    results = {}

    for split in ["train", "val", "test"]:
        results[split] = validate_split(split)

    print(f"\n{'=' * 60}")
    print("FINAL RESULT")
    print(f"{'=' * 60}")

    for split, passed in results.items():
        print(f"{split.upper():<10}: {'PASS ✅' if passed else 'FAIL ❌'}")

    if all(results.values()):
        print("\n🎉 YOLO DATASET VALIDATION PASSED!")
    else:
        print("\n⚠️ YOLO DATASET VALIDATION FAILED.")


if __name__ == "__main__":
    main()
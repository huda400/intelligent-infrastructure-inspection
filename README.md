# Intelligent Infrastructure Inspection System

A computer vision project for **crack detection and instance segmentation in infrastructure images** using YOLO11s-seg. The project investigates dataset quality, compares training configurations, evaluates pixel-level segmentation performance, and analyzes model errors across different image sources and crack sizes.

## Project Highlights

- Audited image-mask annotations and investigated empty-mask cases.
- Prepared and validated a YOLO segmentation dataset containing **5,188 images**.
- Compared four training configurations: Baseline, Exp1, Exp2, and Exp3.
- Evaluated segmentation quality using crack-only IoU and Dice, detection misses, false positives, and Ultralytics validation metrics.
- Selected an operating confidence threshold using validation data.
- Analyzed test performance by image source and crack size to identify generalization challenges.

## Dataset Overview

The experimental YOLO dataset contains:

| Split | Images |
|---|---:|
| Training | 2,796 |
| Validation | 697 |
| Test | 1,695 |
| **Total** | **5,188** |

The dataset uses a single class, `crack`, represented by YOLO segmentation polygons. Images without crack annotations are retained as background examples.

The original dataset's provenance, licensing, and complete raw-mask-to-polygon conversion procedure still require explicit documentation. The split also contains source-level differences, so the evaluation should not be interpreted as a fully source-independent benchmark.

## Experiments

Four configurations were compared using the validation set:

- **Baseline:** Reference training configuration.
- **Exp1:** Modified geometric augmentation, copy-paste, and reduced mosaic probability.
- **Exp2:** Exp1 configuration without mosaic augmentation.
- **Exp3:** Exp1-style training using additional crops derived from training images.

### Validation Results

Custom crack-only metrics at confidence threshold `0.25`:

| Experiment | Crack mean IoU | Crack mean Dice | Missed crack images | False-positive backgrounds |
|---|---:|---:|---:|---:|
| Baseline | 0.5588 | 0.6859 | 31 / 458 | 0 |
| Exp1 | 0.5677 | 0.6992 | 18 / 458 | 1 |
| Exp2 | 0.5507 | 0.6782 | 34 / 458 | 0 |
| Exp3 | 0.5691 | 0.6976 | 23 / 458 | 3 |

Exp1 was selected for further evaluation. A confidence-threshold sweep on the validation set found its highest tested crack-only mean IoU at `0.07`, with a mean IoU of `0.5908` and mean Dice of `0.7275`. This threshold was not equivalently tuned for every experiment.

## Final Test Evaluation

Exp1 was evaluated on the test split at confidence threshold `0.07`.

| Metric | Result |
|---|---:|
| Crack images | 1,474 |
| Background images | 221 |
| Crack images with at least one predicted crack pixel | 1,302 / 1,474 |
| Image-level crack detection rate | 88.33% |
| Crack-only mean IoU | 0.3719 |
| Crack-only mean Dice | 0.4907 |
| Correctly rejected background images | 212 / 221 |
| False-positive background images | 9 / 221 |

**Metric note:** Image-level detection means that at least one predicted crack pixel exists; it does not guarantee good overlap with the ground-truth mask. Crack-only IoU and Dice are calculated over images with non-empty ground-truth crack masks. These custom metrics are distinct from Ultralytics mAP.

### Error Analysis

Performance varied substantially across test image sources. Rissbilder contributed `136` of the `172` missed crack images, while GAPS384 contributed another `16`. Both sources were absent from the training and validation splits.

Performance also varied by crack size. Large cracks were generally easier to segment, but miss rates did not increase monotonically as cracks became smaller. Source, annotation style, and crack size may interact; their individual effects have not been isolated.

## Project Structure

```text
.
├── docs/
│   └── project_report.md
├── notebooks/
│   └── yolo_crack.ipynb
├── scripts/
│   ├── check_leakage.py
│   ├── prepare_yolo_dataset.py
│   └── validate_yolo_dataset.py
├── src/
│   ├── dataset_audit.py
│   ├── dataset_statistics.py
│   ├── mask_analysis.py
│   └── visualize_samples.py
├── .gitignore
└── README.md
```

The notebook documents the experimental workflow on a prepared YOLO dataset. Dataset archives and model weights are excluded from the repository.

## Limitations

- Results are based on a single training run per configuration; variability across random seeds has not been measured.
- Validation data was used for experiment comparison and confidence-threshold selection.
- Only Exp1 received the reported threshold sweep; the other configurations were compared at `0.25`.
- Final test mAP was not computed, and the other experiments were not evaluated on the test split.
- The test split contains source-level differences, and performance on unseen sources varies considerably.
- Evaluation ground truth was rasterized from YOLO polygon labels rather than independently loaded from the original raster masks.
- Dataset provenance, licensing, and some preparation details remain to be documented.

## Documentation

See [`docs/project_report.md`](docs/project_report.md) for the detailed technical report, including dataset analysis, experiment configurations, evaluation methodology, and error analysis.

## Technologies

Python · PyTorch · Ultralytics YOLO11 · OpenCV · NumPy · Pandas · Matplotlib · Google Colab

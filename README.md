# Intelligent Infrastructure Inspection System

> A computer vision pipeline for automated crack detection and segmentation in infrastructure images, with emphasis on dataset quality, annotation reliability, reproducible data preparation, model comparison, and error analysis.

## Overview

Infrastructure inspection datasets can contain more than just clean image-mask pairs. Inconsistent directory structures, empty masks, ambiguous annotations, and domain differences can significantly affect model training and evaluation.

This project investigates the complete pipeline rather than treating model training as the only objective:

**Dataset Audit → Annotation Investigation → Dataset Construction → YOLO Segmentation → Training → Pixel-Level Evaluation → Error Analysis**

The goal is to build a reproducible and evidence-driven crack segmentation workflow and understand **where the model performs well, where it fails, and why**.

---

## Key Highlights

* Audited **11,298 images and 11,298 masks** at 448 × 448 resolution.
* Identified a split-level organization issue in the original dataset without discarding valid image-mask pairs.
* Verified **0 missing image-mask pairs globally**.
* Investigated **1,454 empty masks**, including **1,411 confirmed `noncrack` cases**.
* Built a reproducible YOLO segmentation dataset using a **global mask lookup**.
* Locked the original test set before constructing the training and validation splits.
* Used a fixed **20% validation ratio** and random seed **42**.
* Compared a baseline against three targeted experiments:

  * Small-crack focused augmentation
  * No-Mosaic ablation
  * Crack-focused cropping
* Evaluated models using both standard segmentation metrics and custom pixel-level IoU/Dice measurements.
* Performed held-out test analysis and source-domain error analysis.
* Identified **Rissbilder** as the dominant source among missed test crack images.

---

## 1. Problem

Cracks in infrastructure surfaces can vary significantly in:

* Size
* Thickness
* Shape
* Contrast
* Texture
* Orientation
* Background appearance
* Image source and acquisition conditions

This makes crack segmentation a challenging computer vision problem, particularly when cracks occupy only a small portion of an image.

A strong solution therefore requires more than selecting a segmentation architecture. The dataset must first be understood, cleaned logically, converted correctly, validated, and evaluated using metrics that reflect the actual inspection problem.

---

## 2. Dataset Investigation

The original dataset contained:

* **11,298 images**
* **11,298 masks**
* JPG format
* **448 × 448** resolution

The original directory structure contained an important inconsistency: the train directory did not contain all expected local masks. Instead of assuming those images were invalid, the dataset was investigated globally.

### Audit Results

| Check                |   Result  |
| :------------------- | :-------: |
| Images               |   11,298  |
| Masks                |   11,298  |
| Images without masks |     0     |
| Masks without images |     0     |
| Unreadable files     |     0     |
| Dimension mismatches |     0     |
| Image resolution     | 448 × 448 |
| Original test images |   1,695   |

The investigation showed that the apparent missing-mask problem was primarily a **directory organization issue**, not a globally incomplete dataset.

This led to a key engineering decision:

> Preserve the original source dataset and construct the working dataset using a global image-to-mask lookup rather than deleting images based only on their original directory location.

---

## 3. Annotation Investigation

### Empty Masks

A total of:

**1,454 empty masks**

were identified.

Among them:

* **1,411** were associated with filenames beginning with `noncrack_`
* This represents **97.04%** of all empty masks.
* The remaining **43** empty masks were treated as ambiguous rather than automatically classified as background.

This distinction was important because an empty mask can represent either:

1. A genuine background/non-crack image
2. An annotation issue
3. An ambiguous sample requiring further investigation

The pipeline therefore used conservative rules when selecting confirmed background images.

### Mask Threshold

Binary crack masks were converted using a threshold of:

**127**

Pixels above the threshold were treated as foreground crack pixels.

---

## 4. Dataset Construction

The YOLO dataset was constructed programmatically rather than manually reorganizing files.

### Construction Strategy

The preparation pipeline:

1. Builds a global image-to-mask lookup.
2. Locks the original test set first.
3. Identifies positive crack images.
4. Identifies confirmed `noncrack` background images.
5. Splits the available training candidates into train and validation sets.
6. Uses a fixed random seed of **42**.
7. Converts binary masks into YOLO segmentation polygons.
8. Generates the YOLO `data.yaml`.
9. Validates the resulting annotations.
10. Checks for train/validation/test leakage.

### Final Working Split

| Split          | Positive Images | Background Images | Total Images |
| :------------- | --------------: | ----------------: | -----------: |
| **Train**      |           1,836 |               960 |        2,796 |
| **Validation** |             458 |               239 |          697 |
| **Test**       |           1,474 |               221 |        1,695 |
| **Total**      |       **3,768** |         **1,420** |    **5,188** |

The training and validation background proportions were highly consistent:

* Train: **34.33%**
* Validation: **34.29%**
* Difference: **0.04 percentage points**

This provides a closely matched background composition between the training and validation sets.

---

## 5. YOLO Segmentation Pipeline

The project uses YOLO segmentation annotations where each object is represented by a class ID followed by normalized polygon coordinates.

The dataset preparation pipeline includes:

* Binary mask thresholding
* Connected foreground extraction
* Contour extraction
* Polygon generation
* Coordinate normalization
* YOLO label generation
* Empty-label support for background images
* Dataset YAML generation
* Annotation validation

### Validation Checks

The validation script checks for:

* Missing labels
* Unexpected labels
* Malformed annotation lines
* Invalid class IDs
* Coordinates outside `[0, 1]`
* Invalid annotation structure
* Empty labels
* Split integrity

Empty labels are intentionally allowed for background-only images.

A separate leakage check verifies that image stems do not overlap between:

* Train
* Validation
* Test

---

## 6. Model and Training Strategy

The project uses a pretrained **YOLO11s-seg** architecture.

Training was performed in **Google Colab using a Tesla T4 GPU**, while the local machine was used for dataset investigation, preparation, validation, and engineering tasks.

### Training Configuration

| Parameter          | Value       |
| :----------------- | :---------- |
| Architecture       | YOLO11s-seg |
| Image Size         | 448 × 448   |
| Batch Size         | 16          |
| Epochs             | 50          |
| Patience           | 15          |
| Optimizer          | AdamW       |
| Learning Rate      | 0.002       |
| Momentum           | 0.9         |
| Workers            | 2           |
| Random Seed        | 42          |
| Pretrained Weights | Yes         |

---

## 7. Experiment Design

Four training configurations were investigated.

### Baseline

The baseline provides a reference point for evaluating subsequent changes.

### Experiment 1 — Small-Crack Focused Augmentation

Additional augmentation was investigated with the goal of improving robustness to small and visually subtle cracks.

### Experiment 2 — No-Mosaic Ablation

Mosaic augmentation was removed to investigate its contribution to crack segmentation performance.

### Experiment 3 — Crack-Focused Cropping

Crack-focused cropping was introduced to provide the model with more localized views of crack regions and increase the representation of small crack structures.

The experiments were designed as controlled engineering comparisons rather than simply searching for the highest score.

---

## 8. Validation Results

All experiments were evaluated on the same **697-image validation set**, containing:

* 458 crack-containing images
* 239 background images

| Model / Experiment       | Crack mIoU | Crack Dice | Miss Rate | False Positives | Mask mAP50 | Mask mAP50-95 |
| :----------------------- | :--------: | :--------: | :-------: | :-------------: | :--------: | :-----------: |
| **Baseline**             |   0.5588   |   0.6859   |   6.77%   |        0        |    0.319   |     0.0969    |
| **Exp 1 — Augmentation** |   0.5677   |   0.6992   |   3.93%   |        1        |    0.327   |     0.0986    |
| **Exp 2 — No-Mosaic**    |   0.5507   |   0.6782   |   7.42%   |        0        |    0.310   |     0.0937    |
| **Exp 3 — Cropping**     | **0.5691** |   0.6976   |   5.02%   |        3        |    0.323   |   **0.1010**  |

### What the Results Suggest

#### Augmentation

Experiment 1 improved over the baseline in:

* Crack mIoU
* Crack Dice
* Miss rate
* Mask mAP50
* Mask mAP50-95

The number of missed crack images decreased from:

**31 → 18**

on the 458 crack-containing validation images.

#### No-Mosaic Ablation

Removing Mosaic produced a noticeable performance decrease:

* Crack mIoU: **0.5677 → 0.5507**
* Missed cracks: **18 → 34**

For the investigated very-small-crack subset, **8/8** cracks were missed in the No-Mosaic experiment.

This suggests that Mosaic augmentation was useful for the evaluated dataset, particularly for difficult small spatial patterns. The result is treated as experimental evidence rather than proof of a universal causal relationship.

#### Crack-Focused Cropping

Experiment 3 achieved:

* Highest Crack mIoU: **0.5691**
* Highest Mask mAP50-95: **0.1010**

For the corresponding very-small-crack subset, IoU improved from:

**0.0000 → 0.1759**

and **4 of 8** samples were recovered.

However, the number of false positives on background images increased:

**0 → 3**

This highlights an important trade-off between sensitivity to small cracks and background rejection.

---

## 9. Evaluation Methodology

Standard segmentation metrics were complemented with custom pixel-level evaluation.

### Pixel-Level IoU

Intersection over Union was calculated between the ground-truth binary mask and the combined predicted mask.

### Dice Score

Dice was used to measure overlap while providing a complementary perspective to IoU.

### Edge Cases

The evaluation explicitly handles:

* Ground truth empty + prediction empty → IoU = 1, Dice = 1
* Ground truth non-empty + prediction empty → IoU = 0, Dice = 0
* Ground truth and prediction both non-empty → standard overlap calculation

Predicted segmentation instances are combined into a unified binary mask before pixel-level evaluation.

The masks are rasterized at the original **448 × 448** resolution.

---

## 10. Final Held-Out Test Evaluation

The held-out test set contains:

**1,695 images**

The final evaluation identified:

* **1,474 crack-containing images**
* **221 background-only images**

The operating confidence threshold used for the final evaluation was:

**0.07**

### Test Results

| Test Outcome                | Number | Percentage |
| :-------------------------- | :----: | :--------: |
| **Total Test Images**       |  1,695 |   100.00%  |
| **Crack-Containing Images** |  1,474 |   86.96%   |
| ├─ Detected Cracks          |  1,302 |   88.33%   |
| └─ Missed Cracks            |   172  |   11.67%   |
| **Background-Only Images**  |   221  |   13.04%   |
| ├─ Correctly Rejected       |   212  |   95.93%   |
| └─ False Positives          |    9   |    4.07%   |

### Final Test Interpretation

On crack-containing images:

* **1,302 / 1,474** were detected.
* Image-level recall was **88.33%**.
* **172 / 1,474** crack images were missed.

On background-only images:

* **212 / 221** were correctly rejected.
* Background rejection was **95.93%**.
* **9 / 221** produced false positives.

The final reported crack segmentation metrics include:

* **Crack mIoU: 0.5908**
* **Crack Dice: 0.7275**

These values summarize pixel-level segmentation quality and complement the image-level detection analysis above.

---

## 11. Error Analysis

The missed test crack images were further analyzed by source domain.

| Source Domain     | Missed Images | Percentage of Misses |
| :---------------- | :-----------: | :------------------: |
| **Rissbilder**    |      136      |        79.07%        |
| **GAPS384**       |       16      |         9.30%        |
| **Other Sources** |       20      |        11.63%        |
| **Total**         |    **172**    |      **100.00%**     |

### Main Finding

The majority of missed cracks originated from **Rissbilder**.

This indicates a potential **source-domain generalization gap**.

The result is important because it changes the interpretation of model failure.

Instead of simply concluding:

> "The model is not accurate enough."

the analysis suggests asking:

> "What visual characteristics of this source domain are different from the data the model learned from?"

Potential factors include:

* Crack appearance
* Image texture
* Contrast
* Surface material
* Crack thickness
* Acquisition conditions
* Annotation characteristics

This provides a more useful direction for future dataset expansion and targeted training.

---

## 12. Engineering Decisions

### Preserve the Source Dataset

The original dataset was not destructively modified.

Instead, preparation scripts generate a separate YOLO-ready dataset.

### Global Image-Mask Lookup

Because the original split organization was inconsistent, image-mask pairing was performed globally rather than assuming that corresponding files must exist in the same original directory.

### Lock the Test Set First

The original test set was preserved before constructing the new train/validation split.

This reduces the risk of accidentally introducing test information into training.

### Conservative Background Selection

Only clearly identified `noncrack` samples were used as confirmed background images.

Ambiguous empty masks were not automatically treated as confirmed negatives.

### Reproducibility

The preparation and validation process uses deterministic settings where appropriate, including:

* Random seed 42
* Explicit validation ratio
* Programmatic dataset construction
* Automated annotation validation
* Leakage checks

---

## 13. Repository Structure

```text
intelligent-infrastructure-inspection/
│
├── src/
│   ├── dataset_audit.py
│   ├── dataset_statistics.py
│   ├── mask_analysis.py
│   └── visualize_samples.py
│
├── scripts/
│   ├── prepare_yolo_dataset.py
│   ├── validate_yolo_dataset.py
│   └── check_leakage.py
│
├── docs/
│   └── project_report.md
│
├── .gitignore
└── README.md
```

### Main Files

| File                               | Purpose                                     |
| :--------------------------------- | :------------------------------------------ |
| `src/dataset_audit.py`             | Dataset integrity and structure audit       |
| `src/mask_analysis.py`             | Empty-mask and pixel-value analysis         |
| `src/dataset_statistics.py`        | Dataset statistics and crack-area analysis  |
| `src/visualize_samples.py`         | Image, mask, and overlay visualization      |
| `scripts/prepare_yolo_dataset.py`  | Reproducible YOLO dataset construction      |
| `scripts/validate_yolo_dataset.py` | YOLO annotation validation                  |
| `scripts/check_leakage.py`         | Train/validation/test leakage check         |
| `docs/project_report.md`           | Full technical investigation and case study |

---

## 14. Reproducibility

The repository focuses on reproducible dataset engineering rather than storing generated datasets and model weights directly in Git.

The `.gitignore` excludes:

* Virtual environments
* Python cache files
* Model weights
* Local YOLO dataset archives
* Local experiment outputs

The main dataset preparation process can be reproduced through:

```bash
python scripts/prepare_yolo_dataset.py
```

Dataset annotations can then be validated with:

```bash
python scripts/validate_yolo_dataset.py
```

Potential train/validation/test overlap can be checked with:

```bash
python scripts/check_leakage.py
```

---

## 15. Limitations

Several limitations remain.

### Small-Crack Sensitivity

Very small and thin cracks remain difficult to segment reliably.

### Domain Generalization

The test error analysis shows that missed detections are strongly concentrated in the Rissbilder source domain.

### Dataset Composition

The working dataset is derived from a specific collection of crack datasets and therefore may not represent every real-world infrastructure surface.

### Computational Constraints

Training was performed using cloud GPU resources rather than dedicated local GPU hardware.

### Threshold Dependence

Detection outcomes depend on the confidence threshold. The selected operating threshold should therefore be considered an engineering operating point rather than a universal optimum.

---

## 16. Future Work

Potential next steps include:

### Dataset Expansion

Add more examples from underrepresented source domains, particularly domains contributing a large proportion of missed detections.

### Small-Crack Focus

Investigate:

* Higher-resolution training
* Multi-scale training
* More targeted cropping
* Small-object augmentation
* Alternative segmentation architectures

### Domain Generalization

Evaluate domain-aware training strategies and cross-source validation to determine whether the model generalizes beyond the dominant training domains.

### Threshold Calibration

Perform a systematic confidence-threshold analysis to select an operating point based on the intended inspection scenario.

### Deployment Optimization

Investigate:

* Inference latency
* Model size
* Memory usage
* Edge deployment
* Batch vs. single-image inference

### Inspection System Integration

A future production system could combine crack segmentation with:

* Crack severity estimation
* Crack length and width measurement
* Defect tracking across inspections
* Inspection history
* Automated reporting
* Infrastructure asset monitoring

---

## 17. Technical Report

The complete investigation, including dataset auditing, annotation analysis, dataset construction, experiment details, evaluation methodology, error analysis, and engineering decisions is available in:

**[Read the Full Technical Project Report](docs/project_report.md)**

The report serves as the detailed technical case study behind this repository, while this README provides the high-level engineering overview.

---

## 18. Final Takeaway

This project was not treated as a simple:

**"Train YOLO and report accuracy"**

exercise.

The main focus was building a complete computer vision workflow in which every stage could be investigated and justified:

**Audit → Understand → Prepare → Validate → Train → Measure → Analyze → Improve**

The most important outcome is therefore not a single metric.

It is the ability to trace model performance back to:

* Dataset structure
* Annotation quality
* Background composition
* Experiment design
* Evaluation methodology
* Small-crack behavior
* Source-domain differences

This makes the project a practical case study in **computer vision engineering, dataset quality analysis, segmentation, reproducible experimentation, and model error analysis**.

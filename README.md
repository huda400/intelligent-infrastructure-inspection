# Intelligent Infrastructure Inspection System

A computer vision pipeline for automated crack detection and segmentation in infrastructure images, with emphasis on **dataset quality, annotation reliability, reproducible dataset preparation, model comparison, pixel-level evaluation, and error analysis**.

The project goes beyond simply training a segmentation model. It investigates the original dataset, identifies annotation issues, constructs a controlled YOLO segmentation dataset, compares multiple training configurations, selects an operating confidence threshold using validation data, and evaluates generalization across multiple crack-image sources.

---

## Project Workflow

```text
Dataset Audit
     ↓
Annotation Investigation
     ↓
Dataset Construction
     ↓
YOLO Segmentation Conversion
     ↓
Dataset Validation
     ↓
Model Training
     ↓
Validation-Based Model Comparison
     ↓
Confidence Threshold Selection
     ↓
Held-Out Test Evaluation
     ↓
Source-Level Error Analysis
```

---

## Key Highlights

* Audited an original dataset containing **11,298 images and 11,298 masks**.
* Verified global image-mask pairing with **0 images without masks and 0 masks without images**.
* Investigated **1,454 empty masks**:

  * 1,411 correspond to filenames containing `noncrack`
  * 43 remain ambiguous and require further inspection.
* Used a global filename lookup to recover valid masks despite inconsistencies in the original train directory.
* Constructed a controlled working dataset of **5,188 images**:

  * 2,796 training images
  * 697 validation images
  * 1,695 test images
* Used a deterministic **20% validation split with seed 42** within each training class.
* Converted binary masks to YOLO segmentation polygons using a fixed threshold of **>127**.
* Trained and compared four configurations:

  * Baseline
  * Exp1
  * Exp2
  * Exp3
* Used **YOLO11s-seg** with pretrained weights.
* Selected the final operating confidence threshold using **validation data only**.
* Final validation threshold: **0.07**.
* Final held-out test result at confidence 0.07:

  * Mean per-image crack IoU: **0.3719**
  * Mean per-image crack Dice: **0.4907**
  * Crack-image detection rate: **88.33%**
* Rissbilder represents only **38.5% of crack images in the test set (567 / 1,474)**, but accounts for **79.07% of all missed crack images**.

---

# 1. Dataset Audit

## Original Dataset

The original dataset contains:

* **11,298 images**
* **11,298 masks**
* Image format: JPG
* Resolution: **448 × 448**

The initial audit revealed a major inconsistency in the local directory structure:

```text
train/images → 2,294 images
train/masks  → only 9 masks
```

A local directory-only comparison would therefore incorrectly suggest that most training images were missing annotations.

Instead, a **global filename-based lookup** was performed across the available mask files.

The global pairing check found:

```text
Images without masks: 0
Masks without images: 0
```

This showed that the apparent mismatch was primarily a directory-organization issue rather than a global absence of annotations.

### Important qualification

The audit's "unreadable image" check was based on opening image headers and validating dimensions. It does **not** constitute a full pixel-level decompression/integrity test for every file.

---

# 2. Empty Mask Investigation

A total of:

```text
1,454 empty masks
```

were identified.

Among them:

```text
1,411 filenames contain "noncrack"
43 empty masks remain ambiguous
```

Therefore:

```text
1,411 / 1,454 = 97.04%
```

of the empty masks were strongly associated with intentional background images.

The remaining 43 empty masks were not automatically assumed to be background. They were treated as ambiguous rather than silently relabeled.

This distinction was important because an empty mask can represent either:

* a true background image, or
* an annotation that failed to capture an existing crack.

---

# 3. Working Dataset Construction

After auditing the original dataset, a controlled working dataset was constructed.

| Split      | Crack Images | Background Images | Total     |
| ---------- | ------------ | ----------------- | --------- |
| Train      | 1,836        | 960               | 2,796     |
| Validation | 458          | 239               | 697       |
| Test       | 1,474        | 221               | 1,695     |
| **Total**  | **3,768**    | **1,420**         | **5,188** |

The split was performed class-wise, with approximately **20% of the available training examples reserved for validation** using a deterministic random seed of **42**.

The test set was preserved separately and was not used for threshold selection.

### Dataset utilization

The original dataset contained 11,298 images, while the current working dataset contains 5,188 images.

Therefore:

```text
11,298 - 5,188 = 6,110
```

original images are not currently included in the working dataset.

This is an important limitation and should be investigated before claiming that the final model represents the full original dataset.

---

# 4. Dataset Source Distribution

The training and validation sets are dominated by two sources:

| Source              | Train | Validation |
| ------------------- | ----- | ---------- |
| CRACK500            | 1,758 | 436        |
| CFD                 | 78    | 22         |
| Other crack sources | 0     | 0          |

The test set contains substantially more source diversity:

| Test Source    | Crack Images |
| -------------- | ------------ |
| CRACK500       | 505          |
| CFD            | 18           |
| Rissbilder     | 567          |
| GAPS384        | 76           |
| Volker         | 148          |
| DeepCrack      | 78           |
| CrackTree200   | 31           |
| Sylvie Chambon | 25           |
| Eugen Muller   | 8            |
| forest         | 18           |
| **Total**      | **1,474**    |

This means the validation set is primarily **dominated by the training-source distribution**, while the test set introduces substantial **cross-dataset / unseen-source variation**.

In particular, Rissbilder, GAPS384, Volker, DeepCrack, CrackTree200, Sylvie Chambon, Eugen Muller, and forest are not represented in the training or validation positives.

---

# 5. Train/Validation/Test Leakage Check

A filename-stem overlap check was performed between the dataset splits.

No direct filename overlap was found across the evaluated splits.

However, this check is **filename-based only**.

It does not detect:

* duplicate image content,
* near-duplicate images,
* sibling tiles originating from the same parent image,
* visually similar images with different filenames.

This is particularly relevant for datasets such as CRACK500, where filenames can encode relationships between image tiles.

A stronger future leakage analysis should include:

* image perceptual hashes,
* duplicate detection,
* parent-image grouping where available,
* source-aware splitting.

The possibility of related CRACK500 tiles across splits also means that validation performance may be somewhat optimistic.

---

# 6. Mask-to-YOLO Conversion

Binary masks were converted into YOLO segmentation polygons.

The foreground threshold was:

```text
pixel > 127
```

The conversion process:

1. Load binary mask.
2. Convert pixels above the threshold to foreground.
3. Extract external contours.
4. Convert contour coordinates to normalized YOLO coordinates.
5. Write class `0` segmentation polygons.

The resulting labels use:

```text
class_id x1 y1 x2 y2 ... xn yn
```

with normalized coordinates in the range:

```text
0–1
```

Coordinates were written with six decimal places.

### Conversion considerations

Using external contours is simple and reproducible, but it can introduce limitations for:

* very thin cracks,
* JPEG artifacts,
* small isolated regions,
* holes or nested structures,
* fragmented annotations.

No minimum contour-area filter was applied.

---

# 7. Dataset Validation

The generated YOLO dataset was validated for:

* images without labels,
* labels without matching images,
* malformed label files,
* invalid class IDs,
* coordinates outside the expected range.

The validation process helped ensure that the generated dataset was structurally compatible with YOLO segmentation training.

However, the current validation does not fully detect all geometric edge cases, such as:

* zero-area polygons,
* collinear polygons,
* certain invalid numerical values such as NaN.

A stronger geometry validation stage is planned for future iterations.

---

# 8. Model Architecture

The project uses:

**YOLO11s-seg**

with pretrained weights.

Main training configuration:

| Parameter     | Value                 |
| ------------- | --------------------- |
| Model         | YOLO11s-seg           |
| Input size    | 448 × 448             |
| Batch size    | 16                    |
| Epochs        | 50                    |
| Patience      | 15                    |
| Optimizer     | AdamW (auto-selected) |
| Learning rate | 0.002                 |
| Momentum      | 0.9                   |
| Workers       | 2                     |
| Pretrained    | Yes                   |

All four experiments completed the full **50 epochs**.

The validation mAP was still increasing near the end of training, so the experiments should not be interpreted as fully converged.

---

# 9. Experiment Design

Four configurations were evaluated.

## Baseline

The baseline used the standard training configuration with:

```text
seed = 0
```

---

## Exp1

Exp1 introduced a combined augmentation configuration:

| Parameter  | Baseline | Exp1 |
| ---------- | -------- | ---- |
| Mosaic     | 1.0      | 0.5  |
| Scale      | 0.5      | 0.3  |
| Degrees    | 0        | 5    |
| FlipUD     | 0        | 0.1  |
| Copy-paste | 0        | 0.15 |
| Seed       | 0        | 42   |

Exp1 should therefore be interpreted as a **combined augmentation configuration**, not as the isolated effect of one augmentation.

---

## Exp2

Exp2 was derived directly from Exp1 by removing Mosaic:

```text
Exp1 Mosaic: 0.5
Exp2 Mosaic: 0
```

All other Exp1 settings were retained.

Therefore, Exp2 provides an ablation of Mosaic relative to Exp1.

It is **not** a direct baseline-to-no-Mosaic comparison.

---

## Exp3

Exp3 extended Exp1 with crack-focused training crops.

The training dataset increased from:

```text
2,796 → 4,593 images
```

The approximate background proportion changed from:

```text
34.3% → 20.9%
```

The additional crops were:

```text
224 × 224
```

and were **upscaled 2× to 448 × 448 when the dataset was generated**.

The generated crops were then **added to the original 2,796 training images**, rather than replacing them.

The additional crops were generated from the training set only.

Validation and test sets remained unchanged.

---

# 10. Validation Results

For experiment comparison, confidence threshold **0.25** was used consistently.

| Experiment | Crack IoU | Crack Dice | Missed | Miss Rate | FP | mAP50 | mAP50-95 |
| ---------- | --------- | ---------- | ------ | --------- | -- | ----- | -------- |
| Baseline   | 0.5588    | 0.6859     | 31     | 6.77%     | 0  | 0.319 | 0.0969   |
| Exp1       | 0.5677    | 0.6992     | 18     | 3.93%     | 1  | 0.327 | 0.0986   |
| Exp2       | 0.5507    | 0.6782     | 34     | 7.42%     | 0  | 0.310 | 0.0937   |
| Exp3       | 0.5691    | 0.6976     | 23     | 5.02%     | 3  | 0.323 | 0.1010   |

### Interpretation

At the common comparison threshold of **confidence 0.25**, Exp1 was the validation configuration with the **lowest miss rate (3.93%)**, and it was selected for the threshold sweep and final held-out test evaluation.

This selection should be understood as a **practical project decision**, rather than as a formally pre-registered selection rule in the notebook.

Exp3 produced a slightly higher crack IoU than Exp1:

```text
0.5691 vs 0.5677
```

but also produced:

```text
Misses: 18 → 23
False positives: 1 → 3
mAP50: 0.327 → 0.323
```

Therefore, the small IoU improvement of **0.0014** did not translate into better detection behavior.

Because all experiments used a single seed and were not fully converged, these small differences should not be treated as statistically conclusive.

---

# 11. Validation-Based Confidence Threshold Selection

The final confidence threshold was selected by evaluating **Exp1 on the validation set**.

The threshold sweep was performed before evaluating the held-out test set.

| Confidence | Crack IoU  | Missed / 458 | FP / 239 |
| ---------- | ---------- | ------------ | -------- |
| 0.05       | 0.5901     | 2            | 8        |
| **0.07**   | **0.5908** | **4**        | **7**    |
| 0.10       | 0.5893     | 5            | 5        |
| 0.15       | 0.5850     | 7            | 3        |
| 0.20       | 0.5774     | 10           | 1        |
| 0.25       | 0.5677     | 18           | 1        |
| 0.30       | 0.5544     | 30           | 0        |
| 0.40       | 0.5006     | 77           | 0        |

The selected operating threshold was:

```text
confidence = 0.07
```

because **0.07 produced the highest mean per-image crack IoU in the validation sweep: 0.5908**.

At this threshold:

* Crack mean per-image IoU = **0.5908**
* Crack Dice = **0.7275**
* Misses = **4 / 458**
* False positives = **7 / 239**

The threshold of 0.05 produced fewer misses (**2 vs. 4**) but also produced more false positives (**8 vs. 7**) and a slightly lower crack IoU (**0.5901 vs. 0.5908**).

The mean per-image crack IoU is also nearly flat between confidence values **0.05 and 0.10**:

```text
0.05 → 0.5901
0.07 → 0.5908
0.10 → 0.5893
```

The differences are only **0.0007** and **0.0015**, respectively. Therefore, 0.07 should be viewed as an operating point within a relatively flat region rather than as a sharp optimum.

The test set was not used for threshold selection.

---

# 12. Final Test Evaluation

The final configuration was:

```text
Experiment: Exp1
Confidence threshold: 0.07
```

The notebook contains a **single test evaluation run** at this selected threshold.

## Crack-only pixel-level metrics

For the 1,474 crack images:

| Metric                    | Result     |
| ------------------------- | ---------- |
| Mean per-image crack IoU  | **0.3719** |
| Mean per-image crack Dice | **0.4907** |
| Median IoU                | **0.3388** |
| Median Dice               | **0.5061** |

These are calculated per image using the predicted and ground-truth crack masks.

A missed crack image contributes:

```text
IoU = 0
Dice = 0
```

for the crack-only averages.

---

# 13. Overall Test Metrics

When both crack and background images are included, and an empty prediction against an empty ground-truth mask is treated as a perfect match:

| Metric            | Result     |
| ----------------- | ---------- |
| Overall mean IoU  | **0.4485** |
| Overall mean Dice | **0.5518** |

Because background images can contribute perfect empty-empty scores, these overall values should **not** be confused with crack-only segmentation performance.

For this reason, the crack-only metrics are the primary metrics used to describe segmentation quality.

---

# 14. Image-Level Test Results

The test set contains:

```text
1,474 crack images
221 background images
```

For crack images:

```text
Detected: 1,302
Missed: 172
```

Therefore:

```text
Detection rate = 1,302 / 1,474 = 88.33%
```

For background images:

```text
Correctly rejected: 212
False positives: 9
```

Therefore:

```text
Background rejection = 212 / 221 = 95.93%
```

### Test background composition

The 221 background images consist of:

```text
212 known noncrack_ images
9 empty-label images from crack sources:
6 Rissbilder
3 Sylvie Chambon
```

The counts of 6 and 3 are **derived from total versus crack-image counts per source**.

These 9 images may be part of the **43 ambiguous empty masks** identified during the earlier annotation investigation.

The exact false-positive distribution across these background groups was not established, so no stronger claim is made.

### Important definition

In this analysis, a crack image is considered **detected** if the model produces any non-empty predicted crack mask.

Therefore, detection rate does **not** imply good segmentation quality.

A model can detect an image while producing a poorly aligned mask with very low IoU.

---

# 15. Source-Domain Error Analysis

The test set contains crack images from multiple sources.

| Source         | Crack Images | Missed  | Miss Rate  | Mean IoU   |
| -------------- | ------------ | ------- | ---------- | ---------- |
| Rissbilder     | 567          | 136     | 23.99%     | 0.2038     |
| CRACK500       | 505          | 4       | 0.79%      | 0.6054     |
| Volker         | 148          | 9       | 6.08%      | 0.2494     |
| DeepCrack      | 78           | 2       | 2.56%      | 0.6000     |
| GAPS384        | 76           | 16      | 21.05%     | 0.2641     |
| CrackTree200   | 31           | 1       | 3.23%      | 0.0548     |
| Sylvie Chambon | 25           | 4       | 16.00%     | 0.1002     |
| CFD            | 18           | 0       | 0.00%      | 0.4639     |
| forest         | 18           | 0       | 0.00%      | 0.5081     |
| Eugen Muller   | 8            | 0       | 0.00%      | 0.1738     |
| **Total**      | **1,474**    | **172** | **11.67%** | **0.3719** |

### Main finding

Rissbilder is the dominant source of failures:

```text
136 / 172 = 79.07%
```

of all missed crack images.

Rissbilder represents only **38.5% of crack images in the test set (567 / 1,474)**, but accounts for **79.07% of the misses**.

Rissbilder is **not represented in the training set**, but the absence of a source from training does not by itself explain the failure.

For comparison, **DeepCrack is also unseen during training**, yet it achieved:

```text
Miss rate = 2.56%
Mean IoU = 0.6000
```

This suggests that source-specific characteristics beyond simply being "unseen" contribute to generalization difficulty.

---

# 16. Why Detection Rate Alone Is Not Enough

The source-level results demonstrate why image-level detection should be interpreted together with segmentation quality.

For example:

```text
CrackTree200
Detection miss rate: 3.23%
Mean IoU: 0.0548
```

The model detects almost all CrackTree200 images, but the predicted segmentation masks overlap the ground truth very poorly.

Therefore:

> A detected crack image does not necessarily mean that the crack was segmented accurately.

This is why the project reports both:

* image-level detection behavior, and
* pixel-level IoU/Dice.

---

# 17. Error Analysis

The test errors are not uniformly distributed.

## 17.1 Source/domain shift

The strongest evidence is the high miss rate on Rissbilder and GAPS384 compared with CRACK500.

This suggests differences in:

* image appearance,
* crack morphology,
* acquisition conditions,
* annotation style,
* background texture,
* source-specific visual characteristics.

---

## 17.2 Small or difficult cracks

Many missed cracks are visually difficult.

However, crack size alone should **not** be presented as the sole explanation for the failures.

The current analysis found that:

```text
139 / 172
```

missed cracks fall into the medium-size range of approximately **1–5% of image area**.

Only a small number of misses fall into the very-small category.

Therefore, the evidence suggests that the errors cannot simply be described as "the model misses tiny cracks."

Crack size is also confounded with dataset source, so a stronger analysis would examine size distributions separately for each source.

---

## 17.3 Fragmented annotations

The test labels are considerably more fragmented than the validation labels.

Approximate polygon/object counts:

```text
Validation:
1,094 objects / 458 crack images ≈ 2.4 objects/image

Test:
32,088 objects / 1,474 crack images ≈ 21.8 objects/image
```

This difference can make the test segmentation problem substantially harder.

It may also reflect differences in annotation style between datasets.

---

## 17.4 Mask-to-polygon conversion

The conversion from raster masks to polygons can introduce additional error, especially for:

* thin cracks,
* fragmented cracks,
* small disconnected regions,
* JPEG artifacts.

Because the conversion uses external contours without a minimum-area filter, small components are retained.

---

# 18. Key Findings

The project produced several important findings.

### Finding 1 — Dataset organization was a major issue

The local train directory appeared to contain only 9 masks for 2,294 images.

A global filename lookup showed:

```text
0 images without masks
0 masks without images
```

This demonstrates why dataset auditing should happen before model training.

---

### Finding 2 — Most empty masks were intentional background

```text
1,454 empty masks
1,411 noncrack
97.04%
```

Only 43 remained ambiguous.

---

### Finding 3 — Validation performance may overestimate generalization

Exp1 achieved:

```text
Validation crack IoU @ 0.07 = 0.5908
```

while the held-out test achieved:

```text
Test crack IoU @ 0.07 = 0.3719
```

The difference is consistent with the much greater source diversity and domain shift in the test set.

Validation may also be somewhat optimistic because the current leakage check is filename-based and may not detect related tiles originating from the same parent image, particularly in CRACK500.

---

### Finding 4 — Rissbilder dominates the missed detections

```text
136 / 172 missed crack images
= 79.07%
```

come from Rissbilder.

Rissbilder is not represented in the training data, but unseen-source status alone does not explain the result: DeepCrack is also unseen during training and achieved a **2.56% miss rate and 0.6000 mean IoU**.

This indicates that source-specific visual and annotation characteristics likely play an important role.

---

### Finding 5 — More training crops did not clearly improve robustness

Exp3 increased the training set:

```text
2,796 → 4,593
```

but compared with Exp1:

```text
IoU:      0.5677 → 0.5691
Misses:   18 → 23
FP:        1 → 3
mAP50:    0.327 → 0.323
```

The IoU improvement was only **0.0014**, while image-level detection became worse.

The background share also changed from **34.3% to 20.9%**, which may explain part of the increase in false positives. Therefore, the result cannot be attributed to the added crops alone.

---

# 19. Limitations

This project has several important limitations.

### Dataset limitations

* 6,110 original images are not currently included in the working dataset.
* Training and validation positives are dominated by CRACK500 and CFD.
* Several sources appear only in the test set.
* Test annotations are considerably more fragmented.
* Some empty masks remain ambiguous.

### Experimental limitations

* Each configuration was trained with a single seed.
* Baseline uses seed 0, while Exp1–Exp3 use seed 42.
* Exp1 changes several augmentation parameters simultaneously.
* Experiments completed 50 epochs but were still improving near the end.
* Therefore, small metric differences should not be interpreted as statistically significant.
* Experiments were compared at **confidence 0.25**, while the threshold sweep was performed only for Exp1. Therefore, Exp1 was not compared against the other experiments at each experiment's individually optimized threshold.

### Leakage-analysis limitations

The current leakage check is filename-based.

It does not detect:

* duplicate pixels,
* perceptual duplicates,
* sibling tiles,
* shared parent images.

### Evaluation limitations

* The confidence threshold is data-dependent.
* The threshold sweep was performed on the validation set only.
* The test set was evaluated at one selected operating threshold.
* mAP was not computed for the final test run.
* Detection is defined by any non-empty prediction and therefore does not measure segmentation quality by itself.
* Source-level differences may reflect both domain shift and annotation-style differences.

### Annotation-processing consistency

Different helper scripts currently use slightly different foreground definitions:

* `>127` in `prepare_yolo_dataset.py`
* `≥127` in `visualize_samples.py`
* `≥128` in `dataset_statistics.py`
* `max() == 0` for empty-mask detection in `mask_analysis.py`

The practical effect appears small, but standardizing these definitions would improve consistency and reproducibility.

### Engineering/reproducibility limitations

The current repository does not yet include all of the following:

* the complete training notebook,
* a pinned `requirements.txt`,
* trained model weights,
* experiment result CSVs,
* formal dataset source/license documentation,
* a single CLI/configuration file for reproducing every experiment.

The training environment used for the experiments was:

```text
Python 3.13.15
Ultralytics 8.4.155
Google Colab
NVIDIA T4
```

Training paths also contain environment-specific configuration.

---

# 20. Future Work

Future improvements should focus on both model performance and experimental rigor.

### Dataset

* Recover and investigate the remaining 6,110 original images.
* Inspect the 43 ambiguous empty masks.
* Perform source-aware dataset analysis.
* Review annotation quality across sources.
* Investigate test-label fragmentation.

### Leakage and splitting

* Add perceptual duplicate detection.
* Group related tiles by parent image.
* Use source-aware or group-aware splitting where appropriate.

### Modeling

* Evaluate larger YOLO segmentation models.
* Test stronger augmentation strategies.
* Explore crack-specific preprocessing.
* Evaluate alternative segmentation architectures.
* Investigate source-aware training.

### Evaluation

* Perform threshold calibration.
* Report per-source performance systematically.
* Analyze IoU by crack size.
* Analyze performance by image appearance and source.
* Add precision/recall curves.
* Evaluate the final configuration on additional external datasets.

### Reproducibility

* Add `requirements.txt`.
* Add a clean training configuration.
* Export experiment results to CSV.
* Save model weights and metadata.
* Document dataset sources and licenses.
* Include the final training/evaluation notebook.

---

# 21. Project Structure

```text
intelligent-infrastructure-inspection/
│
├── src/
│   ├── dataset_audit.py
│   ├── dataset_statistics.py
│   ├── mask_analysis.py
│   ├── visualize_samples.py
│   └── ...
│
├── scripts/
│   ├── prepare_yolo_dataset.py
│   ├── validate_yolo_dataset.py
│   ├── check_leakage.py
│   └── ...
│
├── docs/
│   └── project_report.md
│
├── README.md
└── ...
```

If the training/evaluation notebook is added to the repository, it should also be listed here using its actual repository path and filename.

---

# 22. Technical Takeaway

The main lesson from this project was that **model training was only one part of the problem**.

The most important work involved:

```text
Audit the data
      ↓
Understand the annotations
      ↓
Build a controlled dataset
      ↓
Validate the conversion
      ↓
Compare experiments
      ↓
Select threshold on validation
      ↓
Evaluate the final configuration on the held-out test set
      ↓
Analyze failures by source
```

The final configuration achieved:

```text
Validation crack IoU @ conf 0.07: 0.5908
Test crack IoU @ conf 0.07:       0.3719
Test crack Dice:                  0.4907
Test crack detection rate:        88.33%
Background rejection:             95.93%
```

The largest generalization challenge was not simply detecting cracks, but maintaining reliable segmentation across **different datasets, annotation styles, and visual domains**.

That finding is the main engineering takeaway of the project.

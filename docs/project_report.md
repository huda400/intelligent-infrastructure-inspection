# Technical Project Report: Intelligent Infrastructure Inspection System

## 1. Dataset Investigation & Audit

### 1.1 Project Objective

The objective of this project is to develop an automated infrastructure inspection system capable of identifying and segmenting surface cracks from images.

The project focuses on **semantic/instance-style crack segmentation using YOLO segmentation**, with the broader engineering goal of supporting automated infrastructure inspection rather than treating the task as a simple image-classification problem.

The workflow was designed around the following principle:

> **Audit the data first, understand the annotations, construct a reproducible training dataset, establish a baseline, investigate failure modes, and then improve the system based on evidence.**

The complete workflow was:

```text
Raw Dataset
    ↓
Dataset Audit
    ↓
Mask / Annotation Investigation
    ↓
Dataset Construction
    ↓
YOLO Segmentation Conversion
    ↓
Dataset Validation
    ↓
Baseline Training
    ↓
Controlled Experiments
    ↓
Custom Evaluation
    ↓
Error Analysis
    ↓
Final Test Evaluation
    ↓
Engineering Conclusions
```

---

### 1.2 Dataset Composition

The original dataset contained:

* **11,298 images**
* **11,298 corresponding masks**
* Image format: `.jpg`
* Mask format: `.jpg`
* Resolution: **448 × 448 pixels**

The dataset therefore contained a globally complete image-mask collection.

However, the directory organization required investigation before training.

---

### 1.3 Initial Directory-Level Anomaly

During the initial audit, the training directory showed an unexpected structure:

* `train/images`: **2,294 images**
* `train/masks`: only **9 masks**

At first glance, this appeared to indicate a severe image-mask mismatch.

However, a global audit showed:

* Images without a corresponding mask: **0**
* Masks without a corresponding image: **0**

This distinction was important.

The problem was therefore not missing annotations at the dataset level. Instead, the problem was an **inconsistent split-level directory organization**.

This led to an important engineering decision:

> The source dataset should not be modified destructively or discarded based solely on the local train/mask directory structure.

Instead, the preparation pipeline was designed around a **global image-to-mask lookup**, allowing every image to be matched with its corresponding mask regardless of the original directory placement.

This preserved the original dataset while producing a clean training structure separately.

---

### 1.4 Dataset Integrity Checks

Several integrity checks were performed before training.

#### Image-Mask Matching

The global audit found:

```text
Images without masks: 0
Masks without images: 0
```

Therefore, the complete global dataset had a one-to-one image-mask correspondence.

#### Image Readability

The dataset was checked for unreadable or corrupted image files.

Result:

```text
Unreadable files: 0
```

#### Image Dimensions

The expected image resolution was:

```text
448 × 448
```

The dimension audit found:

```text
Dimension mismatches: 0
```

#### Test Split

The original test split contained:

```text
1,695 images
1,695 masks
```

The test set was treated as a fixed evaluation set and was not used for training or validation.

---

### 1.5 Leakage Investigation

A separate leakage check was performed on the generated YOLO dataset.

The purpose was to ensure that the same image identity did not appear across multiple splits.

The final dataset construction and leakage validation confirmed that train, validation, and test images were separated without stem overlap.

---

## 2. Mask Analysis & Annotation Investigation

### 2.1 Why Mask Analysis Was Necessary

Before converting the masks to YOLO segmentation labels, the annotation structure needed to be understood.

A segmentation mask is not simply a binary label saying whether an image contains a crack.

Instead, it defines the pixels belonging to the crack region.

Therefore, several questions had to be answered:

1. Are masks actually binary?
2. How many images contain no crack?
3. Are empty masks concentrated in a specific subset?
4. Can background-only images be identified reliably?
5. What threshold should be used when converting masks into binary regions?

---

### 2.2 Empty Mask Investigation

The mask analysis identified:

**1,454 empty masks** in total.

An empty mask means that no crack pixels were present according to the mask representation.

A filename-based investigation showed that:

* **1,411 / 1,454 empty masks**
* approximately **97.04%**

were associated with filenames containing the `noncrack` designation.

This provided strong evidence that the dataset intentionally contained background/non-crack examples.

However, **43 empty masks** did not follow that filename pattern.

This was important because it showed that filename-based filtering alone was insufficient.

Therefore, the preparation pipeline used both:

* filename information where available, and
* actual mask content.

The `noncrack` designation was treated as a strong signal rather than blindly assuming that every image without the string was positive.

---

### 2.3 Mask Pixel Analysis

The mask analysis script examined:

* maximum pixel value
* unique pixel values
* most frequent pixel values
* threshold-based foreground detection

Several candidate thresholds were inspected, including:

```text
1
10
50
100
127
128
200
240
```

The main threshold selected for the segmentation conversion pipeline was:

```text
MASK_THRESHOLD = 127
```

Pixels above the threshold were treated as foreground crack pixels.

This threshold was then used consistently in the main mask-to-YOLO conversion workflow and visualization pipeline.

---

### 2.4 Visualization

Random samples were visualized using:

1. Original image
2. Ground-truth mask
3. Overlay between image and mask

The visualization process was used to verify that the masks represented actual crack regions and to inspect the relationship between image appearance and annotation boundaries.

A fixed random seed of:

```text
42
```

was used for reproducibility.

---

## 3. Dataset Preparation Strategy

### 3.1 Why a New YOLO Dataset Was Created

The original dataset structure was not directly suitable for YOLO segmentation training.

Therefore, a separate prepared dataset was generated:

```text
data/yolo_crack/
```

The original source dataset remained unchanged.

This separation provided two benefits:

* preservation of the original data
* reproducibility of the training dataset construction process

The preparation process was implemented in:

```text
scripts/prepare_yolo_dataset.py
```

---

### 3.2 Split Strategy

The preparation pipeline used:

```text
Validation ratio = 20%
Random seed = 42
Class ID = 0
Class name = crack
Mask threshold = 127
```

The test set was **locked first**.

This was an important decision because the test set represents the final generalization benchmark and should not be indirectly altered while constructing the training/validation data.

After locking the test set, the remaining data was used to construct the training and validation subsets.

---

### 3.3 Positive and Background Sampling

The pipeline separately considered:

* crack-containing images
* confirmed background/non-crack images

The goal was to ensure that the training and validation sets contained meaningful background examples rather than allowing the distribution to be determined accidentally by the original directory structure.

After excluding test images, **1,199 confirmed non-crack candidates** were available for the train/validation construction process.

The final training and validation composition was:

| Split          | Positive Images | Background Images | Total Images |
| :------------- | --------------: | ----------------: | -----------: |
| **Train**      |           1,836 |               960 |        2,796 |
| **Validation** |             458 |               239 |          697 |
| **Test**       |           1,474 |               221 |        1,695 |
| **Total**      |       **3,768** |         **1,420** |    **5,188** |

The test composition shown above reflects the later final-test analysis.

---

### 3.4 Background Distribution

The final training set contained:

```text
960 / 2,796 = 34.33% background images
```

The validation set contained:

```text
239 / 697 = 34.29% background images
```

The difference between the two proportions was only:

```text
0.04 percentage points
```

This indicates that the background-image proportions were highly consistent between training and validation.

The result is important because the validation set therefore contained a background distribution similar to the training data, reducing the risk that performance differences were caused simply by a major shift in background prevalence.

---

## 4. YOLO Segmentation Conversion & Validation

### 4.1 Mask-to-YOLO Conversion

YOLO segmentation does not directly consume raster masks.

The binary masks therefore had to be converted into polygon-based segmentation labels.

The conversion process was:

```text
Raster Mask
    ↓
Binary Crack Region
    ↓
Connected / Contour Extraction
    ↓
Polygon Coordinates
    ↓
YOLO Segmentation Label
```

The crack class was assigned:

```text
Class ID = 0
Class name = crack
```

---

### 4.2 Background Images

Background-only images do not contain crack polygons.

Therefore, they were intentionally allowed to have empty YOLO label files.

This was handled explicitly by the validation script rather than treating empty labels as an automatic dataset error.

---

### 4.3 YOLO Dataset Validation

The generated dataset was validated using:

```text
scripts/validate_yolo_dataset.py
```

The validation checked:

* missing labels
* extra labels
* malformed annotation lines
* invalid class IDs
* coordinates outside `[0, 1]`
* malformed polygon coordinates
* invalid label structures
* empty labels

Empty labels were allowed when they corresponded to legitimate background images.

---

### 4.4 Leakage Validation

The generated dataset was also checked using:

```text
scripts/check_leakage.py
```

The purpose was to ensure that image identities did not overlap between:

```text
Train
Validation
Test
```

No train/validation/test stem overlap was detected.

---

## 5. Training Strategy

### 5.1 Compute Strategy

The local development machine was not equipped with a dedicated NVIDIA GPU.

The local machine specifications were:

* Intel Core i7-10510U
* 8 GB RAM
* Intel UHD integrated graphics
* No NVIDIA GPU

Rather than treating this as a blocker, the project separated lightweight engineering tasks from GPU-intensive training.

### Local machine

Used for:

* dataset auditing
* mask analysis
* visualization
* dataset preparation
* YOLO conversion
* validation
* leakage checking
* project organization

### Google Colab

Used for:

* model training
* GPU-accelerated inference
* experiment execution

The training environment used a **Tesla T4 GPU**.

This was an engineering compute decision designed to make experimentation practical without requiring dedicated local GPU hardware.

---

### 5.2 Model Selection

The selected architecture was:

**YOLO11s-seg**

A pretrained YOLO segmentation model was used as the starting point.

Transfer learning was chosen because the project goal was to adapt an established segmentation architecture to infrastructure crack segmentation rather than train a segmentation model entirely from scratch.

---

### 5.3 Training Configuration

The main training configuration was:

| Parameter        | Value       |
| :--------------- | :---------- |
| Model            | YOLO11s-seg |
| Input Resolution | 448 × 448   |
| Batch Size       | 16          |
| Epochs           | 50          |
| Patience         | 15          |
| Optimizer        | AdamW       |
| Learning Rate    | 0.002       |
| Momentum         | 0.9         |
| Workers          | 2           |
| Random Seed      | 42          |

The same general training framework was used across the experiments so that changes in results could be interpreted in relation to the experimental modification.

---

## 5.4 Experimental Design

The project did not stop after obtaining a baseline.

Instead, controlled experiments were used to investigate specific hypotheses about crack segmentation.

The four actual training runs were:

1. **Baseline**
2. **Experiment 1 — Small-Crack Focused Augmentation**
3. **Experiment 2 — No-Mosaic Ablation**
4. **Experiment 3 — Crack-Focused Cropping**

The baseline was not counted as an experiment number.

---

## 5.5 Unified Experiment Comparison

All four runs were evaluated on the same validation set:

```text
697 validation images
458 crack-containing images
239 background images
```

The verified comparison was:

| Model / Experiment       | Crack Mean IoU | Crack Mean Dice | Missed Cracks (out of 458) | Miss Rate (%) | False Positives (out of 239) | Mask mAP50 | Mask mAP50-95 |
| :----------------------- | -------------: | --------------: | -------------------------: | ------------: | ---------------------------: | ---------: | ------------: |
| **Baseline**             |         0.5588 |          0.6859 |                         31 |         6.77% |                            0 |      0.319 |        0.0969 |
| **Exp 1 — Augmentation** |     **0.5677** |      **0.6992** |                         18 |     **3.93%** |                            1 |  **0.327** |        0.0986 |
| **Exp 2 — No-Mosaic**    |         0.5507 |          0.6782 |                         34 |         7.42% |                            0 |      0.310 |        0.0937 |
| **Exp 3 — Cropping**     |     **0.5691** |          0.6976 |                         23 |         5.02% |                            3 |      0.323 |    **0.1010** |

These results provide a more complete picture than relying on a single metric.

For example:

* Experiment 1 achieved the best validation miss rate.
* Experiment 3 achieved the highest Crack Mean IoU.
* Experiment 3 achieved the highest Mask mAP50-95.
* Experiment 2 performed worse than Experiment 1 across the major crack-segmentation metrics.

---

## 5.6 Baseline

The baseline model achieved:

```text
Crack Mean IoU: 0.5588
Crack Mean Dice: 0.6859
Missed cracks: 31 / 458
Miss rate: 6.77%
False positives: 0 / 239
Mask mAP50: 0.319
Mask mAP50-95: 0.0969
```

The baseline established a reference point for evaluating subsequent modifications.

Without a baseline, it would not be possible to determine whether an augmentation or preprocessing change actually improved the system.

---

## 5.7 Experiment 1 — Small-Crack Focused Augmentation

Experiment 1 introduced augmentation intended to improve robustness to small and difficult crack patterns.

The results were:

```text
Crack Mean IoU: 0.5677
Crack Mean Dice: 0.6992
Missed cracks: 18 / 458
Miss rate: 3.93%
False positives: 1 / 239
Mask mAP50: 0.327
Mask mAP50-95: 0.0986
```

Compared with the baseline:

```text
Missed cracks:
31 → 18

Miss rate:
6.77% → 3.93%

Crack Mean IoU:
0.5588 → 0.5677

Crack Mean Dice:
0.6859 → 0.6992
```

This provided evidence that the augmentation strategy improved the model's ability to detect crack-containing validation images.

At the default confidence threshold of **0.25**, Experiment 1 missed:

```text
18 / 458 = 3.93%
```

The final operating confidence threshold was later set to:

```text
0.07
```

However, the previously reported **0.87% miss-rate value is not included as a verified result**, because the threshold-sweep output that supposedly produced that number was not retained in the reviewed notebook.

---

## 5.8 Experiment 2 — No-Mosaic Ablation

Experiment 2 removed Mosaic augmentation.

The motivation was to investigate whether Mosaic was contributing meaningfully to the model's ability to learn small and spatially difficult crack patterns.

The results were:

```text
Crack Mean IoU: 0.5507
Crack Mean Dice: 0.6782
Missed cracks: 34 / 458
Miss rate: 7.42%
False positives: 0 / 239
Mask mAP50: 0.310
Mask mAP50-95: 0.0937
```

Compared with Experiment 1:

```text
Crack Mean IoU:
0.5677 → 0.5507

Missed cracks:
18 → 34
```

The experiment also showed a notable result for the evaluated very-small crack subset:

```text
Very-small cracks:
8 / 8 missed
IoU = 0.0
```

This provides evidence that disabling Mosaic harmed performance on the evaluated very-small crack cases.

The correct engineering interpretation is not that Mosaic is universally necessary for every crack segmentation problem.

Rather:

> In this dataset and experimental setup, removing Mosaic was associated with poorer performance, particularly on the evaluated very-small crack subset.

---

## 5.9 Experiment 3 — Crack-Focused Cropping

Experiment 3 investigated crack-focused cropping.

The cropping strategy increased the number of training images to:

```text
4,593 images
```

The motivation was to increase the visual prominence of crack regions and expose the model to more localized crack patterns.

The results were:

```text
Crack Mean IoU: 0.5691
Crack Mean Dice: 0.6976
Missed cracks: 23 / 458
Miss rate: 5.02%
False positives: 3 / 239
Mask mAP50: 0.323
Mask mAP50-95: 0.1010
```

Experiment 3 achieved:

* highest Crack Mean IoU: **0.5691**
* highest Mask mAP50-95: **0.1010**

The very-small crack analysis also showed:

```text
Very-small crack IoU:
0.0 → 0.1759
```

and:

```text
4 of the 8 previously missed very-small cracks were recovered
```

However, this improvement came with a trade-off:

```text
False positives:
0 → 3
```

Therefore, cropping improved performance on some difficult small-crack cases but also increased false detections on background images.

This is an important engineering trade-off rather than a simple "cropping is better" conclusion.

---

## 5.10 Experiment Selection

The experiments produced different strengths.

### Experiment 1

Best for:

* lowest validation miss rate
* strongest crack detection among the evaluated runs
* good balance between recall and false positives

### Experiment 3

Best for:

* highest Crack Mean IoU
* highest Mask mAP50-95
* improved very-small crack segmentation

But it also introduced more false positives.

### Experiment 2

Performed worst among the compared configurations on the major validation segmentation metrics and had the highest number of missed cracks.

Therefore, the experimental evidence favored retaining Mosaic-based augmentation and treating crack-focused cropping as a promising direction requiring further refinement rather than an unconditional replacement.

---

# 6. Evaluation Methodology

## 6.1 Why Multiple Metrics Were Used

A single metric cannot fully describe the behavior of a crack segmentation system.

The project therefore used:

* Crack Mean IoU
* Crack Mean Dice
* missed crack count
* miss rate
* false positives on background images
* Mask mAP50
* Mask mAP50-95

The custom pixel-level metrics were particularly useful because crack regions can be thin and irregular.

---

## 6.2 Intersection over Union

For a predicted binary mask $P$ and ground-truth mask $G$:

$$
IoU = \frac{|P \cap G|}{|P \cup G|}
$$

IoU measures the overlap between the predicted crack region and the ground-truth crack region.

Higher values indicate better spatial agreement.

---

## 6.3 Dice Score

The Dice score was calculated as:

$$
Dice = \frac{2|P \cap G|}{|P| + |G|}
$$

Dice is especially useful when dealing with relatively small foreground regions because it emphasizes overlap between the predicted and ground-truth pixels.

---

## 6.4 Special Cases

The evaluation implementation explicitly handled zero-area cases.

If:

```text
Ground truth = empty
Prediction = empty
```

then:

```text
IoU = 1.0
Dice = 1.0
```

If:

```text
Ground truth contains crack
Prediction is empty
```

then:

```text
IoU = 0.0
Dice = 0.0
```

This prevented undefined values from appearing in the evaluation.

---

## 6.5 Prediction Rasterization

YOLO outputs segmentation polygons.

For pixel-level evaluation, the predicted polygons were rasterized to the same resolution as the ground-truth masks:

```text
448 × 448
```

This ensured that prediction and ground truth were compared in the same pixel coordinate space.

---

## 6.6 Multiple Predicted Instances

When multiple crack instances were predicted for one image, the individual predictions were combined into one unified binary crack mask.

The combination was performed using pixel-wise logical union / maximum operation.

The final result was therefore a single binary prediction mask representing all predicted crack regions in the image.

---

## 6.7 Evaluation Output

The evaluation process generated per-image measurements including:

```text
image
gt_pixels
gt_area_pct
pred_pixels
pred_area_pct
iou
dice
error_type
```

This allowed the project to move beyond aggregate metrics and investigate individual failure cases.

---

# 7. Error Analysis

## 7.1 Final Test Set

The final test set contained:

```text
1,695 images
```

The final test analysis identified:

```text
1,474 crack-containing images
221 background-only images
```

The results were:

| Test Outcome                | Number of Images | Percentage |
| :-------------------------- | :--------------: | :--------: |
| **Total Test Images**       |       1,695      |   100.00%  |
| **Crack-Containing Images** |       1,474      |   86.96%   |
| ├─ Detected Cracks          |       1,302      |   88.33%   |
| └─ Missed Cracks (FN)       |        172       |   11.67%   |
| **Background-Only Images**  |        221       |   13.04%   |
| ├─ Correctly Rejected (TN)  |        212       |   95.93%   |
| └─ False Positives (FP)     |         9        |    4.07%   |

---

## 7.2 Test Recall and Miss Rate

For crack-containing images:

$$
\text{Recall} = \frac{1302}{1474} = 88.33\%
$$

The corresponding miss rate was:

$$
\text{Miss Rate} = \frac{172}{1474} = 11.67\%
$$

This means the final system detected cracks in approximately 88 out of every 100 crack-containing test images, while missing approximately 12 out of every 100.

---

## 7.3 Background Rejection

Among the 221 background-only test images:

```text
Correctly rejected = 212
False positives = 9
```

Therefore:

```text
Background rejection rate = 95.93%
False-positive rate among background images = 4.07%
```

This demonstrates that the system was generally able to avoid generating crack detections on background-only images, although false positives remained.

---

## 7.4 Source-Level Error Distribution

The 172 missed crack images were further analyzed by source.

| Source Domain           | Missed Crack Images | Percentage of Total Misses |
| :---------------------- | :-----------------: | :------------------------: |
| **Rissbilder**          |         136         |           79.07%           |
| **GAPS384**             |          16         |            9.30%           |
| **Other Sources**       |          20         |           11.63%           |
| **Total Missed Cracks** |       **172**       |         **100.00%**        |

The most important observation is that:

```text
Rissbilder + GAPS384
= 152 / 172 missed images
= 88.37% of all misses
```

This concentration suggests a possible **source/domain generalization gap**.

In other words, the problem is not necessarily only that the model is "weak."

The model may be encountering visual characteristics in specific source domains that differ from the patterns represented sufficiently during training.

---

## 7.5 Crack Size and Difficulty

The error analysis also showed that thin and very small cracks were more difficult to detect than larger surface fractures.

This is consistent with the experimental results:

* removing Mosaic harmed the evaluated very-small crack subset
* crack-focused cropping improved very-small crack IoU
* some small crack instances remained difficult even after augmentation

This suggests that **small-object visibility and spatial resolution** are important factors in the problem.

---

## 7.6 Runtime and Reproducibility Issues

Several runtime issues occurred during notebook execution.

These were implementation/environment issues and should not be interpreted as model-performance failures.

### Contrast Analysis

Some contrast-analysis cells produced an error similar to:

```text
Could not read image: ...
```

The issue was caused by an incorrect/local image path before the required images were correctly loaded or indexed.

This was a file-access/path issue rather than evidence that the images themselves were corrupted.

### Google Drive Save

A later cell attempting to inspect/save information involving the test image directory produced:

```text
PermissionError: Operation not permitted
```

This occurred during Google Drive file handling.

The practical solution was to avoid unnecessarily copying very large image directories to Drive and instead preserve only the required artifacts, such as:

```text
best.pt
evaluation CSVs
results
```

These runtime issues should be documented for reproducibility but are separate from the actual model results.

---

# 8. Engineering Decisions

## 8.1 Preserve the Source Dataset

The original dataset was not destructively reorganized.

Instead:

```text
Original Dataset
       ↓
Reproducible Preparation Script
       ↓
Generated YOLO Dataset
```

This makes it possible to reproduce the training dataset from the original data.

---

## 8.2 Global Mask Lookup

Because the original directory structure contained inconsistent local split organization, the preparation script used a global image-mask mapping.

This avoided incorrectly labeling images as missing annotations simply because their corresponding masks were stored elsewhere.

---

## 8.3 Lock the Test Set First

The test set was locked before constructing the train/validation subsets.

This protects the final evaluation set from accidental contamination during dataset construction.

---

## 8.4 Explicit Background Handling

Background images were deliberately included rather than treating them as noise.

This was important because a real infrastructure inspection system must not only detect cracks when they exist; it must also avoid generating detections on normal surfaces.

---

## 8.5 Use a Baseline Before Optimization

The baseline provided a measurable reference point.

Without it, improvements such as augmentation or cropping could not be evaluated objectively.

---

## 8.6 Controlled Experiments

Each experiment was designed to investigate a specific question.

### Experiment 1

> Does targeted augmentation improve robustness to small/difficult cracks?

### Experiment 2

> How much does removing Mosaic affect crack segmentation?

### Experiment 3

> Can crack-focused cropping improve the representation of difficult crack regions?

This experimental approach turned model improvement into an evidence-driven process rather than random hyperparameter tuning.

---

## 8.7 Evaluate at Pixel Level

Because cracks can be thin, elongated, and irregular, image-level accuracy alone would not adequately represent segmentation quality.

Pixel-level IoU and Dice therefore provided an additional view of spatial segmentation quality.

---

# 9. Limitations

Several limitations remain.

### 9.1 Domain Generalization

The concentration of missed detections in Rissbilder and GAPS384 suggests that the model may not generalize equally well across all source domains.

---

### 9.2 Small and Thin Cracks

Very small cracks remain challenging.

Even though cropping improved the evaluated very-small subset, the overall test analysis still showed missed crack images.

---

### 9.3 False Positives

The final test set contained:

```text
9 false positives / 221 background images
```

Experiment 3 also increased validation false positives from:

```text
0 → 3
```

This demonstrates that improving sensitivity can introduce a trade-off with specificity.

---

### 9.4 Dataset Composition

The test set contains a high proportion of crack-containing images:

```text
1,474 / 1,695 = 86.96%
```

Therefore, test-set image-level recall is useful but should not be interpreted as a complete representation of deployment conditions.

A real infrastructure inspection environment may contain substantially more normal/background images.

---

### 9.5 Threshold Sweep Reproducibility

The final operating confidence threshold was:

```text
0.07
```

However, the previously mentioned validation miss rate of **0.87% at that threshold is not retained in the reviewed notebook**.

Therefore, that number was intentionally excluded from the verified results.

This highlights an important reproducibility lesson:

> Threshold-selection experiments should be saved as explicit result tables rather than relying on temporary notebook output.

---

# 10. Future Work

## 10.1 Domain-Aware Training

Because most missed test cases originated from specific source domains, future work should investigate:

* source-aware sampling
* domain-balanced training
* source-specific augmentation
* domain adaptation
* additional images from underperforming domains

---

## 10.2 Small-Crack Enhancement

Future experiments should investigate methods specifically targeting thin and small cracks, including:

* higher-resolution training
* multi-scale training
* small-object-aware augmentation
* patch-based training
* adaptive cropping
* stronger feature-preserving preprocessing

---

## 10.3 Improved Cropping Strategy

Experiment 3 demonstrated that cropping can improve very-small crack segmentation but also increase false positives.

A future version could therefore use:

* crack-aware crop selection
* controlled crop ratios
* mixed original + cropped images
* background-preserving crops
* hard-negative mining

The goal would be to retain the small-crack benefit without unnecessarily increasing false positives.

---

## 10.4 Hard-Negative Mining

The false-positive images could be collected and reintroduced during training as hard negatives.

This could help the model learn to distinguish:

```text
Actual crack
vs.
Crack-like surface texture
```

---

## 10.5 Threshold Optimization

A reproducible threshold sweep should be implemented and saved to a structured result file.

For example:

```text
confidence
miss_rate
recall
false_positive_rate
mean_iou
mean_dice
```

This would allow the final operating threshold to be selected according to the actual deployment objective.

---

## 10.6 Deployment-Oriented Evaluation

Future evaluation should include:

* inference latency
* memory usage
* model size
* throughput
* confidence calibration
* robustness to different image conditions

This would move the project from an experimental segmentation model toward a deployable inspection system.

---

## 10.7 Real-World Infrastructure Workflow

A future system could extend the current segmentation model into a complete inspection pipeline:

```text
Image / Camera
      ↓
Crack Detection & Segmentation
      ↓
Crack Localization
      ↓
Crack Area / Geometry
      ↓
Severity Estimation
      ↓
Inspection Report
      ↓
Historical Monitoring
```

This would transform the current computer-vision model into a broader infrastructure-inspection solution.

---

# 11. Final Results

## 11.1 Final Test Performance

The final evaluation produced:

```text
Test images: 1,695
Crack-containing images: 1,474
Detected crack images: 1,302
Missed crack images: 172
Background-only images: 221
False positives: 9
Correctly rejected backgrounds: 212
```

Key image-level metrics:

```text
Crack Recall: 88.33%
Crack Miss Rate: 11.67%
Background Rejection Rate: 95.93%
False Positive Rate among background images: 4.07%
```

---

## 11.2 Best Validation Configurations

The experiments showed that there was no single configuration that dominated every metric.

### Best validation miss rate

**Experiment 1 — Small-Crack Focused Augmentation**

```text
Miss rate = 3.93%
```

### Best Crack Mean IoU

**Experiment 3 — Crack-Focused Cropping**

```text
Crack Mean IoU = 0.5691
```

### Best Mask mAP50-95

**Experiment 3 — Crack-Focused Cropping**

```text
Mask mAP50-95 = 0.1010
```

### Worst-performing controlled modification

**Experiment 2 — No-Mosaic Ablation**

```text
Crack Mean IoU = 0.5507
Miss rate = 7.42%
Mask mAP50-95 = 0.0937
```

---

## 11.3 Final Operating Configuration

The final inference configuration used:

```text
Model: YOLO11s-seg
Confidence threshold: 0.07
```

The threshold was selected as the final operating configuration during the project workflow.

The exact validation sweep that produced the threshold choice was not retained sufficiently to support the previously stated 0.87% miss-rate figure, so that value is intentionally not presented as a verified result.

---

# 12. Conclusion

This project evolved from a raw crack-image dataset into a reproducible segmentation pipeline through a sequence of data investigation, controlled dataset preparation, model training, experimentation, and error analysis.

The most important engineering lesson was that the dataset required investigation before model training.

The initial directory structure suggested a severe annotation problem because the training directory contained thousands of images but only a small number of local masks. A global audit showed that the image-mask collection was actually complete, revealing that the primary issue was **dataset organization rather than missing annotations**.

This led to a reproducible preparation strategy based on global image-mask matching instead of destructive manual restructuring.

The resulting pipeline then:

1. audited the raw dataset,
2. analyzed mask structure,
3. identified background images,
4. locked the test set,
5. constructed train/validation subsets,
6. converted raster masks into YOLO segmentation labels,
7. validated the generated dataset,
8. checked for leakage,
9. trained a baseline,
10. conducted three controlled experiments,
11. evaluated segmentation at pixel level,
12. analyzed errors by source domain,
13. and evaluated the final system on a held-out test set.

The experiments also demonstrated that model improvement involved trade-offs.

Small-crack-focused augmentation improved validation crack detection and reduced the miss rate from the baseline.

Removing Mosaic produced worse results, particularly on the evaluated very-small crack subset.

Crack-focused cropping achieved the highest Crack Mean IoU and Mask mAP50-95 and improved some very-small crack cases, but it also increased false positives.

The final test evaluation showed:

```text
88.33% image-level recall on crack-containing images
95.93% background rejection
11.67% crack miss rate
4.07% false-positive rate among background images
```

Most importantly, the error analysis revealed that the majority of missed cracks came from specific source domains, with **Rissbilder and GAPS384 accounting for 88.37% of all missed crack images**.

This shifts the direction of future work from simply "train a stronger model" toward more targeted improvements in:

* domain generalization,
* small-crack representation,
* hard-negative handling,
* threshold calibration,
* and deployment-oriented evaluation.

The project therefore demonstrates not only a trained segmentation model, but a complete **evidence-driven computer vision engineering workflow** in which dataset problems, model behavior, experimental results, and deployment limitations are explicitly investigated and documented.

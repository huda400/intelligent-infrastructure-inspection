Intelligent Infrastructure Inspection System
Technical Project Report

Task: Crack Segmentation and Infrastructure Inspection
Model: YOLO11s-seg
Framework: Ultralytics
Environment: Google Colab / NVIDIA T4
Dataset: Combined crack-image dataset from multiple public sources

1. Dataset Investigation & Audit
1.1 Project Objective

The objective of this project was to develop and evaluate a crack-segmentation system for infrastructure inspection using YOLO instance segmentation.

The project focused not only on model training, but also on understanding the dataset, investigating annotation quality, constructing a controlled working dataset, comparing model configurations, selecting a confidence threshold using validation data, and performing a held-out test evaluation.

The model was trained using instance segmentation, while the custom evaluation also reduced all predicted instances to a single merged binary crack mask for image-level and pixel-level evaluation.

The overall workflow was:

Raw Dataset → Dataset Audit → Mask/Annotation Investigation → Dataset Construction → YOLO Segmentation Conversion → Dataset Validation → Baseline Training → Configuration Comparisons → Validation-Based Model Selection → Confidence Threshold Selection → Held-Out Test Evaluation → Source-Level Error Analysis → Engineering Conclusions

The main engineering questions were:

How reliable and internally consistent is the original dataset?
How should raster masks be converted into YOLO segmentation labels?
How should positive and background images be selected?
How does the model behave across different source datasets?
Do targeted configuration changes improve validation performance?
How large is the gap between validation and held-out test performance?
Which sources and crack characteristics contribute most to failures?
Is the resulting model suitable as a reliable infrastructure-inspection component?
1.2 Dataset Composition

The original dataset contained:

11,298 images
11,298 raster masks
Image format: JPG
Image size: 448 × 448 pixels

The dataset was inspected globally rather than relying only on the initially provided training directory.

The initial local directory structure showed a major anomaly: the training image directory contained 2,294 images, while the corresponding training mask directory contained only 9 mask files.

This did not immediately imply that the dataset was corrupted or unusable. Instead, it motivated a global filename-based lookup across the available dataset.

1.3 Initial Directory-Level Anomaly

The initial training directory contained:

2,294 images
9 masks

A direct local-directory comparison therefore suggested a severe image-mask mismatch.

However, the dataset was subsequently searched globally using filename stems rather than assuming that masks had to exist in the same directory.

The global lookup found matching image-mask pairs for the evaluated files, avoiding an unnecessary assumption that the original directory structure represented the complete annotation organization.

This was an important preprocessing decision because blindly treating the local directory as the complete dataset would have discarded potentially valid annotated images.

1.4 Dataset Integrity Checks

The audit included:

image-to-mask filename matching
image dimension checks
file opening/read checks
missing image-mask pair detection
mask inspection
train/test consistency checks

For the audited files:

2,294 / 2,294 image dimensions were checked successfully.
No files failed the audit's read/open checks.
No missing image-mask pairs were found in the global filename lookup for the evaluated data.

The phrase "unreadable files: 0" should therefore be interpreted as no files failed the audit's read/open checks, rather than as a complete guarantee of pixel-level file integrity under every possible decoder condition.

1.5 Leakage Investigation

A filename-stem overlap check was performed across the evaluated dataset splits.

No direct filename overlap was found between the evaluated train, validation, and test sets.

However, this check has an important limitation: it is a filename-based check.

It does not detect:

duplicate image content under different filenames
near-duplicate images
sibling tiles generated from the same parent image
related crops
visually similar images
parent-image relationships

This is particularly relevant for sources such as CRACK500, where related image tiles may exist.

Therefore, the leakage check provides useful evidence against direct filename duplication, but it does not prove complete independence between all samples.

2. Mask Analysis & Annotation Investigation
2.1 Why Mask Analysis Was Necessary

Before converting the dataset to YOLO segmentation format, the raster masks were investigated to understand:

foreground/background distribution
empty masks
source characteristics
connected crack regions
possible annotation inconsistencies
the effect of the chosen foreground threshold

This step was necessary because segmentation performance is strongly affected by the quality and structure of the target masks.

2.2 Empty Mask Investigation

A total of 1,454 masks were found to contain no foreground pixels under the project's empty-mask check.

Among these:

1,411 were associated with filenames containing noncrack
This represents 97.04% of the empty masks.
The remaining 43 masks were ambiguous and were not automatically classified as background.

This distinction is important because an empty mask does not necessarily mean that the image is intentionally a negative sample. Some may represent annotation or dataset-processing cases requiring further investigation.

2.3 Mask Pixel Analysis

The main mask-processing pipeline used a foreground threshold of:

pixel value > 127

Pixels above this threshold were treated as crack foreground.

The analysis also investigated connected components and contours because a single crack image may contain multiple disconnected regions.

The mask structure was particularly important for understanding the later difference between validation and test performance, where the number of polygon objects per image differed substantially.

2.4 Visualization

Representative images and masks were visually inspected to verify that:

cracks were present where expected
background images were generally empty
masks corresponded to visible crack regions
the mask-to-polygon conversion did not obviously invert foreground/background semantics

Visualization was treated as a qualitative validation step rather than as a substitute for quantitative evaluation.

3. Dataset Preparation Strategy
3.1 Why a Working Dataset Was Constructed

The original dataset contained 11,298 images, but the project did not use all of them in the final training/validation/test pipeline.

A smaller working dataset was constructed to allow:

controlled experimentation
manageable training time
explicit source distribution
a fixed held-out test set
reproducible comparison between configurations

The resulting working dataset contained 5,188 images.

This means that 6,110 images from the original dataset were not used in the current working dataset.

3.2 Split Strategy

The working dataset was divided as follows:

Split	Crack	Background	Total
Train	1,836	960	2,796
Validation	458	239	697
Test	1,474	221	1,695
Total	3,768	1,420	5,188

The test set was locked before final model evaluation and was not used for model or threshold selection.

3.3 Positive and Background Sampling

The training and validation positive images were primarily drawn from:

CRACK500
CFD

The test set introduced substantially more source diversity, including datasets that were not represented in training.

This created a useful held-out evaluation scenario for investigating cross-source generalization.

The test set contained:

1,474 crack images
221 background images
3.4 Background Distribution

Background images were explicitly included in the evaluation to measure false-positive behavior.

The test background set contained 221 images.

Of these:

212 were known noncrack_ images.
9 were empty-label images originating from crack sources.

The 9 crack-source empty-label images were inferred from the difference between total test images and crack-image counts per source:

6 Rissbilder
3 Sylvie Chambon

The exact false-positive distribution between these two groups is not known.

3.5 Source Distribution

The validation set is drawn from the same two sources as the training set.

The training/validation positive distribution was therefore substantially different from the test distribution.

The test set contained the following crack-source distribution:

Source	Crack Images
Rissbilder	567
CRACK500	505
Volker	148
DeepCrack	78
GAPS384	76
CrackTree200	31
Sylvie Chambon	25
CFD	18
forest	18
Eugen Muller	8
Total	1,474

This difference in source composition is central to interpreting the validation-to-test performance gap.

4. YOLO Segmentation Conversion & Validation
4.1 Mask-to-YOLO Conversion

The raster masks were converted into YOLO segmentation labels.

The main conversion pipeline:

loaded the raster mask
applied a foreground threshold of >127
extracted external contours
converted contour coordinates to normalized YOLO coordinates
assigned class ID 0 for crack
stored coordinates with six decimal places

No minimum contour-area threshold was applied during the main conversion.

This preserves small connected regions but also means that small artifacts or fragmented regions may become individual polygons.

Potential consequences include:

fragmented crack labels
very small polygons
sensitivity to JPEG artifacts
thin-crack fragmentation
differences caused by holes or nested structures
4.2 Background Images

Images identified as background were represented by empty YOLO label files.

This allowed the model to be evaluated not only on crack detection but also on its ability to reject images without cracks.

The background set was intentionally retained because false positives are operationally important in an infrastructure-inspection system.

4.3 YOLO Dataset Validation

A dataset validation process was used to check:

images without labels
labels without matching images
malformed label files
invalid class IDs
coordinates outside the expected normalized range

The validator provided useful structural checks, but it should not be interpreted as a complete geometric validator.

For example, it does not fully guarantee detection of:

zero-area polygons
collinear polygon points
every possible NaN-related edge case
all forms of geometrically degenerate annotations
4.4 Leakage Validation

The constructed YOLO dataset was checked again for filename-based overlap.

No direct filename-stem overlap was found between the evaluated splits.

However, as discussed earlier, this does not detect visual duplicates, related tiles, or parent-image relationships.

A stronger future leakage analysis should use perceptual hashing and source/parent grouping.

5. Training Strategy
5.1 Compute Strategy

Training was performed in:

Google Colab
NVIDIA T4 GPU
Python 3.13.15
Ultralytics 8.4.155

The local hardware did not provide a dedicated NVIDIA GPU, so cloud GPU training was used.

5.2 Model Selection

The selected architecture was:

YOLO11s-seg

The model was initialized from pretrained weights.

The model was chosen as a practical balance between segmentation capability and computational cost.

5.3 Training Configuration

The main training configuration was:

Parameter	Value
Model	YOLO11s-seg
Image size	448 × 448
Batch size	16
Epochs	50
Patience	15
Optimizer	AdamW (auto-selected)
Learning rate	0.002
Momentum	0.9
Workers	2
Baseline random seed	0
Exp1–Exp3 random seed	42

All four experiments completed the planned 50 epochs.

The training curves indicated that mAP was still increasing near the end of training, suggesting that the models were not fully converged within the 50-epoch budget.

5.4 Experiment Configurations

Four configurations were evaluated.

Baseline

The baseline used the original training configuration with:

seed = 0
Mosaic = 1.0
Scale = 0.5
Degrees = 0
FlipUD = 0
Copy-paste = 0
Experiment 1 — Combined Augmentation Configuration

Experiment 1 modified several augmentation and training settings together:

Mosaic: 1.0 → 0.5
Scale: 0.5 → 0.3
Degrees: 0 → 5
FlipUD: 0 → 0.1
Copy-paste: 0 → 0.15
Seed: 0 → 42

Because several variables changed simultaneously, Exp1 should be interpreted as a combined augmentation configuration, not as evidence that any single augmentation component caused the observed difference.

Experiment 2

Exp2 was derived from Exp1 by changing:

Mosaic: 0.5 → 0

Therefore, Exp2 should be compared with Exp1 when assessing the effect of removing Mosaic under this configuration.

It is not a direct baseline-to-zero-Mosaic comparison.

Experiment 3

Exp3 extended Exp1 with crack-focused crop augmentation.

The training dataset changed from:

2,796 → 4,593 images

The background share changed from:

34.3% → 20.9%

The crops were:

224 × 224
upscaled 2× to 448 × 448 when the crop dataset was generated
added to the original training images rather than replacing them

The crop-generation process created:

1,797 crops
from 1,836 positive training images
with 39 positive images skipped

Validation and test sets were not changed.

The crops were therefore a training-only augmentation strategy.

5.5 Unified Experiment Comparison

All experiments were compared at the common confidence threshold of 0.25.

Experiment	Crack IoU	Crack Dice	Missed	Miss Rate	FP	mAP50	mAP50-95
Baseline	0.5588	0.6859	31	6.77%	0	0.319	0.0969
Exp1	0.5677	0.6992	18	3.93%	1	0.327	0.0986
Exp2	0.5507	0.6782	34	7.42%	0	0.310	0.0937
Exp3	0.5691	0.6976	23	5.02%	3	0.323	0.1010

The results show that Exp1 had:

the lowest miss rate
the fewest missed crack images among the four configurations
the highest mAP50
slightly higher IoU and Dice than the baseline

Exp3 had the highest crack IoU and mAP50-95, but also more misses and false positives than Exp1 at the common threshold.

5.6 Baseline

The baseline achieved:

Crack Mean IoU: 0.5588
Crack Mean Dice: 0.6859
Missed crack images: 31 / 458
Miss rate: 6.77%
False positives: 0 / 239
mAP50: 0.319
mAP50-95: 0.0969

The baseline served as the reference point for evaluating the subsequent configuration changes.

5.7 Experiment 1 — Combined Augmentation Configuration

At the common comparison threshold of 0.25, Exp1 achieved:

Crack Mean IoU: 0.5677
Crack Mean Dice: 0.6992
Missed crack images: 18 / 458
Miss rate: 3.93%
False positives: 1 / 239
mAP50: 0.327
mAP50-95: 0.0986

The validation threshold sweep for Experiment 1 (Section 9.5) shows that at confidence 0.07 the model missed 4 / 458 crack images (0.87%) with 7 / 239 background false positives. At the common comparison threshold of 0.25, Experiment 1 missed 18 / 458 (3.93%).

The previously reported 0.87% figure is therefore supported by the retained threshold-sweep results rather than being treated as an unverified value.

Exp1 should not be described as a small-crack-specific experiment because multiple variables were changed simultaneously.

5.8 Experiment 2 — Mosaic Removal

Exp2 removed Mosaic from the Exp1 configuration:

Mosaic 0.5 → 0

At confidence 0.25, Exp2 achieved:

Crack Mean IoU: 0.5507
Crack Mean Dice: 0.6782
Missed: 34 / 458
Miss rate: 7.42%
FP: 0 / 239
mAP50: 0.310
mAP50-95: 0.0937
Very-small crack subset

The same eight very-small crack images were evaluated across configurations:

Experiment	Mean IoU	Images with IoU < 0.5
Baseline	0.0768	7 / 8
Exp1	0.2523	5 / 8
Exp2	0.0000	8 / 8
Exp3	0.1759	4 / 8

On this small 8-image sample, removing Mosaic was associated with a drop from Exp1's 0.2523 mean IoU to 0.0000 in Exp2.

This is an indicator rather than conclusive evidence, because the evaluated subset contains only eight images.

5.9 Experiment 3 — Crack-Focused Cropping Extension

Exp3 extended Exp1 with additional training crops.

The training dataset increased from:

2,796 → 4,593 images

The background proportion decreased from:

34.3% → 20.9%

A total of:

1,797 crops
were generated from 1,836 positive images
with 39 images skipped

The crops were 224 × 224 and were upscaled 2× to 448 × 448 during dataset generation before being added to the original training images.

At confidence 0.25, Exp3 achieved:

Crack Mean IoU: 0.5691
Crack Mean Dice: 0.6976
Missed: 23 / 458
Miss rate: 5.02%
FP: 3 / 239
mAP50: 0.323
mAP50-95: 0.1010

Exp3 therefore produced the highest crack IoU and mAP50-95 among the four configurations, but it did not minimize missed detections or false positives.

Very-small subset

On the same eight very-small images:

Baseline: 0.0768
Exp1: 0.2523
Exp2: 0.0000
Exp3: 0.1759

Exp3 was better than Baseline and Exp2 on this subset, but worse than Exp1.

Therefore, cropping cannot be considered a general improvement for the very-small subset. The subset contains only eight images, so the comparison should be treated as a small diagnostic rather than a general conclusion.

5.10 Experiment Selection

At the common validation confidence threshold of 0.25:

Exp1 had the lowest miss rate: 3.93%
Exp1 had the fewest missed crack images: 18 / 458
Exp1 had the highest mAP50: 0.327
Exp3 had slightly higher crack IoU: 0.5691
Exp3 had the highest mAP50-95: 0.1010
Exp3 also had more false positives: 3 / 239

Exp1 was therefore selected for the subsequent confidence-threshold sweep and final held-out test evaluation.

This selection should be understood as a practical project decision, rather than as a formally pre-registered selection rule in the notebook.

The comparison also uses a single seed per experiment, and the observed differences are relatively small. They should therefore not be interpreted as statistically conclusive.

Because Exp1 changed several variables simultaneously, the results cannot isolate the contribution of any individual augmentation.

6. Evaluation Methodology
6.1 Why Multiple Metrics

No single metric fully describes segmentation quality.

The evaluation therefore considered:

IoU
Dice
missed crack images
miss rate
false positives
background rejection
mAP50
mAP50-95
source-level performance

This combination separates:

pixel-level segmentation quality
image-level crack detection
background rejection
source-specific generalization
6.2 IoU

Intersection over Union was calculated as:

[
IoU = \frac{|Prediction \cap GroundTruth|}
{|Prediction \cup GroundTruth|}
]

IoU measures the overlap between predicted and ground-truth crack regions.

Higher IoU indicates better spatial agreement.

6.3 Dice

Dice similarity was calculated as:

[
Dice = \frac{2|Prediction \cap GroundTruth|}
{|Prediction| + |GroundTruth|}
]

Dice is generally more forgiving than IoU when the predicted and target regions have partial overlap.

6.4 Special Cases

For crack-only evaluation:

missed crack images contribute IoU = 0
missed crack images contribute Dice = 0

For the overall test evaluation, an image with both an empty ground truth and an empty prediction is treated as a correct empty-empty case.

This distinction is important because overall metrics and crack-only metrics answer different questions.

6.5 Prediction Rasterization

YOLO inference returns binary instance masks through:

result.masks.data

The predicted masks are resized to 448 × 448 using nearest-neighbor interpolation and merged into one binary crack mask using a pixel-wise maximum operation.

The ground truth used for this evaluation is not the original raster mask.

Instead, it is the YOLO polygon label representation rasterized back into a binary mask using:

cv2.fillPoly

Therefore, the evaluation target includes the effects of the mask-to-polygon conversion and subsequent polygon rasterization.

This means that effects such as filled holes or dropped tiny contours can be reflected in the evaluation target.

6.6 Multiple Predicted Instances

Because YOLO segmentation can return multiple predicted instances for a single image, all predicted binary masks were merged into one image-level crack mask.

This creates an image-level binary segmentation evaluation rather than an instance-by-instance matching evaluation.

The approach is appropriate for the project's primary question:

How accurately can the system identify and localize crack pixels in an image?

7. Error Analysis
7.1 Final Test Set

The final held-out test set contained:

1,695 images
1,474 crack images
221 background images

The final operating configuration was:

Experiment 1 + confidence threshold 0.07

The test set was not used for threshold selection.

7.2 Test Recall and Miss Rate

On the 1,474 crack images:

Detected: 1,302
Missed: 172
Detection/recall: 88.33%
Miss rate: 11.67%

In this analysis, a crack image is counted as detected if the model produces any non-empty predicted mask.

Detection therefore does not imply accurate segmentation.

For example:

CrackTree200 miss rate: 3.23%
CrackTree200 mean IoU: 0.0548

This shows that a model can detect the presence of a crack while producing poor spatial segmentation.

7.3 Background Rejection

Among the 221 background test images:

Correctly rejected: 212
False positives: 9

Therefore:

Background rejection: 95.93%
False-positive rate among background images: 4.07%

This indicates that the final model generally rejected background images successfully, but still produced non-empty predictions on a small number of negative samples.

7.4 Source-Level Error Distribution

The final crack-image results by source were:

Source	Crack Images	Missed	Miss Rate	Mean IoU
Rissbilder	567	136	23.99%	0.2038
CRACK500	505	4	0.79%	0.6054
Volker	148	9	6.08%	0.2494
DeepCrack	78	2	2.56%	0.6000
GAPS384	76	16	21.05%	0.2641
CrackTree200	31	1	3.23%	0.0548
Sylvie Chambon	25	4	16.00%	0.1002
CFD	18	0	0.00%	0.4639
forest	18	0	0.00%	0.5081
Eugen Muller	8	0	0.00%	0.1738
Total	1,474	172	11.67%	0.3719

Rissbilder is particularly important:

Rissbilder represents only 38.5% of crack images in the test set (567 / 1,474), yet accounts for 79.07% of all misses (136 / 172).

This indicates that the overall test error is strongly influenced by one source domain.

However, unseen-source status alone does not explain the result. Rissbilder was absent from training, but DeepCrack was also an unseen source and achieved:

2.56% miss rate
0.6000 mean IoU

This suggests that source-specific visual or annotation characteristics are likely important.

7.5 Crack Size and Difficulty

The test crack images were grouped by approximate crack area:

Size	Images	Missed	Miss Rate
Very small (<0.5%)	61	10	16.4%
Small (0.5–1%)	82	9	11.0%
Medium (1–5%)	939	139	14.8%
Large (≥5%)	392	14	3.6%

The largest absolute contribution to missed detections came from the Medium category:

139 of 172 misses (81%)

Therefore, the test failures cannot be explained simply by very-small cracks.

The medium-sized category contains the majority of the missed images, and crack size is also potentially confounded with source distribution.

A previous eight-image very-small diagnostic showed that Exp3 did not outperform Exp1 on that subset:

Exp1: 0.2523
Exp3: 0.1759

Therefore, the evidence does not support a broad claim that cropping solved the very-small-crack problem.

7.6 Runtime and Reproducibility Issues

The project was developed primarily in Google Colab.

The current repository does not yet contain:

a fully packaged training notebook
pinned requirements.txt
trained weights
all experiment CSV outputs
a formal dataset source/license record
one unified CLI/configuration for training and evaluation

Training paths are also environment-specific.

The retained threshold sweep and evaluation outputs improve reproducibility, but a future version should package the complete experimental environment more systematically.

7.7 Test Ground-Truth Fragmentation

The validation labels contain approximately:

1,094 polygon objects
over 458 crack images
approximately 2.4 objects/image

The test labels contain approximately:

32,088 polygon objects
over 1,474 crack images
approximately 21.8 objects/image

This large difference may reflect:

source-specific annotation style
JPEG artifacts above the conversion threshold
fragmented masks

It likely contributes to the lower test IoU.

The cause was not investigated in this project and should therefore be treated as a hypothesis rather than a confirmed explanation.

7.8 Validation–Test Performance Gap

Exp1 achieved:

Validation mean IoU: 0.5908
Validation Dice: 0.7275

At the same confidence threshold of 0.07, the final test results were:

Test mean IoU: 0.3719
Test Dice: 0.4907

The difference is substantial.

Validation was drawn from CRACK500 and CFD, which were also represented in training, whereas 66% of test crack images came from sources absent from training.

Validation may also be somewhat optimistic because the leakage check was filename-based and related CRACK500 tiles could potentially appear across train and validation.

The gap therefore provides evidence that in-domain validation performance does not fully represent cross-source generalization.

8. Engineering Decisions
8.1 Preserve the Source Dataset

The original dataset was preserved rather than modifying it directly.

This allowed the project to retain:

original masks
original images
source information
the ability to reconstruct or audit preprocessing decisions

Derived datasets were created separately.

8.2 Global Mask Lookup

The initial directory mismatch motivated a global image-mask lookup.

This prevented potentially valid image-mask pairs from being discarded simply because the files were not stored together in the initially inspected directories.

8.3 Lock Test Set First

The test set was kept separate from training and validation experimentation.

Model configuration and threshold decisions were made using validation results.

The final held-out test set was then used to estimate generalization performance.

8.4 Explicit Background Handling

Background images were explicitly included in the working dataset.

This enabled measurement of:

false positives
background rejection
practical image-level behavior

This is important for an inspection system because repeatedly flagging crack-free images can create unnecessary manual inspection workload.

8.5 Baseline Before Optimization

A baseline configuration was trained before evaluating additional configurations.

This established a reference point for:

IoU
Dice
miss rate
false positives
mAP

Without a baseline, later changes would be difficult to interpret.

8.6 Configuration Comparisons

The project used targeted configuration comparisons rather than unstructured trial-and-error tuning.

However, Exp1 changed multiple variables simultaneously.

Therefore, the experiment provides evidence about the combined configuration, not about the independent contribution of each augmentation.

8.7 Evaluate at Pixel Level

mAP alone was not sufficient for this project because infrastructure inspection requires spatially meaningful crack localization.

Custom pixel-level metrics were therefore used:

per-image IoU
per-image Dice
crack-only averages
source-level segmentation quality

This exposed cases where detection succeeded but segmentation quality remained poor.

8.8 Separate Detection from Segmentation

The evaluation intentionally distinguished:

whether a crack was detected at all
how accurately its pixels were segmented

This distinction is particularly important in cases such as CrackTree200, where the miss rate was low but the mean IoU was only 0.0548.

8.9 Treat Source-Level Performance as a First-Class Metric

Overall averages can hide severe source-specific failures.

The Rissbilder results demonstrate this clearly:

38.5% of test crack images
79.07% of all misses

Therefore, source-level analysis was necessary to understand where the model actually fails.

9. Limitations
9.1 Domain Generalization

The largest limitation is the difference between training/validation sources and the heterogeneous test set.

The final test performance is substantially lower than validation performance.

This indicates that the model does not generalize equally well across source domains.

9.2 Small and Thin Cracks

Small and thin cracks remain challenging.

However, the error analysis does not support the conclusion that very-small cracks are the sole or dominant cause of failures.

The largest number of missed images occurred in the Medium category:

139 / 172 misses
81%

The very-small category accounted for only:

10 / 172 misses

The eight-image configuration comparison should therefore be treated as a diagnostic rather than a general conclusion.

9.3 False Positives

The final model produced:

9 false positives
among 221 background images

Although the background rejection rate was high at 95.93%, the remaining false positives indicate that additional hard-negative analysis may be useful.

9.4 Dataset Composition

The working dataset contains only:

5,188 of the original 11,298 images

Therefore:

6,110 original images were not used in the current training/validation/test pipeline.

This limits the diversity represented by the current model and should be investigated before concluding that additional data would not help.

9.5 Confidence Threshold Sweep Reproducibility

The Exp1 validation threshold sweep was retained in:

threshold_tuning_exp1_fine.csv

and the corresponding notebook results were recorded in the threshold-tuning evaluation cell.

The previous 0.87% figure is therefore supported:

4 / 458 = 0.87%

The limitation is not the existence of the sweep, but rather that:

the sweep was performed only for Exp1
there was no equivalent per-experiment optimal-threshold sweep
the selection was not formally pre-registered

Therefore, 0.07 should be interpreted as the selected practical operating threshold for Exp1, not as a universally optimal threshold.

9.6 Test mAP Was Not Calculated

The final held-out test evaluation focused on custom pixel-level and image-level metrics:

IoU
Dice
detection/recall
missed images
false positives
background rejection
source-level results

mAP was not calculated on the final test set.

Therefore, no test mAP conclusion should be inferred from this report.

9.7 Unused Dataset Images

The current working dataset uses 5,188 of the original 11,298 images.

The remaining:

6,110 images

were not incorporated into the current experiments.

Their source distribution, annotation quality, and usefulness for training were not fully investigated.

9.8 Ambiguous Empty Masks

Among the 1,454 empty masks, 43 were not clearly identified as intentional background images.

These ambiguous cases were not automatically assigned a semantic meaning.

Further inspection is required before using them for negative sampling or training.

9.9 Annotation and Polygon Conversion

The conversion from raster masks to polygon labels can introduce differences between the original raster representation and the YOLO representation.

Potential effects include:

contour fragmentation
small components
filled holes
dropped or altered structures
source-dependent polygon complexity

In addition, threshold conventions are not completely identical across all analysis scripts:

prepare_yolo_dataset.py: >127
visualize_samples.py: ≥127
dataset_statistics.py: ≥128
mask_analysis.py: checks max() == 0

These differences are small and do not materially change the main reported result, but they should be standardized for stronger reproducibility.

9.10 Experimental and Reproducibility Limitations

The experiments used:

one seed for Baseline
one seed for Exp1–Exp3
a 50-epoch training budget

The models were still showing increasing mAP near the end of training.

Therefore:

results may vary across seeds
the models may not be fully converged
small differences between configurations should not be overinterpreted

The current repository also lacks a fully pinned environment, complete experiment artifacts, trained weights, and formal dataset metadata.

10. Future Work
10.1 Domain-Aware Training

The strongest next step is to investigate source-specific performance.

In particular:

analyze Rissbilder failure cases
compare image appearance across sources
inspect annotation style
measure crack-size distribution per source
identify source-specific preprocessing differences

The objective should be to understand why some unseen domains generalize well while others fail substantially.

10.2 Small-Crack Enhancement

Future experiments can investigate:

higher-resolution training
alternative crop strategies
stronger thin-object augmentation
morphology-aware preprocessing
segmentation architectures designed for thin structures

The very-small subset should be evaluated on a larger sample before making general claims.

10.3 Improved Cropping Strategy

Exp3 showed that cropping can change the training distribution, but the eight-image very-small comparison did not demonstrate a general improvement over Exp1.

Future crop strategies should therefore be evaluated using:

larger crack-size subsets
source-balanced subsets
controlled comparisons
multiple seeds

The goal should be to improve small-crack representation without reducing generalization or increasing false positives.

10.4 Hard-Negative Mining

The nine test false positives should be inspected individually.

Potential hard negatives include:

texture patterns
shadows
edges
stains
surface discontinuities
background structures resembling cracks

These images can inform future background sampling.

10.5 Threshold Optimization

A broader threshold study could evaluate:

multiple experiments
per-source behavior
precision-recall trade-offs
detection versus segmentation quality
operational cost of false positives versus missed cracks

The final threshold should ideally be selected using an explicitly defined objective.

10.6 Deployment-Oriented Evaluation

Future evaluation should include:

inference latency
throughput
GPU/CPU memory
model size
confidence calibration
image preprocessing cost
batch versus single-image inference

These metrics are necessary before considering deployment in a real inspection pipeline.

10.7 Real-World Infrastructure Workflow

A production-oriented system could eventually integrate:

Image Acquisition → Crack Detection → Segmentation → Severity Estimation → Location/Tracking → Inspection Report

The segmentation model would therefore become one component within a larger infrastructure-inspection workflow.

10.8 Dataset and Evaluation Expansion

The next dataset iteration should include:

Investigating the 6,110 unused images
determine their sources
inspect annotation quality
evaluate their usefulness
recover valid annotated samples where appropriate
Reviewing the 43 ambiguous empty masks
determine whether they are true backgrounds
identify annotation errors
document their final treatment
Investigating test-label fragmentation
inspect why test labels contain approximately 21.8 polygon objects/image compared with approximately 2.4 in validation
determine whether this is source-related or conversion-related
Stronger leakage analysis
perceptual hashing
near-duplicate detection
grouping by parent image
source-aware splitting
Multi-seed experiments
repeat important configurations with multiple random seeds
report mean and variance
determine whether small configuration differences are stable

Increasing the diversity and coverage of training data should be considered alongside understanding the characteristics of the underperforming source domains.

11. Final Results
11.1 Final Test Performance

The final held-out test evaluation used:

Experiment 1 + confidence threshold 0.07

Crack-only evaluation

For the 1,474 crack images:

Mean per-image IoU: 0.3719
Mean per-image Dice: 0.4907
Median IoU: 0.3388
Median Dice: 0.5061

Missed crack images contributed:

IoU = 0
Dice = 0

to the crack-only averages.

Overall evaluation

Across all 1,695 test images, with one empty-empty background case included:

Mean IoU: 0.4485
Mean Dice: 0.5518
Image-level results
Detected crack images: 1,302 / 1,474
Missed crack images: 172 / 1,474
Detection/recall: 88.33%
Background correctly rejected: 212 / 221
False positives: 9 / 221
Background rejection: 95.93%
Background false-positive rate: 4.07%

These detection results should not be interpreted as equivalent to segmentation accuracy because "detected" means that the model produced any non-empty predicted mask.

11.2 Best Validation Configurations

At the common confidence threshold of 0.25:

Experiment	Crack IoU	Crack Dice	Missed	Miss Rate	FP	mAP50	mAP50-95
Baseline	0.5588	0.6859	31	6.77%	0	0.319	0.0969
Exp1	0.5677	0.6992	18	3.93%	1	0.327	0.0986
Exp2	0.5507	0.6782	34	7.42%	0	0.310	0.0937
Exp3	0.5691	0.6976	23	5.02%	3	0.323	0.1010

Exp1 was selected because it provided the lowest miss rate and fewest missed crack images at the common comparison threshold while also achieving the highest mAP50.

Exp3 had slightly higher IoU and mAP50-95, but its higher miss count and false-positive count made Exp1 the more practical choice for the next evaluation stage.

11.3 Final Operating Configuration

The final operating configuration was:

Model: YOLO11s-seg
Selected experiment: Exp1 — Combined Augmentation Configuration
Input size: 448 × 448
Confidence threshold: 0.07

The retained Exp1 validation threshold sweep was:

Confidence	Crack IoU	Missed / 458	FP / 239
0.05	0.5901	2	8
0.07	0.5908	4	7
0.10	0.5893	5	5
0.15	0.5850	7	3
0.20	0.5774	10	1
0.25	0.5677	18	1
0.30	0.5544	30	0
0.40	0.5006	77	0

The selected threshold of 0.07 produced the highest mean per-image crack IoU in the retained validation sweep:

0.5908

At this threshold:

Missed crack images: 4 / 458
Miss rate: 0.87%
Background false positives: 7 / 239

The final threshold was selected using validation data only.

This should be understood as a practical project decision rather than a formally pre-registered optimization rule.

The threshold sweep was performed only for Exp1, so it does not constitute a fair per-experiment optimal-threshold comparison.

The held-out test set was not used to select the threshold.

11.4 Source-Level Final Result

The final test results demonstrate a substantial difference between source domains.

The strongest-performing sources included:

CRACK500: 0.6054 IoU
DeepCrack: 0.6000 IoU

while several sources were substantially harder:

Rissbilder: 0.2038 IoU
GAPS384: 0.2641 IoU
Sylvie Chambon: 0.1002 IoU
CrackTree200: 0.0548 IoU

Rissbilder was the dominant source of missed detections:

136 / 172 misses = 79.07%

despite representing only:

567 / 1,474 = 38.5%

of the test crack images.

This strongly suggests that overall performance is driven not only by crack size, but also by source-specific characteristics and domain differences.

12. Conclusion

This project developed a complete crack-segmentation pipeline extending from raw dataset investigation to held-out test evaluation.

The work demonstrated that model performance cannot be understood from a single validation metric alone.

Several engineering findings were particularly important:

The original dataset contained 11,298 images and masks, but the initial directory structure did not represent the complete annotation organization.
A global filename-based image-mask lookup was necessary to recover valid image-mask relationships.
The working dataset contained 5,188 images, leaving 6,110 original images unused.
The dataset contained 1,454 empty masks, of which 1,411 were associated with noncrack filenames and 43 remained ambiguous.
The four model configurations showed relatively small differences at the common validation threshold.
Exp1 was selected because it achieved the lowest miss rate and fewest missed crack images at confidence 0.25.
Exp3 achieved slightly higher overall validation IoU than Exp1, but its results did not demonstrate a general improvement on the very-small subset. On the same 8-image subset:
Exp1: 0.2523
Exp3: 0.1759

Removing Mosaic in Exp2 was associated with a substantial drop on the same small eight-image subset:

Exp1: 0.2523
Exp2: 0.0000

This is an indicator rather than conclusive evidence because the subset contains only eight images.

The retained Exp1 validation sweep identified 0.07 as the selected operating threshold, where:
mean crack IoU = 0.5908
missed = 4 / 458
miss rate = 0.87%
false positives = 7 / 239
On the held-out test set at confidence 0.07, crack-only mean IoU dropped to 0.3719, while Dice dropped to 0.4907.
The final model detected 88.33% of crack images and rejected 95.93% of background images.
The validation-to-test gap is substantial and indicates that validation performance does not fully represent cross-source generalization.
Rissbilder represents only 38.5% of the test crack images but contributes 79.07% of all missed crack images, making it the dominant source of failure.
Crack size alone does not explain the errors: 139 of 172 misses (81%) occur in the Medium crack-size category.
The large difference in ground-truth polygon fragmentation between validation and test suggests that annotation structure and source-specific characteristics may also influence segmentation quality.

Overall, the project should not be interpreted as having solved crack segmentation for arbitrary infrastructure imagery. Instead, it provides an evidence-based baseline and a structured diagnostic pipeline that reveals where the current model succeeds, where it fails, and what should be investigated next.

The most important next steps are to investigate the 6,110 unused images, review the 43 ambiguous empty masks, analyze the severe Rissbilder failure pattern, investigate test-label fragmentation, strengthen leakage detection, and repeat key experiments across multiple random seeds.

The central engineering lesson is that improving a segmentation system requires more than changing model settings: it requires understanding the dataset, annotation process, source-domain differences, evaluation methodology, and the operational meaning of detection versus accurate segmentation.
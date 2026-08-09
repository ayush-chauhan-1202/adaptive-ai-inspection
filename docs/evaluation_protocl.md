Evaluation Protocol

1. Objective

The evaluation framework measures the ability of the inspection system to distinguish normal components from anomalous/defective components.

Accuracy will not be treated as the sole performance metric.

2. Dataset Splits

The dataset will be divided into:

* Training set
* Validation set
* Test set

The training set will primarily contain normal samples for the initial anomaly-detection experiments.

The validation set will be used for model and threshold selection.

The test set will remain isolated until final evaluation.

3. Image-Level Metrics

The following metrics will be reported:

Precision

Measures the proportion of predicted defects that are actually defective.

Recall

Measures the proportion of actual defects detected by the system.

F1 Score

Harmonic mean of precision and recall.

AUROC

Measures discrimination across classification thresholds.

AUPRC

Measures precision-recall performance across thresholds and is particularly useful when defective samples are relatively rare.

False Positive Rate

Measures the proportion of normal samples incorrectly classified as defective.

False Negative Rate

Measures the proportion of defective samples incorrectly classified as normal.

4. Threshold Selection

The anomaly threshold will be selected using the validation set.

The test set will not be used for threshold tuning.

The initial baseline will use F1 as the threshold-selection criterion.

This strategy may change in later milestones when industrial operating constraints are introduced.

5. Industrial Inspection Considerations

False negatives are particularly important because a missed defect may represent a failed inspection.

Therefore, later experiments should evaluate operating points such as:

Recall at a specified false-positive rate

rather than relying exclusively on a single aggregate metric.

6. Localization Metrics

When pixel-level ground truth is available, the system will additionally be evaluated using:

* Intersection over Union (IoU)
* Dice coefficient
* Pixel-level precision
* Pixel-level recall
* Pixel-level AUROC

Localization evaluation will be introduced in later milestones.

7. Reproducibility

Each experiment should record:

* Random seed
* Dataset version
* Configuration
* Model configuration
* Evaluation metrics
* Code version

The goal is to make experimental results traceable and reproducible.
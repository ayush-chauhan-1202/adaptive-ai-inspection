Adaptive AI Inspection Platform

Problem Definition

1. Problem

Industrial inspection systems must detect defects despite several practical constraints:

* Rare defects
* Limited labeled data
* Small defects
* Previously unseen defect types
* Changing acquisition conditions
* Domain shift
* Human review requirements

Traditional supervised classification assumes that representative examples of the defects to be detected are available during training.

This project investigates an alternative approach: learning the distribution of acceptable components and identifying deviations from that distribution.

2. Initial Problem Formulation

The initial system will treat industrial inspection as an anomaly-detection problem.

Given an inspection image:

Inspection Image
      ↓
Preprocessing
      ↓
Feature Representation
      ↓
Anomaly Detection
      ↓
Anomaly Score
      ↓
Normal / Anomalous

The initial baseline will focus on image-level anomaly detection.

Localization will be introduced as the system becomes more sophisticated.

3. Long-Term Objective

The eventual system should be capable of:

1. Detecting known defects.
2. Detecting previously unseen anomalies.
3. Localizing suspicious regions.
4. Estimating prediction confidence.
5. Routing uncertain cases to human inspectors.
6. Incorporating human annotations into the training dataset.
7. Detecting changes in production data.
8. Supporting controlled model evaluation and retraining.

4. Initial Scope

Milestone 1 will intentionally use a simple baseline.

The objective is not to maximize performance.

The objective is to establish:

* A reproducible dataset pipeline.
* A measurable baseline.
* A standardized evaluation process.
* A foundation for subsequent model improvements.

5. Guiding Principle

The platform will evolve incrementally.

Each subsequent milestone should address a measurable limitation of the current system.

Complexity should be introduced only when it provides a clear technical benefit.
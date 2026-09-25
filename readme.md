# Adaptive AI Inspection Platform

A production-oriented AI inspection platform for industrial anomaly and defect detection.

**Objective**

The project investigates how an AI inspection system can operate under realistic industrial constraints:

* Rare defects
* Limited labeled data
* Small defects
* Previously unseen defects
* Domain shift
* Human review
* Continuous model improvement

The system will progressively evolve from a simple anomaly-detection baseline into a production-oriented ML platform.

**Project Progression**

Classical Baseline
       ↓
Deep Learning
       ↓
Anomaly Detection
       ↓
Unseen Defects
       ↓
Human-in-the-Loop
       ↓
MLOps
       ↓
Production Inference
       ↓
CI/CD
       ↓
Monitoring
       ↓
Drift Detection
       ↓
Continuous Improvement

**Milestones**

**Milestone 0 — Complete**

Problem definition and system architecture.

**Milestone 1 — Complete**

Dataset and classical anomaly-detection baseline.

**Milestone 2 — Complete**

Deep-learning baseline (learned visual embeddings).

**Milestone 3 — Complete**

Modern anomaly detection and localization (patch-based, PatchCore-style, using a pretrained ResNet-18 feature extractor and a patch-level memory bank).

**Milestone 4 — Complete**

Rare and unseen defect detection, validated by holding out a defect type entirely from training and measuring detection rate on it specifically.

**Milestone 5 — Complete**

Human-in-the-loop inspection: a triage layer that routes predictions into AUTO_NORMAL, HUMAN_REVIEW, or AUTO_DEFECT based on confidence, rather than forcing a single automatic call on every image.

**Milestone 6 — In Progress**

MLOps foundation: packaging the inspection pipeline behind a REST API and containerizing it for deployment.

**Milestone 7**

Production inference.

**Milestone 8**

CI/CD.

**Milestone 9**

Monitoring and drift detection.

**Milestone 10**

Continuous improvement and automated retraining.

**Repository Structure**

adaptive-ai-inspection/
├── configs/
├── data/
├── docs/
├── experiments/
├── notebooks/
├── scripts/
├── src/
└── tests/

**Development Philosophy**

Start simple.

Measure everything.

Understand the limitations of the current system.

Introduce additional complexity only when those limitations justify it.

**Current Status**

Milestones 0-5 are complete. The platform has a working PatchCore-style anomaly localization pipeline validated on MVTec AD, tested for generalization to a held-out unseen defect type, and wrapped in a human-in-the-loop triage layer.

Milestone 6 (MLOps foundation) is underway: wrapping the pipeline in a REST API and container so it can be deployed and called as a service rather than run as a script.

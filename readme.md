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

**Milestone 6 — Complete**

MLOps foundation: the pipeline is wrapped behind a FastAPI REST API (`/inspect`, `/health`, `/version`, `/metrics`), containerized with a multi-stage Dockerfile, and trained models are logged, versioned, and registered through MLflow (experiment tracking + Model Registry) instead of living as an in-memory object inside a script.

**Milestone 7 — Complete**

Production inference: the API loads whichever model is aliased `production` in the MLflow Model Registry rather than a hardcoded path, so promoting a new model never requires a redeploy. Structured JSON logging and real input validation (size limits, decode errors) replace the print-statement style of the milestone scripts. Deployed to Cloud Run — see `RUNBOOK.md` for the one-time GCP setup and the live URL.

**Milestone 8 — Complete**

CI/CD via GitHub Actions: `ci.yml` runs ruff + the full pytest suite on every PR; `cd.yml` builds the image, pushes it to Artifact Registry, and deploys to Cloud Run on every merge to `main`.

**Milestone 9 — Complete (scoped)**

Monitoring and drift detection: the API exposes Prometheus-format metrics (`/metrics`), and `scripts/run_drift_report.py` runs an Evidently comparison between training-time and recent production score distributions, producing an HTML report and a non-zero exit code when drift is flagged. A hosted Grafana dashboard was intentionally left out for now — the metrics endpoint is there and wiring a dashboard to it later is a config change, not new code; building one now wasn't worth the account-setup time for a low-traffic demo service.

**Milestone 10 — Scaffolded (manual trigger)**

Continuous improvement: `scripts/retrain_and_promote.py` implements real champion/challenger evaluation (a new model is only promoted if it beats the current production model on AUPRC), and `scripts/add_human_review_batch.py` + a `dvc init`-configured remote version new human-reviewed data batches. What's deliberately not built yet is automatic triggering (on a drift alert or a data-volume threshold) — that's designed in code but left as a manual step until there's real production traffic generating real human-review corrections to trigger on. Automating the trigger later is a small follow-up (a scheduled workflow calling these two scripts), not a redesign.

**Repository Structure**

adaptive-ai-inspection/
├── .github/workflows/     # ci.yml, cd.yml
├── configs/
├── data/
├── docs/
├── experiments/
├── notebooks/
├── scripts/               # training, registration, promotion, drift, data-versioning CLIs
├── src/
│   └── inspection/
│       ├── api/           # FastAPI app (Milestone 6/7)
│       └── serving/       # MLflow pyfunc wrapper, registry loading, GCS publish (Milestone 6/7/10)
├── tests/
├── Dockerfile
├── docker-compose.yml     # local dev: API + MLflow tracking server
└── RUNBOOK.md             # one-time GCP setup, training, deploy, drift checks

**Development Philosophy**

Start simple.

Measure everything.

Understand the limitations of the current system.

Introduce additional complexity only when those limitations justify it.

**Current Status**

Milestones 0-9 are complete; Milestone 10 is scaffolded with manual triggers. The platform has a working PatchCore-style anomaly localization pipeline validated on MVTec AD, tested for generalization to a held-out unseen defect type, wrapped in a human-in-the-loop triage layer, served behind a REST API backed by the MLflow Model Registry, deployed to Cloud Run through an automated CI/CD pipeline, and monitored with a metrics endpoint plus an on-demand drift report.

See `RUNBOOK.md` for how to train the first model, deploy it, promote a new one later, and check for drift. See `docs/architecture.md` for how the serving layer fits into the target architecture.

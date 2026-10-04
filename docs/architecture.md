System Architecture

1. Initial Architecture

The initial system consists of a simple offline inspection pipeline:

Inspection Dataset
       │
       ▼
Data Loading
       │
       ▼
Preprocessing
       │
       ▼
Feature Extraction
       │
       ▼
Anomaly Detection
       │
       ▼
Anomaly Score
       │
       ▼
Evaluation
       │
       ▼
Metrics + Visualization

2. Target Architecture

The platform will progressively evolve toward:

                    Inspection Data
                          │
                          ▼
                   Data Ingestion
                          │
                          ▼
                   Data Validation
                          │
                          ▼
                    Preprocessing
                          │
                          ▼
                 Inspection Model
                          │
                          ▼
              Score + Localization
                          │
                    ┌─────┴─────┐
                    │           │
              High Confidence  Uncertain
                    │           │
                    ▼           ▼
               AI Decision  Human Review
                                │
                                ▼
                           Annotation
                                │
                                ▼
                         Dataset Version
                                │
                                ▼
                         Training Pipeline
                                │
                                ▼
                          Model Evaluation
                                │
                                ▼
                          Model Registry
                                │
                                ▼
                           Deployment
                                │
                                ▼
                           Monitoring
                                │
                         ┌──────┴──────┐
                         ▼             ▼
                      Healthy        Drift
                                       │
                                       ▼
                                Model Evaluation /
                                  Retraining

3. Design Principles

Modularity

Data processing, feature extraction, models, evaluation, and serving should remain independently replaceable.

Reproducibility

Experiments should be reproducible from:

Code Version
+
Dataset Version
+
Configuration
+
Random Seed

Separation of Concerns

Exploratory notebooks should call reusable code from src/ rather than contain the primary implementation.

Incremental Complexity

Production infrastructure will be introduced only when required by the corresponding milestone.

4. Current Architecture Boundary

Milestone 0 only establishes the architecture.

Milestone 1 will implement the first functional pipeline.

5. Target Architecture -> Milestone 6-10 Implementation

Mapping the target-architecture nodes above to what Milestones 6-10 actually
built, for anyone comparing the diagram to the code:

Inspection Model / Score + Localization
    -> src/inspection/serving/pyfunc_model.py (InspectionModel), served via
       POST /inspect in src/inspection/api/main.py

High Confidence / Uncertain / AI Decision / Human Review
    -> unchanged from Milestone 5 (src/inspection/localization/triage.py);
       the API just calls the same classify_score logic per request instead
       of per training-script run

Model Registry
    -> MLflow Model Registry, addressed by alias (models:/<name>@production)
       rather than the deprecated stage API - see
       src/inspection/serving/registry.py

Deployment
    -> Docker (Dockerfile) + Cloud Run, deployed by .github/workflows/cd.yml.
       Deliberately NOT a live tracking-server dependency in production -
       see src/inspection/serving/publish.py for why and how the promoted
       model reaches gs:// instead

Monitoring
    -> GET /metrics (Prometheus format) in the API; no hosted dashboard yet
       (see readme.md's Milestone 9 note on why that was scoped out)

Drift
    -> scripts/run_drift_report.py (Evidently), run on demand against a CSV
       of recent production scores pulled from Cloud Run logs - see
       RUNBOOK.md section 5

Model Evaluation / Retraining
    -> scripts/retrain_and_promote.py (champion/challenger on AUPRC) and
       scripts/add_human_review_batch.py (DVC-versioned correction batches),
       both manually triggered for now - see readme.md's Milestone 10 note
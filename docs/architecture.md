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
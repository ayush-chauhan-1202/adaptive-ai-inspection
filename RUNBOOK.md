# Runbook: one-time GCP setup, training, and deployment

This covers everything needed to go from "code in this branch" to "a live
Cloud Run URL serving real inspection requests." It's split from the README
on purpose: the README explains what the project *is*, this explains the
operational steps to actually run it, most of which only need doing once.

Everything here needs to be run by you, not by Claude: it requires your own
Google Cloud credentials and billing account, which an AI assistant
shouldn't have access to. Budget about 20-30 minutes for the one-time setup
section if this is your first time in GCP.

## 0. Prerequisites

- A Google Cloud project with billing enabled (the free tier covers this
  project's usage, but Cloud Run/Artifact Registry/GCS all require billing
  to be *enabled* on the project even when usage stays in the free tier).
- `gcloud` CLI installed and authenticated (`gcloud auth login`).
- This repo pushed to a GitHub repo you can add secrets to.

Set these once and reuse them in every command below:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"          # any Cloud Run region works
export AR_REPO="adaptive-ai-inspection"
export BUCKET="your-globally-unique-bucket-name"

gcloud config set project "$PROJECT_ID"
```

## 1. One-time GCP setup

```bash
# Enable the APIs this project uses.
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  iamcredentials.googleapis.com

# A place for built Docker images.
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$REGION"

# A bucket for MLflow run artifacts (dev/training) and the published
# production model bundle (what the deployed API actually reads).
gcloud storage buckets create "gs://$BUCKET" --location="$REGION"
```

### Keyless auth for GitHub Actions (Workload Identity Federation)

This avoids ever creating or storing a long-lived service account JSON key -
GitHub's OIDC token is exchanged for short-lived GCP credentials at run
time instead.

```bash
export GITHUB_REPO="your-github-username/adaptive-ai-inspection"

# Service account the CD workflow will act as.
gcloud iam service-accounts create gha-deployer \
  --display-name="GitHub Actions deployer"

export SA_EMAIL="gha-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

# Minimal roles: push images, deploy to Cloud Run, and act as the Cloud Run
# runtime service account.
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/run.admin"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/iam.serviceAccountUser"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.objectAdmin"

# Workload Identity Pool + Provider trusting GitHub's OIDC tokens.
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${GITHUB_REPO}'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

export WIF_POOL_ID=$(gcloud iam workload-identity-pools describe "github-pool" \
  --location="global" --format="value(name)")

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${WIF_POOL_ID}/attribute.repository/${GITHUB_REPO}"

# This is the value for the GCP_WORKLOAD_IDENTITY_PROVIDER secret below.
gcloud iam workload-identity-pools providers describe "github-provider" \
  --location="global" --workload-identity-pool="github-pool" \
  --format="value(name)"
```

### GitHub repository secrets

Settings -> Secrets and variables -> Actions, add:

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | your project id |
| `GCP_REGION` | e.g. `us-central1` |
| `AR_REPOSITORY` | `adaptive-ai-inspection` (or whatever you named it) |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | the full provider name printed above |
| `GCP_SERVICE_ACCOUNT_EMAIL` | `gha-deployer@<project-id>.iam.gserviceaccount.com` |
| `PRODUCTION_MODEL_GCS_PATH` | `gs://<bucket>/production-model` |

(Optional) Create a GitHub Environment named `production` if you want manual
approval before each deploy - `cd.yml` references `environment: production`.
If you'd rather skip that, delete the `environment:` line in the workflow.

## 2. Train and publish the first model

The API has nothing to serve until a model has been trained and promoted at
least once. This step needs the real MVTec AD "bottle" category locally
(download it from MVTec's site and place it under
`data/raw/mvtec_anomaly_detection/`) - the synthetic smoke-test data in
`make_synthetic_dataset.py` is for CI/tests only and produces a meaningless
model.

```bash
pip install -e ".[dev,ml,serve,mlops]"

python scripts/train_and_register.py \
  --data-root data/raw/mvtec_anomaly_detection \
  --category bottle \
  --publish-gcs-path "gs://$BUCKET/production-model"
```

This logs the run to a local `mlruns`/sqlite tracking store, registers the
model, promotes it to the `production` alias (since there's no existing
production model yet), and uploads the model bundle to
`gs://$BUCKET/production-model` - the exact path the deployed API will read
from `PRODUCTION_MODEL_GCS_PATH`.

## 3. Deploy

Push to `main` (or run the `CD` workflow manually from the Actions tab).
`cd.yml` builds the image, pushes it to Artifact Registry, and deploys it to
Cloud Run with `INSPECTION_MODEL_PATH` pointed at the GCS path from step 2.
The workflow's last step prints the live URL.

Verify it:

```bash
curl https://<printed-url>/health
curl https://<printed-url>/version
curl -X POST https://<printed-url>/inspect -F "file=@some_test_image.png"
```

## 4. Promoting a new model later (Milestone 10)

```bash
python scripts/train_and_register.py --data-root data/raw/mvtec_anomaly_detection --category bottle
# -> registers a new version with alias 'staging', e.g. version 3

python scripts/retrain_and_promote.py \
  --data-root data/raw/mvtec_anomaly_detection \
  --category bottle \
  --challenger-version 3 \
  --publish-gcs-path "gs://$BUCKET/production-model"
```

If the challenger wins on AUPRC, it's promoted and published - no redeploy,
no CD run, no code change. That decoupling (new model vs. new deploy) is the
entire point of Milestone 7's registry-driven loading.

## 5. Checking for drift (Milestone 9)

Pull a day of recent `/inspect` scores out of Cloud Run's logs:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="adaptive-ai-inspection" jsonPayload.message="inspect_request"' \
  --format=json --limit=1000 \
  | jq -r '.[] | .jsonPayload.score' \
  | awk 'BEGIN{print "score"} {print}' > current_scores.csv
```

Then compare it against the validation-score distribution from training time
(save that from `train()`'s `validation_scores` in
`scripts/train_and_register.py` during your next training run):

```bash
python scripts/run_drift_report.py \
  --reference-csv reference_scores.csv \
  --current-csv current_scores.csv \
  --output experiments/drift_reports/latest_report.html
```

Non-zero exit code means drift was flagged - wire that into a scheduled
GitHub Actions workflow if/when you want this to run automatically; it's
left as a manually-run script for now (see the project roadmap doc for why).

## 6. Local development loop (no cloud needed)

```bash
docker compose up
# mlflow UI at http://localhost:5000

python scripts/make_synthetic_dataset.py
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/train_and_register.py \
  --data-root data/raw/synthetic_smoke_test --category bottle --no-pretrained-weights

curl -X POST http://localhost:8080/inspect -F "file=@data/raw/synthetic_smoke_test/bottle/test/good/000.png"
```

## 7. Cost control / tearing down

Everything here is sized to stay inside GCP's always-free tier for a
low-traffic demo (`--max-instances 3` on Cloud Run caps worst-case scale; a
few MB in GCS costs nothing measurable). To tear it all down later:

```bash
gcloud run services delete adaptive-ai-inspection --region "$REGION"
gcloud artifacts repositories delete "$AR_REPO" --location="$REGION"
gcloud storage rm -r "gs://$BUCKET"
```

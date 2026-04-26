# BMR-ML-Pipeline

**Industrial-grade Machine Learning & Data Engineering Platform**

[![CI](https://github.com/adityamhaske/BMR-ML-Pipeline_DBMS/actions/workflows/ci.yml/badge.svg)](https://github.com/adityamhaske/BMR-ML-Pipeline_DBMS/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Apache Airflow 2.9](https://img.shields.io/badge/Airflow-2.9-green.svg)](https://airflow.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-blue.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

A production-grade, end-to-end ML data platform built on **publicly available data** at genuine industrial scale. Three fully integrated pillars:

| Pillar | Description | Scale |
|--------|-------------|-------|
| 🧠 **Customer Segmentation** | Batch embedding pipeline → behavioral clustering → precision targeting | 1M+ records |
| ⚙️ **Scalable ETL Infrastructure** | Airflow DAG pipelines with fault isolation, retry orchestration | 3.5M+ records/month |
| 🚀 **Model Deployment & Versioning** | CI/CD, MLflow registry, zero-downtime blue/green ECS deploys | Zero-downtime |

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │           Source Data Layer                   │
                    │  Amazon Reviews · NYC Taxi · Yelp · UCI       │
                    │  (S3 landing zone / NYC Open Data API)         │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │        Orchestration Layer                    │
                    │   Apache Airflow 2.9 — DAG-based ETL          │
                    │   EC2 (t3.large scheduler + c5.2xlarge workers)│
                    └────────────┬──────────────┬──────────────────┘
                                 │              │
          ┌──────────────────────▼──┐    ┌──────▼────────────────────┐
          │  Structured ETL         │    │  Unstructured NLP          │
          │  NYC Taxi → Redshift    │    │  Reviews → Embeddings       │
          │  dbt transformations    │    │  Sentence Transformers      │
          │  Great Expectations     │    │  FAISS / pgvector           │
          └──────────────────────┬──┘    └──────┬────────────────────┘
                                 │              │
                    ┌────────────▼──────────────▼─────────────────┐
                    │           Data Warehouse Layer               │
                    │   AWS Redshift (structured features)          │
                    │   S3 (embeddings, model artifacts)            │
                    │   pgvector (production similarity search)     │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │        ML & Analytics Layer                   │
                    │   HDBSCAN + K-Means segmentation              │
                    │   8 behavioral customer segments               │
                    │   Segment drift detection (weekly)            │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │     Model Serving & Versioning Platform       │
                    │   FastAPI + Uvicorn (p99 < 200ms)             │
                    │   MLflow Model Registry (versioning)          │
                    │   GitHub Actions CI/CD                        │
                    │   ECS Fargate Blue/Green (zero-downtime)      │
                    │   Prometheus + Grafana monitoring             │
                    └─────────────────────────────────────────────┘
```

---

## Public Datasets

| Dataset | Source | Scale | Used For |
|---------|--------|-------|----------|
| Amazon Customer Reviews | [AWS Open Data](https://registry.opendata.aws/amazon-reviews/) | 150M rows | Embedding + segmentation |
| NYC Taxi Trip Records | [NYC Open Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | 3.5M+ rows/month | ETL scale demonstration |
| Yelp Open Dataset | [yelp.com/dataset](https://www.yelp.com/dataset) | 7M reviews | Behavioral NLP signals |
| UCI Online Retail | [UCI ML Repo](https://archive.ics.uci.edu/dataset/352/online+retail) | 500K transactions | Transaction features |

---

## Quick Start (Local Development)

### Prerequisites
- Docker Desktop + Docker Compose
- Python 3.11+
- Make

### 1. Clone and configure
```bash
git clone https://github.com/adityamhaske/BMR-ML-Pipeline_DBMS.git
cd BMR-ML-Pipeline_DBMS
cp .env.example .env
```

### 2. Start all services
```bash
make infra-up
```

This starts:
| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5000 | — |
| Serving API | http://localhost:8000 | — |
| API Docs | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

### 3. Initialize Airflow
```bash
make airflow-init
```

### 4. Download sample data and run embedding pipeline
```bash
make download-sample
make embed-sample
make segment-run
```

### 5. Trigger ETL DAG manually
```bash
make airflow-trigger-etl
```

### 6. Run tests
```bash
make test-unit        # Fast — no external deps
make test-integration # Requires running Docker services
```

---

## Project Structure

```
BMR-ML-Pipeline_DBMS/
├── .github/workflows/       # CI/CD (ci.yml, cd-staging.yml, cd-production.yml)
├── embedding/               # Batch embedding pipeline (Pillar 1)
│   ├── batch_embedder.py    # Core batched inference engine
│   ├── preprocessor.py      # Text cleaning + dedup
│   ├── vector_store.py      # FAISS + pgvector backends
│   └── config.py            # Pydantic-Settings config
├── segmentation/            # Customer segmentation service (Pillar 1)
│   ├── clustering.py        # HDBSCAN + K-Means
│   ├── segment_api.py       # FastAPI segment lookup
│   └── drift_detector.py    # Centroid drift monitoring
├── pipelines/               # Airflow DAGs + operators (Pillar 2)
│   └── dags/
│       ├── nyc_taxi_etl_dag.py     # Monthly structured ETL
│       ├── amazon_reviews_dag.py   # Embedding pipeline trigger
│       └── model_retraining_dag.py # Scheduled retraining
├── etl/                     # ETL extractors, transformers, loaders
├── dbt/                     # SQL transformations on Redshift
├── serving/                 # FastAPI model serving (Pillar 3)
│   ├── api/main.py          # FastAPI app
│   └── model_loader.py      # MLflow registry + hot-swap
├── mlops/                   # Training, evaluation, registry, rollback
├── monitoring/              # Prometheus, Grafana, Evidently
├── infra/
│   ├── terraform/           # AWS infrastructure as code
│   └── docker/              # Dockerfiles
├── tests/                   # Unit, integration, smoke, load tests
├── docker-compose.yml       # Local development stack
└── Makefile                 # All dev commands
```

---

## Key Engineering Features

### Pillar 1 — Customer Segmentation
- **Chunked inference**: batches of 512 records; auto-flush at 50K to bound memory
- **Idempotency**: SHA-256 batch fingerprinting — safe to re-run without duplicates
- **Embedding versioning**: model name stored alongside each vector in FAISS metadata
- **Dual backends**: FAISS for batch jobs; pgvector for real-time similarity search
- **Segment drift detection**: weekly centroid comparison; alert if distribution shifts >15%

### Pillar 2 — ETL Infrastructure
- **Task-level fault isolation**: each Airflow task is atomic; failed transformation ≠ re-extract
- **Idempotent loads**: Redshift MERGE / DuckDB DELETE+INSERT prevent duplicates
- **Partition-aware ingestion**: Hive-partitioned S3 keys (`year=YYYY/month=MM/`)
- **Great Expectations**: schema + statistical validation before every load
- **dbt**: SQL transformations are version-controlled, testable, and documented
- **SLA monitoring**: alert if monthly pipeline not complete by Day+2

### Pillar 3 — Model Serving & Versioning
- **Zero-downtime hot-swap**: asyncio.Lock + double-buffering; old model serves during new model load
- **Blue/green ECS**: ALB weighted traffic shift 10% → 50% → 100% with health gates
- **Auto-rollback**: CodeDeploy reverts on error rate >1% or p99 >500ms for 3 minutes
- **MLflow registry**: `None → Staging → Production → Archived` with automated promotion rules
- **CI/CD pipeline**: lint → unit tests → integration tests → Docker build → staging → production

---

## API Reference

### `POST /v1/segment/predict`
```json
// Request
{
  "record_id": "cust_12345",
  "text": "Great product, fast shipping! Would buy again."
}

// Response
{
  "record_id": "cust_12345",
  "segment_id": 2,
  "segment_label": "high_value_frequent_buyers",
  "confidence": 0.87,
  "model_version": "42",
  "latency_ms": 23.4
}
```

### `GET /v1/model/info`
```json
{
  "model_name": "bmr-customer-segmentation",
  "model_version": "42",
  "stage": "Production",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "n_segments": 8,
  "loaded_at": "2024-04-25T21:00:00Z"
}
```

---

## Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Embedding throughput (CPU) | ≥ 5,000 rec/min | TBD |
| Monthly ETL throughput | ≥ 3.5M records | NYC Taxi validated |
| Serving API p99 latency | < 200ms | TBD |
| Pipeline latency vs sequential | 35% reduction | Parallel task groups |
| Segment precision vs generic | +20% marketing ROI | A/B simulation |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| NLP/Embeddings | sentence-transformers, torch |
| ETL Orchestration | Apache Airflow 2.9 |
| Data Warehouse | AWS Redshift + dbt |
| Vector Storage | FAISS + pgvector |
| Clustering | HDBSCAN + scikit-learn K-Means |
| Serving | FastAPI + Uvicorn + Gunicorn |
| MLOps | MLflow (tracking + registry) |
| Containerization | Docker + AWS ECR |
| CI/CD | GitHub Actions |
| Deployment | AWS ECS Fargate (blue/green) |
| Infrastructure | Terraform |
| Data Quality | Great Expectations |
| Monitoring | Prometheus + Grafana + Evidently AI |
| Local AWS | LocalStack |

---

## Development Commands

```bash
make help              # Full command list
make infra-up          # Start all local services
make test-unit         # Unit tests (fast, no deps)
make test-integration  # Integration tests
make lint              # Ruff lint + format
make typecheck         # Mypy type check
make dbt-run           # Run dbt models (local DuckDB)
make drift-report      # Evidently drift report
make load-test         # Locust load test
make tf-plan           # Terraform plan (AWS infra)
```

---

## License

MIT © Aditya Mhaske

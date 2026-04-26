# =============================================================================
# BMR-ML-Pipeline Makefile
# =============================================================================
# Usage: make <target>
# Run `make help` to see all available commands.
# =============================================================================

.DEFAULT_GOAL := help
SHELL         := /bin/bash
PROJECT_NAME  := bmr-ml-pipeline
PYTHON        := python3.11
PIP           := pip
VENV          := .venv
VENV_BIN      := $(VENV)/bin

# Colours for pretty output
BOLD  := \033[1m
GREEN := \033[0;32m
CYAN  := \033[0;36m
RESET := \033[0m

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "$(BOLD)$(GREEN)BMR-ML-Pipeline — Available Commands$(RESET)"
	@echo "────────────────────────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ─── Environment ──────────────────────────────────────────────────────────────
.PHONY: venv
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip setuptools wheel
	@echo "$(GREEN)✓ Virtual environment created at $(VENV)$(RESET)"
	@echo "$(CYAN)Activate with: source $(VENV)/bin/activate$(RESET)"

.PHONY: install
install: ## Install base + dev requirements
	$(PIP) install -r requirements/dev.txt
	$(PIP) install -e ".[dev]" 2>/dev/null || true
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

.PHONY: install-serving
install-serving: ## Install serving-only requirements
	$(PIP) install -r requirements/serving.txt
	@echo "$(GREEN)✓ Serving dependencies installed$(RESET)"

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed$(RESET)"

# ─── Code Quality ─────────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run ruff linter
	ruff check . --fix

.PHONY: format
format: ## Run ruff formatter
	ruff format .

.PHONY: typecheck
typecheck: ## Run mypy type checker
	mypy embedding/ segmentation/ etl/ serving/ mlops/ pipelines/ --ignore-missing-imports

.PHONY: quality
quality: lint format typecheck ## Run all code quality checks

# ─── Testing ──────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run all tests
	pytest tests/ -v

.PHONY: test-unit
test-unit: ## Run unit tests only (fast, no external deps)
	pytest tests/unit/ -v -m "unit"

.PHONY: test-integration
test-integration: ## Run integration tests (requires Airflow + Postgres)
	pytest tests/integration/ -v -m "integration"

.PHONY: test-smoke
test-smoke: ## Run smoke tests against running service
	pytest tests/smoke/ -v -m "smoke"

.PHONY: test-coverage
test-coverage: ## Run tests with HTML coverage report
	pytest tests/ --cov --cov-report=html
	@echo "$(GREEN)✓ Coverage report at htmlcov/index.html$(RESET)"

# ─── Local Infrastructure (Docker Compose) ────────────────────────────────────
.PHONY: infra-up
infra-up: ## Start all local services (Airflow, Postgres, Redis, MLflow, Prometheus, Grafana)
	cp -n .env.example .env 2>/dev/null || true
	docker compose up -d
	@echo "$(GREEN)✓ Local infrastructure started$(RESET)"
	@echo "$(CYAN)  Airflow UI:    http://localhost:8080$(RESET)"
	@echo "$(CYAN)  MLflow UI:     http://localhost:5000$(RESET)"
	@echo "$(CYAN)  Prometheus:    http://localhost:9090$(RESET)"
	@echo "$(CYAN)  Grafana:       http://localhost:3000$(RESET)"
	@echo "$(CYAN)  Serving API:   http://localhost:8000$(RESET)"

.PHONY: infra-down
infra-down: ## Stop all local services
	docker compose down
	@echo "$(GREEN)✓ Local infrastructure stopped$(RESET)"

.PHONY: infra-clean
infra-clean: ## Stop services and remove volumes
	docker compose down -v --remove-orphans
	@echo "$(GREEN)✓ Local infrastructure cleaned$(RESET)"

.PHONY: infra-logs
infra-logs: ## Tail logs from all services
	docker compose logs -f

.PHONY: airflow-logs
airflow-logs: ## Tail Airflow scheduler logs
	docker compose logs -f airflow-scheduler

# ─── Airflow ──────────────────────────────────────────────────────────────────
.PHONY: airflow-init
airflow-init: ## Initialize Airflow DB and create admin user
	docker compose run --rm airflow-webserver airflow db migrate
	docker compose run --rm airflow-webserver airflow users create \
		--username admin --password admin --firstname Admin \
		--lastname User --role Admin --email admin@bmrml.local
	@echo "$(GREEN)✓ Airflow initialized (admin/admin)$(RESET)"

.PHONY: airflow-trigger-etl
airflow-trigger-etl: ## Manually trigger the NYC Taxi ETL DAG
	docker compose exec airflow-scheduler airflow dags trigger nyc_taxi_monthly_etl

.PHONY: airflow-trigger-embed
airflow-trigger-embed: ## Manually trigger the embedding pipeline DAG
	docker compose exec airflow-scheduler airflow dags trigger amazon_reviews_embedding

# ─── dbt ──────────────────────────────────────────────────────────────────────
.PHONY: dbt-deps
dbt-deps: ## Install dbt packages
	cd dbt && dbt deps

.PHONY: dbt-run
dbt-run: ## Run dbt models (local DuckDB)
	cd dbt && dbt run --target local

.PHONY: dbt-test
dbt-test: ## Run dbt data tests
	cd dbt && dbt test --target local

.PHONY: dbt-docs
dbt-docs: ## Generate and serve dbt docs
	cd dbt && dbt docs generate && dbt docs serve --port 8001

# ─── Embedding Pipeline ───────────────────────────────────────────────────────
.PHONY: embed-sample
embed-sample: ## Run embedding pipeline on sample data (local)
	$(PYTHON) -m embedding.batch_embedder \
		--input data/sample/amazon_reviews_sample.json \
		--output data/embeddings/sample/ \
		--batch-size 64 \
		--device cpu

.PHONY: embed-full
embed-full: ## Run embedding pipeline on full dataset (requires S3 or local data)
	$(PYTHON) -m embedding.batch_embedder \
		--input s3://$(S3_RAW_BUCKET)/amazon-reviews/ \
		--output s3://$(S3_EMBEDDINGS_BUCKET)/reviews/ \
		--batch-size 512 \
		--device $(EMBEDDING_DEVICE)

# ─── Segmentation ─────────────────────────────────────────────────────────────
.PHONY: segment-run
segment-run: ## Run clustering on latest embeddings
	$(PYTHON) -m segmentation.clustering \
		--embeddings data/embeddings/sample/ \
		--n-clusters 8 \
		--output data/segments/

.PHONY: segment-serve
segment-serve: ## Start segmentation FastAPI service locally
	uvicorn segmentation.segment_api:app --host 0.0.0.0 --port 8001 --reload

# ─── Model Serving ────────────────────────────────────────────────────────────
.PHONY: serve
serve: ## Start model serving API locally
	uvicorn serving.api.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: serve-prod
serve-prod: ## Start model serving API (production mode)
	gunicorn serving.api.main:app \
		-w $(SERVING_WORKERS) \
		-k uvicorn.workers.UvicornWorker \
		--bind $(SERVING_HOST):$(SERVING_PORT)

# ─── MLflow ───────────────────────────────────────────────────────────────────
.PHONY: mlflow-ui
mlflow-ui: ## Start local MLflow tracking server
	mlflow server \
		--backend-store-uri sqlite:///mlruns.db \
		--default-artifact-root ./mlartifacts \
		--host 0.0.0.0 \
		--port 5000

.PHONY: train
train: ## Run model training with MLflow tracking
	$(PYTHON) mlops/train.py

.PHONY: evaluate
evaluate: ## Run model evaluation
	$(PYTHON) mlops/evaluate.py

# ─── Terraform ────────────────────────────────────────────────────────────────
.PHONY: tf-init
tf-init: ## Initialize Terraform
	cd infra/terraform && terraform init

.PHONY: tf-plan
tf-plan: ## Plan Terraform changes
	cd infra/terraform && terraform plan -var-file=terraform.tfvars.example

.PHONY: tf-apply
tf-apply: ## Apply Terraform changes (CAUTION: provisions real AWS resources)
	cd infra/terraform && terraform apply -var-file=terraform.tfvars

.PHONY: tf-destroy
tf-destroy: ## Destroy all Terraform-managed AWS resources
	cd infra/terraform && terraform destroy -var-file=terraform.tfvars

# ─── Data ─────────────────────────────────────────────────────────────────────
.PHONY: download-sample
download-sample: ## Download sample datasets for local development
	@echo "$(CYAN)Downloading sample datasets...$(RESET)"
	$(PYTHON) scripts/download_samples.py
	@echo "$(GREEN)✓ Sample data ready in data/sample/$(RESET)"

.PHONY: validate-data
validate-data: ## Run Great Expectations validation on sample data
	$(PYTHON) -m pytest etl/validation/ -v -m "unit"

# ─── Monitoring ───────────────────────────────────────────────────────────────
.PHONY: drift-report
drift-report: ## Generate Evidently drift report
	$(PYTHON) monitoring/drift/evidently_report.py

.PHONY: load-test
load-test: ## Run Locust load test against serving API
	locust -f tests/load/locustfile.py \
		--host http://localhost:8000 \
		--users 50 --spawn-rate 5 \
		--run-time 60s --headless

# ─── Docker ───────────────────────────────────────────────────────────────────
.PHONY: docker-build-serving
docker-build-serving: ## Build serving Docker image
	docker build -f infra/docker/serving/Dockerfile \
		-t $(PROJECT_NAME)-serving:latest .

.PHONY: docker-build-airflow
docker-build-airflow: ## Build custom Airflow Docker image
	docker build -f infra/docker/airflow/Dockerfile \
		-t $(PROJECT_NAME)-airflow:latest .

.PHONY: docker-push
docker-push: ## Push images to ECR
	aws ecr get-login-password --region $(AWS_REGION) \
		| docker login --username AWS --password-stdin $(ECR_REGISTRY)
	docker tag $(PROJECT_NAME)-serving:latest $(ECR_REGISTRY)/$(ECR_SERVING_REPO):latest
	docker push $(ECR_REGISTRY)/$(ECR_SERVING_REPO):latest

# ─── Housekeeping ─────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove build artifacts and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "$(GREEN)✓ Cleaned$(RESET)"

.PHONY: check-env
check-env: ## Verify required environment variables are set
	$(PYTHON) scripts/check_env.py

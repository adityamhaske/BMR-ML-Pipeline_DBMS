# =============================================================================
# Custom Airflow Dockerfile
# Extends official image with project-specific dependencies
# =============================================================================
FROM apache/airflow:2.9.1-python3.11

USER root

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install project dependencies into Airflow's Python environment
COPY requirements/base.txt /tmp/requirements-base.txt
RUN pip install --no-cache-dir -r /tmp/requirements-base.txt

# Additional Airflow providers
RUN pip install --no-cache-dir \
    apache-airflow-providers-amazon>=8.21.0 \
    apache-airflow-providers-postgres>=5.11.0 \
    dbt-core>=1.8.0 \
    dbt-duckdb>=1.8.0 \
    great-expectations>=0.18.14

# Pre-download embedding model for faster DAG execution
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "\
    from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${EMBEDDING_MODEL}', cache_folder='/home/airflow/.cache/sentence_transformers')"

WORKDIR /opt/airflow

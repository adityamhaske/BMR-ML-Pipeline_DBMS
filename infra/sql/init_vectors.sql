-- =============================================================================
-- pgvector initialization SQL
-- Run on postgres-vectors container startup
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id            SERIAL PRIMARY KEY,
    record_id     TEXT NOT NULL,
    embedding     VECTOR(384),   -- all-MiniLM-L6-v2 dimension
    model_name    TEXT,
    model_version TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_embeddings_record_id UNIQUE (record_id)
);

-- IVFFlat index for approximate nearest neighbor search
-- Requires at least ~(lists * 39) rows before it becomes effective
CREATE INDEX IF NOT EXISTS embeddings_vector_cosine_idx
    ON embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Segment assignments table
CREATE TABLE IF NOT EXISTS segment_assignments (
    id            SERIAL PRIMARY KEY,
    record_id     TEXT NOT NULL REFERENCES embeddings(record_id) ON DELETE CASCADE,
    segment_id    INTEGER NOT NULL,
    segment_label TEXT,
    confidence    FLOAT,
    algorithm     TEXT DEFAULT 'kmeans',
    run_timestamp TIMESTAMPTZ NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_segment_record_id UNIQUE (record_id)
);

CREATE INDEX IF NOT EXISTS idx_segment_assignments_segment_id
    ON segment_assignments(segment_id);

-- Pipeline runs metadata
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id            SERIAL PRIMARY KEY,
    dag_id        TEXT NOT NULL,
    run_id        TEXT NOT NULL UNIQUE,
    target_period TEXT,
    row_count     BIGINT,
    status        TEXT DEFAULT 'running',
    started_at    TIMESTAMPTZ DEFAULT NOW(),
    completed_at  TIMESTAMPTZ
);

SELECT 'pgvector schema initialized' AS status;

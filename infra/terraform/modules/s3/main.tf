# =============================================================================
# Terraform Module — S3 Buckets
# =============================================================================
resource "aws_s3_bucket" "raw" {
  bucket        = "${var.project}-${var.environment}-raw-data"
  force_destroy = var.environment != "production"
}

resource "aws_s3_bucket" "processed" {
  bucket        = "${var.project}-${var.environment}-processed-data"
  force_destroy = var.environment != "production"
}

resource "aws_s3_bucket" "embeddings" {
  bucket        = "${var.project}-${var.environment}-embeddings"
  force_destroy = var.environment != "production"
}

resource "aws_s3_bucket" "artifacts" {
  bucket        = "${var.project}-${var.environment}-model-artifacts"
  force_destroy = var.environment != "production"
}

# ─── Versioning (artifacts bucket — must keep model versions) ─────────────────
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ─── Server-side encryption ───────────────────────────────────────────────────
resource "aws_s3_bucket_server_side_encryption_configuration" "all" {
  for_each = {
    raw        = aws_s3_bucket.raw.id
    processed  = aws_s3_bucket.processed.id
    embeddings = aws_s3_bucket.embeddings.id
    artifacts  = aws_s3_bucket.artifacts.id
  }

  bucket = each.value

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ─── Block public access ──────────────────────────────────────────────────────
resource "aws_s3_bucket_public_access_block" "all" {
  for_each = {
    raw        = aws_s3_bucket.raw.id
    processed  = aws_s3_bucket.processed.id
    embeddings = aws_s3_bucket.embeddings.id
    artifacts  = aws_s3_bucket.artifacts.id
  }

  bucket                  = each.value
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ─── Lifecycle: raw data → Glacier after 90 days ──────────────────────────────
resource "aws_s3_bucket_lifecycle_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id

  rule {
    id     = "archive-raw-data"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# ─── Outputs ──────────────────────────────────────────────────────────────────
output "raw_bucket_name"         { value = aws_s3_bucket.raw.bucket }
output "raw_bucket_arn"          { value = aws_s3_bucket.raw.arn }
output "processed_bucket_name"   { value = aws_s3_bucket.processed.bucket }
output "processed_bucket_arn"    { value = aws_s3_bucket.processed.arn }
output "embeddings_bucket_name"  { value = aws_s3_bucket.embeddings.bucket }
output "embeddings_bucket_arn"   { value = aws_s3_bucket.embeddings.arn }
output "artifacts_bucket_name"   { value = aws_s3_bucket.artifacts.bucket }
output "artifacts_bucket_arn"    { value = aws_s3_bucket.artifacts.arn }

variable "project"     { type = string }
variable "environment" { type = string }

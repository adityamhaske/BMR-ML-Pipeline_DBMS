#!/bin/bash
# =============================================================================
# LocalStack Init Script — Creates AWS resources on LocalStack startup
# =============================================================================
echo "Creating S3 buckets on LocalStack..."

ENDPOINT="http://localhost:4566"
REGION="us-east-1"

# Create S3 buckets
for bucket in bmr-ml-raw-data bmr-ml-processed-data bmr-ml-embeddings bmr-ml-model-artifacts bmr-ml-terraform-state; do
  aws --endpoint-url=$ENDPOINT --region=$REGION s3 mb s3://$bucket 2>/dev/null || true
  echo "  ✓ s3://$bucket"
done

# Create SNS topic for alerts
aws --endpoint-url=$ENDPOINT --region=$REGION sns create-topic \
  --name bmr-ml-alerts 2>/dev/null || true
echo "  ✓ SNS topic: bmr-ml-alerts"

# Create sample Secrets Manager secrets
aws --endpoint-url=$ENDPOINT --region=$REGION secretsmanager create-secret \
  --name "bmr-ml-pipeline/redshift/password" \
  --secret-string '{"password": "local-dev-password"}' 2>/dev/null || true
echo "  ✓ Secrets Manager: redshift credentials"

echo "LocalStack initialization complete!"

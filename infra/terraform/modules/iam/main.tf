# =============================================================================
# Terraform Module — IAM Roles & Policies
# =============================================================================
data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "ecs_task_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# ─── Airflow EC2 Role ─────────────────────────────────────────────────────────
resource "aws_iam_role" "airflow_ec2" {
  name               = "${var.project}-${var.environment}-airflow-ec2"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_role_policy" "airflow_s3" {
  name   = "S3Access"
  role   = aws_iam_role.airflow_ec2.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject", "s3:PutObject", "s3:DeleteObject",
          "s3:ListBucket", "s3:GetBucketLocation",
          "s3:CopyObject", "s3:HeadObject"
        ]
        Resource = [
          var.raw_bucket_arn, "${var.raw_bucket_arn}/*",
          var.processed_bucket_arn, "${var.processed_bucket_arn}/*",
          var.embeddings_bucket_arn, "${var.embeddings_bucket_arn}/*",
          var.artifacts_bucket_arn, "${var.artifacts_bucket_arn}/*",
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = "arn:aws:sns:*:${var.account_id}:${var.project}-*"
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = "arn:aws:secretsmanager:*:${var.account_id}:secret:${var.project}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData", "cloudwatch:GetMetricStatistics"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "airflow" {
  name = "${var.project}-${var.environment}-airflow"
  role = aws_iam_role.airflow_ec2.name
}

# ─── ECS Task Execution Role ──────────────────────────────────────────────────
resource "aws_iam_role" "ecs_execution" {
  name               = "${var.project}-${var.environment}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
}

resource "aws_iam_role_policy_attachment" "ecs_execution_basic" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ─── ECS Task Role (app permissions) ─────────────────────────────────────────
resource "aws_iam_role" "ecs_task" {
  name               = "${var.project}-${var.environment}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  name   = "S3ReadArtifacts"
  role   = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        var.artifacts_bucket_arn, "${var.artifacts_bucket_arn}/*",
        var.embeddings_bucket_arn, "${var.embeddings_bucket_arn}/*",
      ]
    }]
  })
}

# ─── Outputs ──────────────────────────────────────────────────────────────────
output "airflow_instance_profile_name" { value = aws_iam_instance_profile.airflow.name }
output "ecs_execution_role_arn"        { value = aws_iam_role.ecs_execution.arn }
output "ecs_task_role_arn"             { value = aws_iam_role.ecs_task.arn }

variable "project"               { type = string }
variable "environment"           { type = string }
variable "account_id"            { type = string }
variable "raw_bucket_arn"        { type = string }
variable "processed_bucket_arn"  { type = string }
variable "embeddings_bucket_arn" { type = string }
variable "artifacts_bucket_arn"  { type = string }

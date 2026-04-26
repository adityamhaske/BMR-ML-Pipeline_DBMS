# =============================================================================
# Terraform Variables — BMR-ML-Pipeline
# =============================================================================

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming and tagging"
  type        = string
  default     = "bmr-ml-pipeline"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "staging"
  validation {
    condition     = contains(["local", "staging", "production"], var.environment)
    error_message = "environment must be one of: local, staging, production"
  }
}

# ─── Networking ───────────────────────────────────────────────────────────────
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

# ─── EC2 (Airflow) ────────────────────────────────────────────────────────────
variable "airflow_instance_type" {
  description = "EC2 instance type for Airflow scheduler and webserver"
  type        = string
  default     = "t3.large"
}

variable "ec2_key_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
  default     = ""
}

# ─── Redshift ─────────────────────────────────────────────────────────────────
variable "redshift_master_username" {
  description = "Redshift master username"
  type        = string
  sensitive   = true
}

variable "redshift_master_password" {
  description = "Redshift master password (min 8 chars, mixed case, numbers)"
  type        = string
  sensitive   = true
}

variable "redshift_node_type" {
  description = "Redshift node type"
  type        = string
  default     = "dc2.large"
}

variable "redshift_number_of_nodes" {
  description = "Number of Redshift nodes"
  type        = number
  default     = 2
}

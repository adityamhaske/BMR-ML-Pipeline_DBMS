# =============================================================================
# Terraform Root Module — BMR-ML-Pipeline AWS Infrastructure
# =============================================================================
terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state — uncomment for team/production use
  # backend "s3" {
  #   bucket         = "bmr-ml-terraform-state"
  #   key            = "bmr-ml-pipeline/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "bmr-ml-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "BMR-ML-Pipeline"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "aditya-mhaske"
    }
  }
}

# ─── Data sources ─────────────────────────────────────────────────────────────
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ─── Modules ──────────────────────────────────────────────────────────────────
module "s3" {
  source      = "./modules/s3"
  project     = var.project_name
  environment = var.environment
}

module "iam" {
  source              = "./modules/iam"
  project             = var.project_name
  environment         = var.environment
  account_id          = data.aws_caller_identity.current.account_id
  raw_bucket_arn      = module.s3.raw_bucket_arn
  processed_bucket_arn = module.s3.processed_bucket_arn
  embeddings_bucket_arn = module.s3.embeddings_bucket_arn
  artifacts_bucket_arn  = module.s3.artifacts_bucket_arn
}

module "networking" {
  source      = "./modules/networking"
  project     = var.project_name
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

module "ec2_airflow" {
  source            = "./modules/ec2"
  project           = var.project_name
  environment       = var.environment
  instance_type     = var.airflow_instance_type
  subnet_id         = module.networking.private_subnet_ids[0]
  security_group_ids = [module.networking.airflow_sg_id]
  iam_instance_profile = module.iam.airflow_instance_profile_name
  key_name          = var.ec2_key_name
}

module "redshift" {
  source             = "./modules/redshift"
  project            = var.project_name
  environment        = var.environment
  subnet_ids         = module.networking.private_subnet_ids
  security_group_ids = [module.networking.redshift_sg_id]
  master_username    = var.redshift_master_username
  master_password    = var.redshift_master_password
  node_type          = var.redshift_node_type
  number_of_nodes    = var.redshift_number_of_nodes
}

module "ecs" {
  source             = "./modules/ecs"
  project            = var.project_name
  environment        = var.environment
  vpc_id             = module.networking.vpc_id
  subnet_ids         = module.networking.private_subnet_ids
  security_group_ids = [module.networking.serving_sg_id]
  execution_role_arn = module.iam.ecs_execution_role_arn
  task_role_arn      = module.iam.ecs_task_role_arn
  ecr_serving_repo   = module.s3.artifacts_bucket_name  # placeholder
  aws_region         = var.aws_region
}

# ─── Outputs ──────────────────────────────────────────────────────────────────
output "raw_bucket_name" {
  value = module.s3.raw_bucket_name
}

output "redshift_endpoint" {
  value     = module.redshift.endpoint
  sensitive = true
}

output "airflow_ec2_private_ip" {
  value = module.ec2_airflow.private_ip
}

output "ecs_cluster_name" {
  value = module.ecs.cluster_name
}

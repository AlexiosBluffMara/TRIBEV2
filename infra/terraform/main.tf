terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment for remote state (S3 + DynamoDB lock)
  # backend "s3" {
  #   bucket         = "mindscope-tf-state"
  #   key            = "prod/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "mindscope-tf-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "MindScope"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC module (for HIPAA-ready Carle deployment)
module "vpc" {
  source = "./modules/vpc"

  environment = var.environment
  cidr_block  = var.vpc_cidr

  # Enable VPC Flow Logs for audit trail
  enable_flow_logs = var.enable_hipaa_mode
}

# Database module
module "database" {
  source = "./modules/database"

  environment = var.environment
  vpc_id      = module.vpc.id

  # For HIPAA: encryption at rest, multi-AZ, automated backups
  enable_encryption = var.enable_hipaa_mode
  multi_az          = var.enable_hipaa_mode
  backup_retention  = var.enable_hipaa_mode ? 90 : 7
}

# S3 module (for HIPAA: use AWS S3 instead of R2)
module "storage" {
  source = "./modules/storage"

  environment = var.environment

  # For HIPAA: enable encryption, block public access
  enable_encryption       = var.enable_hipaa_mode
  block_public_access     = var.enable_hipaa_mode
  enable_versioning       = var.enable_hipaa_mode
  enable_server_side_logs = var.enable_hipaa_mode
}

# Lambda / ECS compute for TRIBE v2 inference (scalable)
module "compute" {
  source = "./modules/compute"

  environment = var.environment
  vpc_id      = module.vpc.id

  # GPU workers via ECS Fargate (or EC2 with GPU)
  gpu_worker_count = var.gpu_worker_count
  gpu_instance_type = "g4dn.xlarge"  # or h100 equivalent

  # For HIPAA: runs in private subnets, no public IP
  private_deployment = var.enable_hipaa_mode
}

output "vpc_id" {
  value = module.vpc.id
}

output "database_endpoint" {
  value = module.database.endpoint
}

output "storage_bucket" {
  value = module.storage.bucket_name
}

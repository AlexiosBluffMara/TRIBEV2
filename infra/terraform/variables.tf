variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_hipaa_mode" {
  description = "Enable HIPAA-compliant configuration (encryption, audit logs, VPC isolation)"
  type        = bool
  default     = false
}

variable "gpu_worker_count" {
  description = "Number of GPU workers to provision"
  type        = number
  default     = 1
}

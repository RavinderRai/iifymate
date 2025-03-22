provider "aws" {
  region = var.aws_region
}

# S3 bucket for ML data including feature store
resource "aws_s3_bucket" "ml_data" {
  bucket = var.bucket_name
  force_destroy = true

  tags = var.tags
}

# Enable versioning but with a lifecycle rule to clean up old versions
resource "aws_s3_bucket_versioning" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption (free)
resource "aws_s3_bucket_server_side_encryption_configuration" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access (free and recommended)
resource "aws_s3_bucket_public_access_block" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Add lifecycle rules to manage storage and versions
resource "aws_s3_bucket_lifecycle_configuration" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id

  rule {
    id     = "cleanup_old_versions"
    status = "Enabled"

    # Delete old versions after 30 days to stay within free tier storage
    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    # Clean up incomplete multipart uploads to avoid unnecessary storage
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Create basic folders (free)
resource "aws_s3_object" "feature_store" {
  bucket = aws_s3_bucket.ml_data.id
  key    = "feature_store/"
  content_type = "application/x-directory"
}

resource "aws_s3_object" "models" {
  bucket = aws_s3_bucket.ml_data.id
  key    = "models/"
  content_type = "application/x-directory"
}

resource "aws_s3_object" "datasets" {
  bucket = aws_s3_bucket.ml_data.id
  key    = "datasets/"
  content_type = "application/x-directory"
}

# Basic IAM policy (IAM is free)
data "aws_iam_policy_document" "ml_data_access" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
      "s3:DeleteObject"
    ]
    resources = [
      aws_s3_bucket.ml_data.arn,
      "${aws_s3_bucket.ml_data.arn}/*"
    ]
  }
}

resource "aws_iam_policy" "ml_data_access" {
  name        = "ml-data-access"
  description = "Policy for accessing ML data bucket"
  policy      = data.aws_iam_policy_document.ml_data_access.json
}

# Add metric alarms to monitor free tier usage
resource "aws_cloudwatch_metric_alarm" "s3_storage_alarm" {
  alarm_name          = "s3-storage-free-tier"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "BucketSizeBytes"
  namespace           = "AWS/S3"
  period             = "86400"  # 1 day
  statistic          = "Average"
  threshold          = 4.5 * 1024 * 1024 * 1024  # Alert at 4.5GB (90% of free tier)
  alarm_description  = "This metric monitors S3 storage to stay within free tier"
  alarm_actions      = []  # Add SNS topic ARN if you want notifications

  dimensions = {
    BucketName = aws_s3_bucket.ml_data.id
    StorageType = "StandardStorage"
  }
}
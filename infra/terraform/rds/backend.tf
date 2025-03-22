terraform {
  backend "s3" {
    bucket         = "iifymate-terraform-state"  # use your actual bucket name
    key            = "rds/terraform.tfstate"  # will be different for each module
    region         = "us-east-1"  # use your region
    dynamodb_table = "iifymate-terraform-locks"
    encrypt        = true
  }
}
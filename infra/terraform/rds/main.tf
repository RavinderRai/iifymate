provider "aws" {
  region = "us-east-1"  # or your preferred region
}

# Get VPC info
data "terraform_remote_state" "vpc" {
  backend = "s3"
  config = {
    bucket = "iifymate-terraform-state"
    key    = "vpc/terraform.tfstate"
    region = "us-east-1"
  }
}

# Create RDS instance
resource "aws_db_instance" "iifymate_db" {
  identifier        = "iifymate-db"
  engine            = "postgres"
  engine_version    = "16.3"
  instance_class    = "db.t3.micro"  # Free tier eligible
  allocated_storage = 20
  

  db_name  = "iifymate"
  username = var.db_username
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.rds.name
  

  skip_final_snapshot = true  # For development; change for production

  # Free tier optimized settings
  backup_retention_period = 0  # Disable automated backups for free tier
  multi_az               = false
  publicly_accessible    = true

  tags = {
    Environment = "development"
    Project     = "iifymate"
  }
}

# Security group for RDS
resource "aws_security_group" "rds" {
  name        = "iifymate-rds-sg"
  description = "Security group for RDS instance"
  vpc_id      = data.terraform_remote_state.vpc.outputs.vpc_id

  # Allow PostgreSQL traffic from anywhere in the VPC
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]  # VPC CIDR
  }

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]  # Replace with your IP
  }

  # Allow outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "iifymate-rds-sg"
  }
}

# Create DB subnet group
resource "aws_db_subnet_group" "rds" {
  name       = "iifymate-subnet-group"
  subnet_ids = data.terraform_remote_state.vpc.outputs.public_subnets
}

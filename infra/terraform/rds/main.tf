provider "aws" {
  region = "us-east-1"  # Change to your preferred region
}

# Create RDS instance WITHOUT VPC
resource "aws_db_instance" "iifymate_db" {
  identifier        = "iifymate-db"
  engine            = "postgres"
  engine_version    = "16.3"
  instance_class    = "db.t3.micro"  # Free tier eligible
  allocated_storage = 20  # Free tier allows up to 20GB

  db_name  = "iifymate"
  username = var.db_username
  password = var.db_password

  publicly_accessible = true  # ✅ Makes the database available over the internet

  skip_final_snapshot = true  # For development; change for production

  # Free tier optimized settings
  backup_retention_period = 0  # Disable automated backups for free tier
  multi_az               = false

  tags = {
    Environment = "development"
    Project     = "iifymate"
  }
}

# Allow access to RDS from EC2 and your local machine
resource "aws_security_group" "rds_sg" {
  name        = "iifymate-rds-sg"
  description = "Security group for RDS instance"

  # Allow EC2 instances (Replace with your EC2 security group if needed)
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # ❗ Allows global access (restrict later if needed)
  }

  # Allow your PC to access the database (For debugging)
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]  # Your IP address
  }

  # Allow RDS to send outbound traffic (Needed for responses)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

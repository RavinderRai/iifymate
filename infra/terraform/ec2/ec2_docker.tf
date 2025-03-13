provider "aws" {
  region = "us-east-1"  # Change to your preferred AWS region
}

resource "aws_security_group" "calorie_app_sg" {
  name        = "calorie_app_sg"
  description = "Allow necessary ports"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["142.163.34.185/32"] # Restrict SSH to your IP
  }

  ingress {
    from_port   = 8000
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Open APIs to all (Restrict later)
  }

  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Prometheus
  }

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Grafana
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "calorie_app" {
  ami           = "ami-04b4f1a9cf54c11d0"
  instance_type = "t2.micro"
  key_name      = "ml-macro-service"
  security_groups = [aws_security_group.calorie_app_sg.name]

  root_block_device {
    volume_size = 30  # Free Tier limit
    volume_type = "gp2"
  }

  user_data = file("../scripts/setup_docker.sh")

  tags = {
    Name = "CalorieApp-Server"
  }
}

output "instance_ip" {
  value = aws_instance.calorie_app.public_ip
}


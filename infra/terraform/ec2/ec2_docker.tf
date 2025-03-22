provider "aws" {
  region = "us-east-1"  # Change to your preferred AWS region
}

resource "aws_security_group" "calorie_app_sg" {
  name        = "calorie_app_sg"
  description = "Allow necessary ports"

  # ✅ Allow SSH from your personal IP
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["142.163.34.185/32"]  # Replace with your IP
  }

  # ✅ Allow all outbound traffic (for internet access)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
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

  #subnet_id     = "subnet-095c8b08b8c531a3e"
  #vpc_security_group_ids = ["sg-04062b8c7da69d55a"]

  # security_groups = [aws_security_group.calorie_app_sg.name]

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

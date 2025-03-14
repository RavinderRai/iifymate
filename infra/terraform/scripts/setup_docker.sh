#!/bin/bash
set -e

# Update system and install dependencies
sudo apt update -y && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git

# Enable and start Docker service
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# Clone your project repository (replace with your GitHub URL)
cd /home/ubuntu
git clone https://github.com/RavinderRai/iifymate.git
cd iifymate

# Start Docker Compose
docker-compose up -d

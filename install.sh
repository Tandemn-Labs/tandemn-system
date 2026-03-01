#!/bin/bash
sudo apt update
sudo apt upgrade -y
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker

## venv
sudo apt install python3.12-venv
python3 -m venv venv
source venv/bin/activate
pip install -r server_requirements.txt
# echo "source venv/bin/activate" >> ~/.bashrc
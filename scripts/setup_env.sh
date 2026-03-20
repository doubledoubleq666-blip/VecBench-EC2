#!/bin/bash
# EC2 environment setup script for vector-db-bench
# Tested on Ubuntu 22.04 LTS (AWS Academy Learner Lab)
set -e

echo "=== [1/5] System packages ==="
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget unzip ca-certificates \
    software-properties-common python3 python3-pip python3-venv \
    build-essential htop jq

echo "=== [2/5] Docker ==="
if ! command -v docker &> /dev/null; then
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in for group changes."
else
    echo "Docker already installed: $(docker --version)"
fi

echo "=== [3/5] Python venv ==="
cd "$(dirname "$0")/.."
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "=== [4/5] Create data directories ==="
mkdir -p data/raw data/processed bench/results

echo "=== [5/5] Verify ==="
docker --version
docker compose version
python3 --version
pip --version
echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Download SIFT1M:  python scripts/download_sift1m.py"
echo "  2. Prepare subsets:  python scripts/prepare_sift_subsets.py"
echo "  3. Start a database: cd docker/chroma && docker compose up -d"
echo "  4. Run smoke test:   python scripts/smoke_test.py"

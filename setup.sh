#!/bin/bash
set -e 

echo "===== Multilingual Ticket Routing Setup ====="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then 
    echo "Creating Python virtual environment..."
    python3 -m venv venv 
fi 

# Activate venv
source venv/bin/activate 

# Upgrade pip
pip install --upgrade pip 

echo "Installing Python dependencies..."

# Install CPU-only PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from requirements.txt
pip install -r requirements.txt

echo "Setting up folders..."
for folder in checkpoints data; do
    if [ ! -d "$folder" ]; then
        mkdir -p "$folder"
        echo "Created folder: $folder"
    fi
done

# Check for GPU
if python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then 
    echo "GPU detected: Pytorch for CUDA." 
else 
    echo "WARNING: No GPU detected. Training will run on CPU." 
fi 

echo "==== Setup Complete ===="
echo "Activate environment with: source venv/bin/activate"
echo "Run preprocessing: python src/preprocessing/preprocess.py --input data/raw --output data/processed"
echo "Run training: torchrun --nproc_per_node=NUM_GPUS src/training/train_ddp.py"

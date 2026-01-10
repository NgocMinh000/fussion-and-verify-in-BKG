#!/bin/bash
# Installation script for FuseLinker environment
# Supports all variants: DistMult, TransE, ComplEx, ConvE + SapBERT

set -e  # Exit on error

echo "======================================================================"
echo "FuseLinker Environment Setup"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Anaconda or Miniconda first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Conda found${NC}"

# Environment name
ENV_NAME="fuselinker"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}⚠ Environment '$ENV_NAME' already exists.${NC}"
    read -p "Do you want to remove and recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Exiting. Please activate the existing environment with: conda activate $ENV_NAME"
        exit 0
    fi
fi

# Create conda environment
echo ""
echo "Creating conda environment: $ENV_NAME"
echo "Python version: 3.10"
conda create -n $ENV_NAME python=3.10 -y

echo ""
echo -e "${GREEN}✓ Conda environment created${NC}"

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Check CUDA availability
echo ""
echo "======================================================================"
echo "Checking CUDA availability..."
echo "======================================================================"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "CUDA Version: $CUDA_VERSION"

    # Determine PyTorch CUDA version
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        TORCH_CUDA="cu124"
        DGL_CUDA="cu124"
        echo "Using CUDA 12.4 for PyTorch and DGL"
    elif [[ "$CUDA_VERSION" == "11.8" ]]; then
        TORCH_CUDA="cu118"
        DGL_CUDA="cu118"
        echo "Using CUDA 11.8 for PyTorch and DGL"
    else
        echo -e "${YELLOW}⚠ CUDA version $CUDA_VERSION detected. Using CUDA 12.4 packages.${NC}"
        TORCH_CUDA="cu124"
        DGL_CUDA="cu124"
    fi
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected. Installing CPU-only version.${NC}"
    TORCH_CUDA="cpu"
    DGL_CUDA="cpu"
fi

# Install PyTorch
echo ""
echo "======================================================================"
echo "Installing PyTorch..."
echo "======================================================================"

if [[ "$TORCH_CUDA" == "cpu" ]]; then
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/${TORCH_CUDA}
fi

echo -e "${GREEN}✓ PyTorch installed${NC}"

# Install DGL
echo ""
echo "======================================================================"
echo "Installing DGL..."
echo "======================================================================"

if [[ "$DGL_CUDA" == "cpu" ]]; then
    pip install dgl -f https://data.dgl.ai/wheels/repo.html
else
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/${DGL_CUDA}/repo.html
fi

echo -e "${GREEN}✓ DGL installed${NC}"

# Install other dependencies
echo ""
echo "======================================================================"
echo "Installing other dependencies..."
echo "======================================================================"

pip install torchdata==0.7.1
pip install numpy pandas scipy scikit-learn tqdm networkx packaging pydantic psutil

echo -e "${GREEN}✓ Core dependencies installed${NC}"

# Install SapBERT dependencies
echo ""
echo "======================================================================"
echo "Installing SapBERT dependencies (Transformers)..."
echo "======================================================================"

pip install transformers tokenizers huggingface-hub

echo -e "${GREEN}✓ SapBERT dependencies installed${NC}"

# Install visualization dependencies
echo ""
echo "======================================================================"
echo "Installing visualization dependencies (optional)..."
echo "======================================================================"

read -p "Do you want to install visualization tools (Plotly, Dash, PyVis)? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    pip install plotly dash dash-bootstrap-components pyvis umap-learn matplotlib seaborn pillow requests
    echo -e "${GREEN}✓ Visualization dependencies installed${NC}"
else
    echo "Skipping visualization dependencies."
fi

# Verify installation
echo ""
echo "======================================================================"
echo "Verifying installation..."
echo "======================================================================"

python -c "
import sys
import torch
import dgl

print('Python version:', sys.version.split()[0])
print('PyTorch version:', torch.__version__)
print('DGL version:', dgl.__version__)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU memory:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
else:
    print('Running on CPU mode')

# Test imports
try:
    from transformers import AutoModel
    print('✓ Transformers available')
except ImportError:
    print('✗ Transformers not available')

try:
    import plotly
    print('✓ Plotly available')
except ImportError:
    print('✗ Plotly not available (optional)')
"

echo ""
echo "======================================================================"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate $ENV_NAME"
echo "2. Test installation: python test_installation.py"
if [[ "$TORCH_CUDA" != "cpu" ]]; then
    echo "3. Run experiments with GPU: python main.py --use_cuda ..."
else
    echo "3. Run experiments: python main.py ..."
fi
echo ""
echo "To generate SapBERT embeddings:"
echo "  python generate_sapbert_embeddings.py --data fuselinker/suppkg"
echo ""
echo "======================================================================"

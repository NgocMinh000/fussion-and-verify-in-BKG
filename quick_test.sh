#!/bin/bash
# Quick test script for FuseLinker environment
# Tests basic functionality without full training

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "======================================================================"
echo -e "${BLUE}FuseLinker Quick Test${NC}"
echo "======================================================================"

# Test 1: Python imports
echo ""
echo -e "${BLUE}Test 1: Checking Python imports...${NC}"

python3 << 'EOF'
import sys

# Test core imports
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import dgl
    print(f"  ✓ DGL {dgl.__version__}")
except ImportError as e:
    print(f"  ✗ DGL import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    print("  ✓ NumPy, Pandas, SciPy, sklearn")
except ImportError as e:
    print(f"  ✗ Data science packages import failed: {e}")
    sys.exit(1)

try:
    from transformers import AutoModel
    print("  ✓ Transformers (for SapBERT)")
except ImportError:
    print("  ⚠ Transformers not available (optional for SapBERT)")

print("\n  All core imports successful!")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Import test failed${NC}"
    exit 1
fi

# Test 2: CUDA availability
echo ""
echo -e "${BLUE}Test 2: Checking CUDA availability...${NC}"

python3 << 'EOF'
import torch

cuda_available = torch.cuda.is_available()

if cuda_available:
    print(f"  ✓ CUDA available")
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    print(f"  ✓ GPU count: {torch.cuda.device_count()}")
    print(f"  ✓ GPU 0: {torch.cuda.get_device_name(0)}")

    # Test GPU computation
    x = torch.randn(100, 100, device='cuda')
    y = x @ x
    print(f"  ✓ GPU computation test passed")
else:
    print("  ⚠ CUDA not available - will use CPU (slower)")
    print("  ℹ For GPU support, ensure:")
    print("    1. NVIDIA drivers installed")
    print("    2. PyTorch installed with CUDA support")
EOF

# Test 3: DGL CUDA
echo ""
echo -e "${BLUE}Test 3: Checking DGL CUDA...${NC}"

python3 << 'EOF'
import dgl
import torch

if torch.cuda.is_available():
    try:
        if dgl.cuda.is_available():
            print("  ✓ DGL CUDA available")
        else:
            print("  ⚠ DGL CUDA not available")
    except:
        print("  ⚠ DGL CUDA check failed")
else:
    print("  ℹ Skipped (CUDA not available)")
EOF

# Test 4: Model import
echo ""
echo -e "${BLUE}Test 4: Checking FuseLinker model imports...${NC}"

if [ -d "fuselinker" ]; then
    cd fuselinker

    python3 << 'EOF'
import sys

try:
    from model import LinkPredict, RGCN, EmbeddingLayer
    print("  ✓ Can import FuseLinker models")

    from data_loader import Data
    print("  ✓ Can import data loader")

    import myutils
    print("  ✓ Can import utilities")

    print("\n  All FuseLinker modules imported successfully!")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)
EOF

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Model import test failed${NC}"
        cd ..
        exit 1
    fi

    cd ..
else
    echo -e "${YELLOW}  ⚠ fuselinker directory not found${NC}"
    echo "    Run this script from repository root"
fi

# Test 5: Data check
echo ""
echo -e "${BLUE}Test 5: Checking data files...${NC}"

if [ -d "fuselinker/suppkg" ]; then
    if [ -f "fuselinker/suppkg/train.tsv" ]; then
        lines=$(wc -l < fuselinker/suppkg/train.tsv)
        echo "  ✓ train.tsv found ($lines triplets)"
    else
        echo -e "  ${YELLOW}⚠ train.tsv not found${NC}"
    fi

    if [ -f "fuselinker/suppkg/valid.tsv" ]; then
        echo "  ✓ valid.tsv found"
    else
        echo -e "  ${YELLOW}⚠ valid.tsv not found${NC}"
    fi

    if [ -f "fuselinker/suppkg/test.tsv" ]; then
        echo "  ✓ test.tsv found"
    else
        echo -e "  ${YELLOW}⚠ test.tsv not found${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠ suppkg directory not found${NC}"
fi

# Test 6: Quick computation test
echo ""
echo -e "${BLUE}Test 6: Running quick computation test...${NC}"

python3 << 'EOF'
import torch
import time

# CPU test
x_cpu = torch.randn(1000, 1000)
start = time.time()
y_cpu = x_cpu @ x_cpu
cpu_time = time.time() - start
print(f"  ✓ CPU computation: {cpu_time:.4f}s")

# GPU test (if available)
if torch.cuda.is_available():
    x_gpu = torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    y_gpu = x_gpu @ x_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    speedup = cpu_time / gpu_time
    print(f"  ✓ GPU computation: {gpu_time:.4f}s")
    print(f"  ✓ GPU speedup: {speedup:.1f}x")

    if speedup < 5:
        print(f"  ⚠ GPU speedup lower than expected (check CUDA setup)")
EOF

# Summary
echo ""
echo "======================================================================"
echo -e "${GREEN}Quick Test Complete!${NC}"
echo "======================================================================"

python3 << 'EOF'
import torch

print("\nEnvironment Summary:")
print(f"  Python: 3.10+")
print(f"  PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"  CUDA: ✓ Available ({torch.version.cuda})")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"  CUDA: ✗ Not available (CPU mode)")

print("\nYou are ready to run experiments!")
EOF

echo ""
echo "Next steps:"
echo ""
echo "  1. Run full installation test:"
echo -e "     ${BLUE}python test_installation.py${NC}"
echo ""
echo "  2. Check GPU configuration:"
echo -e "     ${BLUE}python check_gpu.py${NC}"
echo ""
echo "  3. Run quick training test (100 iterations):"
echo -e "     ${BLUE}cd fuselinker${NC}"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "     ${BLUE}python main.py --data suppkg --use_cuda --iterations 100${NC}"
else
    echo -e "     ${BLUE}python main.py --data suppkg --iterations 100${NC}"
fi

echo ""
echo "======================================================================"

# FuseLinker Environment Setup Guide

## Quick Start (Automated)

```bash
# 1. Run automated installation
bash install_environment.sh

# 2. Activate environment
conda activate fuselinker

# 3. Test installation
python test_installation.py

# 4. Check GPU (if available)
python check_gpu.py
```

## Manual Installation

### Step 1: Create Conda Environment

```bash
conda create -n fuselinker python=3.10 -y
conda activate fuselinker
```

### Step 2: Install PyTorch with CUDA Support

**For CUDA 12.4** (recommended if you have RTX 30xx/40xx, A100, etc.):
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

**For CUDA 11.8** (older GPUs):
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only** (no GPU):
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install DGL

**For CUDA 12.4**:
```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

**For CUDA 11.8**:
```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
```

**For CPU only**:
```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### Step 4: Install Other Dependencies

```bash
# Core dependencies
pip install torchdata==0.7.1 pydantic psutil networkx packaging numpy pandas scipy scikit-learn tqdm

# SapBERT dependencies
pip install transformers tokenizers huggingface-hub

# Visualization (optional)
pip install plotly dash dash-bootstrap-components pyvis umap-learn matplotlib seaborn
```

### Step 5: Verify Installation

```bash
python -c "import torch, dgl; print('✓ PyTorch:', torch.__version__, '| DGL:', dgl.__version__, '| CUDA:', torch.cuda.is_available())"
```

Expected output:
```
✓ PyTorch: 2.4.1+cu124 | DGL: 2.4.0 | CUDA: True
```

## Complete Dependencies List

### Core (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Base interpreter |
| PyTorch | 2.4.1 | Deep learning framework |
| DGL | 2.4.0 | Graph neural networks |
| NumPy | ≥1.21.0 | Numerical computing |
| Pandas | ≥1.3.0 | Data processing |
| SciPy | ≥1.7.0 | Scientific computing |
| scikit-learn | ≥1.0.0 | Machine learning utilities |
| NetworkX | ≥2.6.0 | Graph algorithms |
| tqdm | ≥4.62.0 | Progress bars |

### SapBERT (Required for SapBERT experiments)

| Package | Version | Purpose |
|---------|---------|---------|
| Transformers | ≥4.30.0 | Hugging Face transformers |
| Tokenizers | ≥0.13.0 | Fast tokenization |
| huggingface-hub | ≥0.16.0 | Model hub access |

### Visualization (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| Plotly | ≥5.14.0 | Interactive plots |
| Dash | ≥2.9.0 | Web dashboards |
| PyVis | ≥0.3.1 | Network visualization |
| UMAP | ≥0.5.3 | Dimensionality reduction |
| Matplotlib | ≥3.5.0 | Static plots |
| Seaborn | ≥0.12.0 | Statistical visualization |

## GPU Setup

### Check GPU Availability

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvidia-smi | grep "CUDA Version"

# Run detailed GPU check
python check_gpu.py
```

### GPU Requirements by Method

| Method | Min GPU RAM | Recommended GPU | Training Time (40K iter) |
|--------|-------------|-----------------|--------------------------|
| DistMult | 4 GB | GTX 1660+ | ~2 hours |
| TransE | 4 GB | GTX 1660+ | ~2.5 hours |
| ComplEx | 6 GB | RTX 2060+ | ~3 hours |
| ConvE | 8 GB | RTX 3060+ | ~5 hours |

### Common GPU Issues

#### Issue 1: CUDA not available in PyTorch

**Check**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, check:
1. NVIDIA drivers installed: `nvidia-smi`
2. PyTorch CUDA version matches your CUDA version
3. Reinstall PyTorch with correct CUDA version

**Fix**:
```bash
# Uninstall PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with CUDA support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

#### Issue 2: Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce hidden dimension: `--n_hidden 150`
2. Reduce number of layers: `--num_hidden_layers 1`
3. Clear GPU cache before training:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
4. Monitor GPU memory: `nvidia-smi`

#### Issue 3: DGL CUDA mismatch

**Check**:
```bash
python -c "import dgl; print(dgl.cuda.is_available())"
```

**Fix**:
```bash
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

## Performance Optimization

### For GPU Training

Add these to your training command or modify `main.py`:

```python
import torch

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optional: Enable mixed precision (faster training, less memory)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### GPU Monitoring During Training

```bash
# Real-time monitoring (refresh every 1 second)
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s u

# Log to file
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv -l 1 > gpu_log.csv
```

### Select Specific GPU

If you have multiple GPUs, select one:

```bash
# Use GPU 0 only
export CUDA_VISIBLE_DEVICES=0

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# Or in Python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

## Testing

### Test 1: Basic Imports

```bash
python test_installation.py
```

Expected output:
```
======================================================================
Core Dependencies
======================================================================
  ✓ PyTorch 2.4.1+cu124
  ✓ CUDA available: 12.4
  ✓ GPU count: 1
      GPU 0: NVIDIA GeForce RTX 3090 (24.0 GB)
  ✓ DGL 2.4.0
  ...
✓ All required dependencies installed successfully!
```

### Test 2: GPU Performance

```bash
python check_gpu.py
```

This will:
- Detect GPUs
- Show specifications
- Test PyTorch CUDA
- Provide optimized configurations
- Give example commands

### Test 3: Quick Training Test

Run a quick test with 100 iterations:

```bash
cd fuselinker
python main.py \
    --data suppkg \
    --use_cuda \
    --iterations 100 \
    --evaluate_every 50
```

Should complete in ~1 minute on GPU, ~5 minutes on CPU.

## Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# CUDA paths (if needed)
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# DGL settings
export DGL_DOWNLOAD_DIR=~/.dgl

# Hugging Face cache
export HF_HOME=~/.cache/huggingface
```

## Troubleshooting

### Problem: Import errors

```bash
# Check which packages are installed
pip list | grep -E "torch|dgl|transformers"

# Reinstall missing packages
pip install -r requirements.txt
```

### Problem: Slow training on GPU

**Checks**:
1. Verify GPU is being used:
   ```python
   import torch
   print(torch.cuda.current_device())
   print(torch.cuda.get_device_name(0))
   ```

2. Monitor GPU utilization:
   ```bash
   nvidia-smi
   ```
   Should show >70% GPU utilization during training

3. Enable optimizations in code:
   ```python
   torch.backends.cudnn.benchmark = True
   ```

### Problem: Version conflicts

```bash
# Create fresh environment
conda deactivate
conda env remove -n fuselinker
conda create -n fuselinker python=3.10 -y
conda activate fuselinker

# Run automated install
bash install_environment.sh
```

## Alternative: Docker Setup (Advanced)

If you prefer Docker:

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace
COPY . .

CMD ["bash"]
```

```bash
# Build and run
docker build -t fuselinker .
docker run --gpus all -it -v $(pwd):/workspace fuselinker
```

## Next Steps

After successful setup:

1. **Test with baseline**:
   ```bash
   cd fuselinker
   python main.py --data suppkg --use_cuda --iterations 1000
   ```

2. **Generate SapBERT embeddings**:
   ```bash
   python generate_sapbert_embeddings.py --data fuselinker/suppkg
   ```

3. **Run full experiments**:
   See `EXPERIMENTS_README.md` for complete experiment guide

4. **Visualize results**:
   ```bash
   cd fuselinker/visualization
   python app.py
   ```

## Quick Reference

```bash
# Activate environment
conda activate fuselinker

# Check installation
python test_installation.py

# Check GPU
python check_gpu.py

# Run training (GPU)
cd fuselinker
python main.py --data suppkg --use_cuda

# Generate SapBERT
python generate_sapbert_embeddings.py --data fuselinker/suppkg

# Monitor GPU
watch -n 1 nvidia-smi

# Deactivate environment
conda deactivate
```

## Support

If you encounter issues:

1. Check `test_installation.py` output
2. Check `check_gpu.py` output
3. Review error messages carefully
4. Check CUDA/PyTorch compatibility
5. Try CPU-only version first
6. Check GitHub issues or create new one

## Summary

✅ **Automated setup**: `bash install_environment.sh`
✅ **Manual setup**: Follow steps 1-5 above
✅ **GPU recommended** for ConvE (8GB+ VRAM)
✅ **CPU fallback** available (slower)
✅ **Test scripts** provided for verification
✅ **Full documentation** in each variant's README

Ready to run experiments! 🚀

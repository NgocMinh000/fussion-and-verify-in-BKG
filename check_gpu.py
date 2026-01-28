#!/usr/bin/env python3
"""
Check GPU availability and provide configuration recommendations

This script:
1. Detects available GPUs
2. Shows GPU specifications and memory
3. Recommends optimal batch sizes and configurations
4. Provides commands to run with GPU acceleration

Usage:
    python check_gpu.py
"""

import subprocess
import sys

# ANSI color codes
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
NC = '\033[0m'

def print_header(title):
    print("\n" + "=" * 70)
    print(f"{BLUE}{title}{NC}")
    print("=" * 70)

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return None

def main():
    print_header("GPU Detection and Configuration")

    # Check if nvidia-smi is available
    nvidia_smi = run_command("which nvidia-smi")

    if not nvidia_smi:
        print(f"\n{RED}✗ nvidia-smi not found{NC}")
        print("\nNo NVIDIA GPU detected or NVIDIA drivers not installed.")
        print("\nYou can still run FuseLinker on CPU, but it will be slower:")
        print(f"  {CYAN}python main.py --data suppkg ...{NC}")
        print("\nFor GPU support, install NVIDIA drivers and CUDA toolkit.")
        return 1

    print(f"\n{GREEN}✓ NVIDIA drivers installed{NC}")

    # Get GPU information
    print_header("GPU Information")

    gpu_info = run_command("nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader")

    if gpu_info:
        print("")
        for line in gpu_info.split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                idx, name, driver, total_mem, free_mem = parts
                print(f"GPU {idx}: {GREEN}{name}{NC}")
                print(f"  Driver: {driver}")
                print(f"  Total Memory: {total_mem}")
                print(f"  Free Memory: {free_mem}")
                print()

    # Get CUDA version
    cuda_version = run_command("nvidia-smi | grep 'CUDA Version' | awk '{print $9}'")
    if cuda_version:
        print(f"CUDA Version: {GREEN}{cuda_version}{NC}")

    # Check PyTorch CUDA
    print_header("PyTorch CUDA Status")

    try:
        import torch

        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {GREEN if torch.cuda.is_available() else RED}{torch.cuda.is_available()}{NC}")

        if torch.cuda.is_available():
            print(f"CUDA version (PyTorch): {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"  Multi-Processors: {props.multi_processor_count}")

            # Test GPU
            print(f"\n{CYAN}Testing GPU performance...{NC}")

            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.mm(x, x)

            print(f"{GREEN}✓ GPU is working correctly{NC}")

            # Get current GPU memory usage
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"\nCurrent GPU memory:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")

        else:
            print(f"\n{RED}✗ PyTorch cannot access CUDA{NC}")
            print("\nPossible issues:")
            print("1. PyTorch installed without CUDA support")
            print("2. CUDA version mismatch")
            print("\nReinstall PyTorch with CUDA:")
            if cuda_version and cuda_version.startswith('12'):
                print(f"  {CYAN}pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124{NC}")
            elif cuda_version and cuda_version.startswith('11'):
                print(f"  {CYAN}pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118{NC}")

    except ImportError:
        print(f"\n{RED}✗ PyTorch not installed{NC}")
        print("Install PyTorch first:")
        print(f"  {CYAN}pip install torch torchvision torchaudio{NC}")
        return 1

    # Recommendations
    if torch.cuda.is_available():
        print_header("Recommendations for GPU Training")

        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"\nYour GPU has {gpu_mem:.1f} GB memory")

        if gpu_mem >= 12:
            print(f"{GREEN}✓ Excellent GPU - can run all methods with large models{NC}")
            batch_size = "64-128"
            hidden_dim = "300-400"
        elif gpu_mem >= 8:
            print(f"{GREEN}✓ Good GPU - can run all methods{NC}")
            batch_size = "32-64"
            hidden_dim = "200-300"
        elif gpu_mem >= 4:
            print(f"{YELLOW}⚠ Moderate GPU - can run most methods{NC}")
            batch_size = "16-32"
            hidden_dim = "150-200"
        else:
            print(f"{YELLOW}⚠ Limited GPU - may need to reduce model size{NC}")
            batch_size = "8-16"
            hidden_dim = "100-150"

        print(f"\nRecommended settings:")
        print(f"  Hidden dimension: {hidden_dim}")
        print(f"  Batch size: {batch_size} (if applicable)")

        print_header("Example Commands (GPU-accelerated)")

        print(f"\n{CYAN}1. DistMult (baseline):{NC}")
        print(f"cd fuselinker")
        print(f"python main.py \\")
        print(f"    --data suppkg \\")
        print(f"    --use_cuda \\")
        print(f"    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \\")
        print(f"    --knowledge_embedding_file poincare_embeddings.npy \\")
        print(f"    --num_hidden_layers 2 \\")
        print(f"    --iterations 40000 \\")
        print(f"    --w 0.75")

        print(f"\n{CYAN}2. TransE:{NC}")
        print(f"cd fuselinker-transe")
        print(f"python main.py --data suppkg --use_cuda --w 0.75")

        print(f"\n{CYAN}3. ComplEx:{NC}")
        print(f"cd fuselinker-complex")
        print(f"python main.py --data suppkg --use_cuda --lr 0.005 --w 0.75")

        print(f"\n{CYAN}4. ConvE (best performance, GPU recommended):{NC}")
        print(f"cd fuselinker-conve")
        print(f"python main.py \\")
        print(f"    --data suppkg \\")
        print(f"    --use_cuda \\")
        print(f"    --n_hidden 200 \\")
        print(f"    --lr 0.003 \\")
        print(f"    --dropout 0.3 \\")
        print(f"    --iterations 60000 \\")
        print(f"    --w 0.75")

        print(f"\n{CYAN}5. Generate SapBERT embeddings (GPU-accelerated):{NC}")
        print(f"python generate_sapbert_embeddings.py \\")
        print(f"    --data fuselinker/suppkg \\")
        print(f"    --batch_size 64")

        print_header("GPU Monitoring")

        print(f"\nMonitor GPU usage during training:")
        print(f"  {CYAN}watch -n 1 nvidia-smi{NC}")

        print(f"\nOr in another terminal:")
        print(f"  {CYAN}nvidia-smi dmon -s u{NC}")

        print_header("Performance Tips")

        print(f"""
{GREEN}✓ GPU-specific optimizations:{NC}

1. Use mixed precision training (if supported):
   Add to main.py:
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()

2. Pin memory for DataLoaders (if using):
   DataLoader(..., pin_memory=True)

3. Set CUDA device before training:
   export CUDA_VISIBLE_DEVICES=0

4. Enable cuDNN autotuner:
   torch.backends.cudnn.benchmark = True

5. Monitor GPU utilization:
   nvidia-smi dmon -s u

{YELLOW}⚠ If you get OOM (Out of Memory) errors:{NC}

1. Reduce hidden dimension: --n_hidden 150
2. Reduce batch size (if applicable)
3. Clear GPU cache: torch.cuda.empty_cache()
4. Use gradient checkpointing
        """)

    print("\n" + "=" * 70 + "\n")

    return 0

if __name__ == '__main__':
    sys.exit(main())

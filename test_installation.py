#!/usr/bin/env python3
"""
Test installation of FuseLinker environment

This script verifies that all required dependencies are installed correctly
and checks GPU availability for faster training.

Usage:
    python test_installation.py
"""

import sys
import os

# ANSI color codes
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def print_section(title):
    print("\n" + "=" * 70)
    print(f"{BLUE}{title}{NC}")
    print("=" * 70)

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"  {GREEN}✓{NC} {package_name}")
        return True
    except ImportError as e:
        print(f"  {RED}✗{NC} {package_name} - {str(e)}")
        return False

def main():
    print_section("FuseLinker Environment Test")

    # Test Python version
    print(f"\nPython version: {sys.version.split()[0]}")
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 9:
        print(f"  {GREEN}✓{NC} Python version is compatible (>= 3.9)")
    else:
        print(f"  {YELLOW}⚠{NC} Python 3.9+ recommended, you have {py_version.major}.{py_version.minor}")

    # Test core dependencies
    print_section("Core Dependencies")

    all_ok = True

    # PyTorch
    try:
        import torch
        print(f"  {GREEN}✓{NC} PyTorch {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"  {GREEN}✓{NC} CUDA available: {torch.version.cuda}")
            print(f"  {GREEN}✓{NC} GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
                print(f"      GPU {i}: {gpu_name} ({gpu_memory} GB)")
        else:
            print(f"  {YELLOW}⚠{NC} CUDA not available - will run on CPU (slower)")
            print(f"      To use GPU, reinstall PyTorch with CUDA support")
    except ImportError:
        print(f"  {RED}✗{NC} PyTorch not installed")
        all_ok = False

    # DGL
    try:
        import dgl
        print(f"  {GREEN}✓{NC} DGL {dgl.__version__}")

        # Check DGL CUDA
        try:
            if dgl.cuda.is_available():
                print(f"  {GREEN}✓{NC} DGL CUDA available")
        except:
            pass
    except ImportError:
        print(f"  {RED}✗{NC} DGL not installed")
        all_ok = False

    # Other core packages
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
        ('networkx', 'NetworkX'),
    ]

    for module, name in packages:
        if not test_import(module, name):
            all_ok = False

    # SapBERT dependencies
    print_section("SapBERT Dependencies")

    sapbert_ok = True
    sapbert_packages = [
        ('transformers', 'Transformers'),
        ('tokenizers', 'Tokenizers'),
    ]

    for module, name in sapbert_packages:
        if not test_import(module, name):
            sapbert_ok = False

    if sapbert_ok:
        try:
            from transformers import AutoModel, AutoTokenizer
            print(f"\n  {GREEN}✓{NC} Can load transformer models")
        except:
            print(f"\n  {YELLOW}⚠{NC} Issue loading transformer models")

    # Visualization dependencies
    print_section("Visualization Dependencies (Optional)")

    viz_packages = [
        ('plotly', 'Plotly'),
        ('dash', 'Dash'),
        ('pyvis', 'PyVis'),
        ('umap', 'UMAP'),
        ('matplotlib', 'Matplotlib'),
    ]

    viz_count = 0
    for module, name in viz_packages:
        if test_import(module, name):
            viz_count += 1

    if viz_count == len(viz_packages):
        print(f"\n  {GREEN}✓{NC} All visualization tools available")
    elif viz_count > 0:
        print(f"\n  {YELLOW}⚠{NC} Some visualization tools missing ({viz_count}/{len(viz_packages)} available)")
    else:
        print(f"\n  {YELLOW}⚠{NC} No visualization tools installed (optional)")

    # Test model imports
    print_section("FuseLinker Model Test")

    # Try importing from fuselinker
    fuselinker_ok = True

    if os.path.exists('fuselinker/model.py'):
        sys.path.insert(0, 'fuselinker')
        try:
            from model import LinkPredict, RGCN, EmbeddingLayer
            print(f"  {GREEN}✓{NC} Can import FuseLinker models")
        except Exception as e:
            print(f"  {RED}✗{NC} Error importing FuseLinker models: {e}")
            fuselinker_ok = False
        sys.path.pop(0)
    else:
        print(f"  {YELLOW}⚠{NC} fuselinker/model.py not found in current directory")
        print(f"      Run this script from the repository root")

    # GPU Performance Test
    if 'torch' in sys.modules and torch.cuda.is_available():
        print_section("GPU Performance Test")

        try:
            # Simple matrix multiplication test
            import time

            size = 5000
            device = torch.device('cuda')

            # Warmup
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            _ = torch.mm(a, b)
            torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            gpu_time = time.time() - start

            # CPU benchmark
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            start = time.time()
            c_cpu = torch.mm(a_cpu, b_cpu)
            cpu_time = time.time() - start

            speedup = cpu_time / gpu_time

            print(f"  Matrix multiplication ({size}x{size}):")
            print(f"    CPU time: {cpu_time:.4f}s")
            print(f"    GPU time: {gpu_time:.4f}s")
            print(f"    {GREEN}✓{NC} GPU is {speedup:.1f}x faster than CPU")

            if speedup < 5:
                print(f"    {YELLOW}⚠{NC} GPU speedup lower than expected. Check CUDA installation.")

        except Exception as e:
            print(f"  {RED}✗{NC} GPU test failed: {e}")

    # Summary
    print_section("Summary")

    if all_ok and sapbert_ok and fuselinker_ok:
        print(f"\n{GREEN}✓ All required dependencies installed successfully!{NC}\n")

        print("You are ready to run FuseLinker experiments:")
        print("\n1. Original FuseLinker (DistMult):")
        print("   cd fuselinker")
        if 'torch' in sys.modules and torch.cuda.is_available():
            print("   python main.py --data suppkg --use_cuda ...")
        else:
            print("   python main.py --data suppkg ...")

        print("\n2. TransE variant:")
        print("   cd fuselinker-transe")
        print("   python main.py --data suppkg ...")

        print("\n3. ComplEx variant:")
        print("   cd fuselinker-complex")
        print("   python main.py --data suppkg --lr 0.005 ...")

        print("\n4. ConvE variant (best performance, needs GPU):")
        print("   cd fuselinker-conve")
        if 'torch' in sys.modules and torch.cuda.is_available():
            print("   python main.py --data suppkg --lr 0.003 --iterations 60000 ...")
        else:
            print(f"   {YELLOW}⚠ ConvE is very slow on CPU. GPU highly recommended.{NC}")

        print("\n5. Generate SapBERT embeddings:")
        print("   python generate_sapbert_embeddings.py --data fuselinker/suppkg")

    else:
        print(f"\n{RED}✗ Some dependencies are missing or not working correctly.{NC}\n")

        if not all_ok:
            print("Install core dependencies:")
            print("  pip install torch torchvision torchaudio dgl numpy pandas scipy scikit-learn")

        if not sapbert_ok:
            print("\nInstall SapBERT dependencies:")
            print("  pip install transformers tokenizers")

        print("\nOr run the automated installation script:")
        print("  bash install_environment.sh")

    print("\n" + "=" * 70 + "\n")

    return 0 if (all_ok and sapbert_ok) else 1

if __name__ == '__main__':
    sys.exit(main())

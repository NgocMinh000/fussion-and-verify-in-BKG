# Hướng Dẫn Cài Đặt Môi Trường FuseLinker

## Phương Án 1: Cài Đặt Tự Động (Khuyến Nghị) ⭐

### Bước 1: Clone Repository (nếu chưa có)

```bash
git clone <repository-url>
cd fussion-and-verify-in-BKG
```

### Bước 2: Chạy Script Cài Đặt Tự Động

```bash
bash install_environment.sh
```

Script sẽ tự động:
- ✅ Tạo conda environment với Python 3.10
- ✅ Detect CUDA version của bạn
- ✅ Cài PyTorch với CUDA support phù hợp
- ✅ Cài DGL với CUDA support
- ✅ Cài tất cả dependencies cần thiết
- ✅ Cài Transformers cho SapBERT
- ✅ Hỏi bạn có muốn cài visualization tools không
- ✅ Verify installation tự động

**Thời gian**: ~5-10 phút (tùy vào internet speed)

### Bước 3: Activate Environment

```bash
conda activate fuselinker
```

### Bước 4: Test Installation

```bash
# Quick test (~30 giây)
bash quick_test.sh

# Hoặc full test
python test_installation.py

# Check GPU
python check_gpu.py
```

**Done!** 🎉

---

## Phương Án 2: Cài Đặt Thủ Công

### Bước 1: Tạo Conda Environment

```bash
conda create -n fuselinker python=3.10 -y
conda activate fuselinker
```

### Bước 2: Kiểm Tra CUDA Version

```bash
nvidia-smi
```

Tìm dòng "CUDA Version" để biết version của bạn (ví dụ: 12.4, 11.8).

### Bước 3: Cài PyTorch

**Nếu CUDA 12.x** (RTX 30xx/40xx, A100, etc.):
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

**Nếu CUDA 11.8** (older GPUs):
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

**Nếu không có GPU** (CPU only):
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

### Bước 4: Cài DGL

**Nếu CUDA 12.x**:
```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

**Nếu CUDA 11.8**:
```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
```

**Nếu CPU only**:
```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### Bước 5: Cài Dependencies Khác

```bash
# Core packages
pip install torchdata==0.7.1 pydantic psutil networkx packaging numpy pandas scipy scikit-learn tqdm

# SapBERT packages
pip install transformers tokenizers huggingface-hub

# Visualization (optional)
pip install plotly dash dash-bootstrap-components pyvis umap-learn matplotlib seaborn
```

### Bước 6: Verify Installation

```bash
python -c "import torch, dgl; print('✓ PyTorch:', torch.__version__, '| DGL:', dgl.__version__, '| CUDA:', torch.cuda.is_available())"
```

**Expected output**:
```
✓ PyTorch: 2.4.1+cu124 | DGL: 2.4.0 | CUDA: True
```

### Bước 7: Test Full Installation

```bash
python test_installation.py
python check_gpu.py
```

---

## Phương Án 3: Từ Commands Bạn Đã Dùng

Bạn nói bạn đã từng cài như sau và chạy được:

```bash
conda create -n fuselinker python=3.10 -y
conda activate fuselinker

# Cài packages
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install torchdata==0.7.1 pydantic psutil networkx packaging numpy pandas scipy scikit-learn tqdm

# Verify
python -c "import torch, dgl; print('✓ PyTorch:', torch.__version__, '| DGL:', dgl.__version__, '| CUDA:', torch.cuda.is_available())"
```

**Bổ sung thêm** để chạy các experiment mới:

```bash
# Thêm SapBERT support
pip install transformers tokenizers huggingface-hub

# Thêm visualization (optional)
pip install plotly dash dash-bootstrap-components pyvis umap-learn matplotlib seaborn
```

**Test**:
```bash
python test_installation.py
```

---

## Các Dependencies Cần Thiết

### Bắt Buộc (Core)

| Package | Version | Mục đích |
|---------|---------|----------|
| Python | 3.10 | Base |
| PyTorch | 2.4.1 | Deep learning |
| DGL | 2.4.0 | Graph neural networks |
| NumPy | ≥1.21 | Numerical computing |
| Pandas | ≥1.3 | Data processing |
| SciPy | ≥1.7 | Scientific computing |
| scikit-learn | ≥1.0 | ML utilities |

### Cho SapBERT (Khuyến nghị)

| Package | Version | Mục đích |
|---------|---------|----------|
| Transformers | ≥4.30 | Hugging Face models |
| Tokenizers | ≥0.13 | Fast tokenization |

### Cho Visualization (Optional)

| Package | Version | Mục đích |
|---------|---------|----------|
| Plotly | ≥5.14 | Interactive plots |
| Dash | ≥2.9 | Web dashboards |
| PyVis | ≥0.3.1 | Network graphs |

---

## Hướng Dẫn Sử Dụng GPU

### Check GPU của Bạn

```bash
# Xem GPU specs
nvidia-smi

# Check CUDA version
nvidia-smi | grep "CUDA Version"

# Run GPU check script
python check_gpu.py
```

### Yêu Cầu GPU cho Các Phương Pháp

| Phương pháp | GPU RAM tối thiểu | GPU khuyến nghị | Thời gian train |
|-------------|-------------------|-----------------|-----------------|
| DistMult | 4 GB | GTX 1660+ | ~2 giờ |
| TransE | 4 GB | GTX 1660+ | ~2.5 giờ |
| ComplEx | 6 GB | RTX 2060+ | ~3 giờ |
| ConvE | 8 GB | RTX 3060+ | ~5 giờ |

### Chạy với GPU

Thêm flag `--use_cuda` vào lệnh training:

```bash
python main.py --data suppkg --use_cuda ...
```

### Monitor GPU Trong Khi Training

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Hoặc
nvidia-smi dmon -s u
```

### Nếu Out of Memory (OOM)

```bash
# Giảm hidden dimension
--n_hidden 150  # thay vì 200

# Giảm số layers
--num_hidden_layers 1  # thay vì 2
```

---

## Troubleshooting

### Vấn Đề 1: CUDA not available

**Kiểm tra**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu `False`:

1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch với CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
   ```

### Vấn Đề 2: DGL import error

```bash
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

### Vấn Đề 3: Transformers not found

```bash
pip install transformers tokenizers huggingface-hub
```

### Vấn Đề 4: Version conflicts

Tạo lại environment từ đầu:
```bash
conda deactivate
conda env remove -n fuselinker
bash install_environment.sh
```

---

## Test Nhanh

Sau khi cài đặt xong, test ngay:

```bash
# Test 1: Quick test
bash quick_test.sh

# Test 2: Full installation test
python test_installation.py

# Test 3: GPU configuration
python check_gpu.py

# Test 4: Mini training run (100 iterations)
cd fuselinker
python main.py --data suppkg --use_cuda --iterations 100 --evaluate_every 50
```

**Test 4 sẽ chạy ~1 phút trên GPU**, ~5 phút trên CPU.

Nếu chạy được → environment setup thành công! ✅

---

## Next Steps Sau Khi Setup

### 1. Generate SapBERT Embeddings

```bash
python generate_sapbert_embeddings.py --data fuselinker/suppkg --batch_size 32
```

**Thời gian**: ~5-10 phút CPU, ~1-2 phút GPU

### 2. Chạy Baseline (DistMult)

```bash
cd fuselinker
python main.py \
    --data suppkg \
    --use_cuda \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --w 0.75 \
    --iterations 40000
```

### 3. Thử TransE

```bash
cd fuselinker-transe
python main.py --data suppkg --use_cuda --w 0.75
```

### 4. Thử ComplEx

```bash
cd fuselinker-complex
python main.py --data suppkg --use_cuda --lr 0.005 --w 0.75
```

### 5. Thử ConvE (Best Performance)

```bash
cd fuselinker-conve
python main.py \
    --data suppkg \
    --use_cuda \
    --n_hidden 200 \
    --lr 0.003 \
    --dropout 0.3 \
    --iterations 60000 \
    --w 0.75
```

---

## Summary

**Khuyến nghị**: Dùng **Phương Án 1** (Automated):

```bash
# 1. Run install script
bash install_environment.sh

# 2. Activate
conda activate fuselinker

# 3. Test
python test_installation.py
python check_gpu.py

# 4. Start experiments!
cd fuselinker
python main.py --data suppkg --use_cuda ...
```

**Hoặc** dùng commands bạn đã biết (Phương Án 3) + thêm SapBERT packages.

Nếu gặp vấn đề, check:
1. `quick_test.sh` output
2. `test_installation.py` output
3. `check_gpu.py` output
4. Error messages

Good luck! 🚀

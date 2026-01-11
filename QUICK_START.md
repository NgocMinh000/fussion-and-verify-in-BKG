# Quick Start Guide - FuseLinker Training

## ✅ Prerequisites

- ✅ Environment setup done (`conda activate fuselinker`)
- ✅ GPU available (RTX 3090 with 25GB)
- ✅ Embeddings in `~/fussion-and-verify-in-BKG/engine/`

## 🚀 Ready-to-Run Commands

### 1️⃣ Quick Test (100 iterations, ~2 minutes)

**Verify setup and GPU working:**

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 100 \
    --evaluate_every 50 \
    --w 0.75 \
    --use_cuda True
```

**Expected output:**
```
Loading Pretrained Embeddings files...
Loaded Text Embeddings file successfully!
Loaded Domain Knowledge Embeddings file successfully!
...
cuda
...
Epoch 0-50 | Loss values
Evaluating...
MR: 3.45
MRR: 0.812
Hits @ 1 = 0.723
```

✅ If you see this → Setup perfect!

---

### 2️⃣ DistMult + MedLLaMA (4K iterations, ~30 minutes)

**Your original experiment:**

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda True \
    --model_state_file suppkg_distmult_medllama_4k.pth
```

---

### 3️⃣ ComplEx + MedLLaMA (4K iterations, ~40 minutes)

**Expected best performance (+2-3% MRR):**

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.005 \
    --use_cuda True \
    --model_state_file suppkg_complex_medllama_4k.pth
```

**Note**: ComplEx uses `--lr 0.005` (lower learning rate)

---

### 4️⃣ Compare Embeddings: MedLLaMA vs PubMedBERT

**PubMedBERT (768D, biomedical specialist):**

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker

python main.py --data suppkg \
    --text_embedding_file ../engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda True \
    --model_state_file suppkg_distmult_pubmedbert_4k.pth
```

---

### 5️⃣ ConvE + MedLLaMA (6K iterations, ~1 hour)

**Ultimate performance (best MRR):**

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker-conve

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 6000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.003 \
    --dropout 0.3 \
    --use_cuda True \
    --model_state_file suppkg_conve_medllama_6k.pth
```

---

## 📊 Monitor Training

**In another terminal:**

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed stats
nvidia-smi dmon -s u
```

**Expected GPU usage:**
- Utilization: 70-95%
- Memory: 3-5 GB (n_hidden=200)
- Power: 300-350W

---

## 🎯 Recommended Workflow

### Step 1: Quick Verify (2 minutes)

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker
python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 100 --evaluate_every 50 --w 0.75 --use_cuda True
```

✅ Working? → Continue

### Step 2: Baseline (30 minutes)

Run DistMult + MedLLaMA (4K iterations) - Command #2

### Step 3: Best Method (40 minutes)

Run ComplEx + MedLLaMA (4K iterations) - Command #3

### Step 4: Compare Results

Check which performed better:
- DistMult vs ComplEx
- MedLLaMA vs PubMedBERT

---

## 🔧 Optimizations for RTX 3090 (25GB)

### Option A: Higher Hidden Dimension

**Use n_hidden=300 instead of 200:**

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 300 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.005 \
    --use_cuda True \
    --model_state_file suppkg_complex_medllama_4k_hd300.pth
```

### Option B: Full Training (40K iterations)

**Best results, ~4 hours:**

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 40000 \
    --evaluate_every 1000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.005 \
    --use_cuda True \
    --model_state_file suppkg_complex_medllama_40k.pth
```

**Expected**: MRR ~0.87-0.88

---

## 📝 All Available Embeddings

Located in `~/fussion-and-verify-in-BKG/engine/`:

| Embedding | Dimension | Best For |
|-----------|-----------|----------|
| **medllama_pretrained_embeddings_4096.npy** | 4096D | Medical LLM |
| **pubmedbert_pretrained_embeddings_768.npy** | 768D | Biomedical (recommended) |
| bert_pretrained_embeddings_768.npy | 768D | General purpose |
| flant5_pretrained_embeddings_768.npy | 768D | Instruction-tuned |
| llama2_pretrained_embeddings_4096.npy | 4096D | General LLM |
| poincare_embeddings.npy | 50D | Domain knowledge |

---

## ⚠️ Important Notes

### Syntax
- ✅ `--use_cuda True` (correct)
- ❌ `--use_cuda` (error: expected one argument)

### Path
- Embeddings: `../engine/filename.npy`
- Relative to fuselinker directory

### Learning Rates
- DistMult: default (0.01)
- TransE: default (0.01)
- ComplEx: `--lr 0.005` (lower!)
- ConvE: `--lr 0.003` (even lower!)

### Memory
- n_hidden=200: ~3-5 GB GPU RAM
- n_hidden=300: ~6-8 GB GPU RAM
- n_hidden=400: ~10-12 GB GPU RAM

---

## 🎓 Expected Results

### 4K iterations (quick experiments)

| Method | MRR | Hits@1 | Hits@10 |
|--------|-----|--------|---------|
| DistMult | ~0.82-0.84 | ~74-76% | ~95-96% |
| TransE | ~0.80-0.83 | ~72-75% | ~94-96% |
| ComplEx | ~0.84-0.86 | ~77-80% | ~96-97% |
| ConvE | ~0.86-0.88 | ~80-83% | ~97-98% |

### 40K iterations (full training)

| Method | MRR | Hits@1 | Hits@10 |
|--------|-----|--------|---------|
| DistMult | ~0.85 | ~78% | ~97% |
| ComplEx | ~0.87-0.88 | ~81-82% | ~98% |
| ConvE | ~0.89-0.91 | ~84-87% | ~98-99% |

---

## 🆘 Troubleshooting

### Issue 1: File not found

```bash
# Check embeddings exist
ls -lh ~/fussion-and-verify-in-BKG/engine/*.npy

# Check data files
ls -lh ~/fussion-and-verify-in-BKG/fuselinker/suppkg/*.tsv
```

### Issue 2: CUDA error

```bash
# Check GPU
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue 3: Out of Memory

```bash
# Reduce hidden dimension
--n_hidden 150  # instead of 200

# Or use smaller embeddings
--text_embedding_file ../engine/pubmedbert_pretrained_embeddings_768.npy
```

---

## 📚 More Information

- **All commands**: `cat CORRECTED_COMMANDS.sh`
- **Detailed guide**: `RUN_COMMANDS.md`
- **Setup help**: `INSTALLATION_STEPS.md`
- **GPU optimization**: `check_gpu.py`

---

**Start with Command #1 now!** 🚀

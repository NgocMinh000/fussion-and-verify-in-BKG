# Hướng Dẫn Chạy Experiments với Embeddings từ engine/

## Cấu Trúc Thư Mục

Dựa vào ảnh bạn cung cấp, embeddings nằm trong thư mục `engine/`:

```
engine/
├── bert_pretrained_embeddings_768.npy
├── flant5_pretrained_embeddings_768.npy
├── llama2_pretrained_embeddings_4096.npy
├── medllama_pretrained_embeddings_4096.npy
├── poincare_embeddings.npy
└── pubmedbert_pretrained_embeddings_768.npy
```

## Path Setup

### Option 1: Engine ở Parent Directory

```bash
# Nếu cấu trúc như sau:
# ~/
#   ├── engine/
#   └── fussion-and-verify-in-BKG/

# Thì dùng relative path:
ENGINE_DIR="../engine"
```

### Option 2: Absolute Path

```bash
# Nếu engine ở vị trí cụ thể:
ENGINE_DIR="/home/user/FuseLinker/engine"

# Hoặc:
ENGINE_DIR="$HOME/engine"
```

### Check Engine Directory

```bash
# Verify engine directory exists và có files
ls -la ../engine/

# Hoặc
ls -la /path/to/engine/
```

---

## Commands Dựa Trên Setup Cũ Của Bạn

### Command Cũ (Reference)

```bash
cd ~/FuseLinker/fuselinker
python main.py \
    --data suppkg \
    --text_embedding_file medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --model_state_file suppkg_model_state.pth
```

---

## 1. DistMult (Baseline) - Giống Command Cũ

### Với MedLLaMA (4096D) - Như Bạn Đã Dùng

```bash
cd fuselinker

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_distmult_medllama_4k.pth
```

**Thay đổi so với command cũ**:
- ✅ Added `../engine/` prefix to embedding paths
- ✅ Added `--n_hidden 200` (compress 4096→200)
- ✅ Added `--use_cuda` for GPU acceleration
- ✅ Added `--validate_every 500` for validation checks
- ✅ Better model filename

### Full Run (40K iterations - Better Results)

```bash
cd fuselinker

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 40000 \
    --evaluate_every 1000 \
    --validate_every 2000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_distmult_medllama_40k.pth
```

**Expected**: MRR ~0.85, Hits@1 ~78% (như baseline report)

---

## 2. Thử Các Text Embeddings Khác

### PubMedBERT (768D) - Khuyến Nghị cho Biomedical

```bash
cd fuselinker

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_distmult_pubmedbert.pth
```

**Ưu điểm**: PubMedBERT trained trên PubMed abstracts, tốt cho biomedical entities

### BERT (768D) - General Purpose

```bash
cd fuselinker

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/bert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_distmult_bert.pth
```

### FlanT5 (768D)

```bash
cd fuselinker

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/flant5_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_distmult_flant5.pth
```

### Llama2 (4096D)

```bash
cd fuselinker

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/llama2_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_distmult_llama2.pth
```

---

## 3. TransE với MedLLaMA

```bash
cd fuselinker-transe

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_transe_medllama.pth
```

**Expected**: MRR ~0.83-0.84 (có thể hơi thấp hơn DistMult)

---

## 4. ComplEx với MedLLaMA (Recommended)

```bash
cd fuselinker-complex

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.005 \
    --use_cuda \
    --model_state_file suppkg_complex_medllama.pth
```

**Important**: ComplEx cần `--lr 0.005` (thấp hơn default)

**Expected**: MRR ~0.87-0.88 (+2-3% over DistMult)

---

## 5. ConvE với MedLLaMA (Best Performance)

```bash
cd fuselinker-conve

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 6000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.003 \
    --dropout 0.3 \
    --use_cuda \
    --model_state_file suppkg_conve_medllama.pth
```

**Important**:
- ConvE cần `--lr 0.003` (thấp hơn)
- `--dropout 0.3` (cao hơn để tránh overfit)
- `--n_hidden 200` phải là `height × width` (10×20)
- Chạy lâu hơn, nên dùng 60K iterations cho full training

**Expected**: MRR ~0.89-0.91 (+4-6% over DistMult)

**Full training**:
```bash
cd fuselinker-conve

python main.py \
    --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 60000 \
    --evaluate_every 1000 \
    --validate_every 2000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.003 \
    --dropout 0.3 \
    --use_cuda \
    --model_state_file suppkg_conve_medllama_60k.pth
```

---

## So Sánh Embeddings

### Quick Comparison Experiment

Chạy tất cả embeddings với DistMult (4000 iterations) để compare:

```bash
# 1. MedLLaMA (4096D)
cd fuselinker && python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 4000 --w 0.75 --use_cuda \
    --model_state_file medllama_test.pth

# 2. PubMedBERT (768D)
cd fuselinker && python main.py --data suppkg \
    --text_embedding_file ../engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 4000 --w 0.75 --use_cuda \
    --model_state_file pubmedbert_test.pth

# 3. BERT (768D)
cd fuselinker && python main.py --data suppkg \
    --text_embedding_file ../engine/bert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 4000 --w 0.75 --use_cuda \
    --model_state_file bert_test.pth

# 4. FlanT5 (768D)
cd fuselinker && python main.py --data suppkg \
    --text_embedding_file ../engine/flant5_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 4000 --w 0.75 --use_cuda \
    --model_state_file flant5_test.pth

# 5. Llama2 (4096D)
cd fuselinker && python main.py --data suppkg \
    --text_embedding_file ../engine/llama2_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 4000 --w 0.75 --use_cuda \
    --model_state_file llama2_test.pth
```

### Expected Ranking (Biomedical KG)

1. **PubMedBERT** - Best for biomedical (trained on PubMed)
2. **MedLLaMA** - Good, medical-specific LLM
3. **Llama2** - Strong general LLM
4. **FlanT5** - Good instruction-following
5. **BERT** - General purpose baseline

---

## Troubleshooting

### Issue 1: File Not Found

```bash
# Error: FileNotFoundError: ../engine/medllama_pretrained_embeddings_4096.npy

# Fix: Check engine directory path
ls ../engine/

# If not found, update path:
--text_embedding_file /absolute/path/to/engine/medllama_pretrained_embeddings_4096.npy
```

### Issue 2: Different Working Directory

```bash
# If you're in different directory:
cd /home/user/fussion-and-verify-in-BKG/fuselinker

# Use absolute path:
ENGINE="/home/user/engine"
python main.py --text_embedding_file $ENGINE/medllama_pretrained_embeddings_4096.npy ...
```

### Issue 3: Out of Memory (4096D embeddings)

```bash
# Reduce hidden dimension:
--n_hidden 150  # instead of 200

# Or use 768D embeddings (smaller):
--text_embedding_file ../engine/pubmedbert_pretrained_embeddings_768.npy
```

---

## Recommended Workflow

### Step 1: Quick Test (4K iterations, ~30 mins)

```bash
cd fuselinker
python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 4000 --w 0.75 --use_cuda
```

### Step 2: Full DistMult Baseline (40K iterations, ~3 hours)

```bash
cd fuselinker
python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 40000 --w 0.75 --use_cuda \
    --model_state_file baseline_medllama_40k.pth
```

### Step 3: Try Best Method (ComplEx, 40K iterations)

```bash
cd fuselinker-complex
python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 40000 --lr 0.005 --w 0.75 --use_cuda \
    --model_state_file complex_medllama_40k.pth
```

### Step 4: Compare Embeddings

Try PubMedBERT (768D) vs MedLLaMA (4096D):

```bash
# PubMedBERT
cd fuselinker-complex
python main.py --data suppkg \
    --text_embedding_file ../engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 40000 --lr 0.005 --w 0.75 --use_cuda \
    --model_state_file complex_pubmedbert_40k.pth
```

### Step 5: Ultimate Performance (ConvE, 60K iterations)

```bash
cd fuselinker-conve
python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --iterations 60000 --lr 0.003 --dropout 0.3 --w 0.75 --use_cuda \
    --model_state_file conve_medllama_60k.pth
```

---

## Monitor Training

### During Training

```bash
# In another terminal:
watch -n 1 nvidia-smi

# Or:
nvidia-smi dmon -s u
```

### Check Progress

Training will print:
```
Epoch 0-1000 | Loss: 0.2531
Evaluating...
MR: 3.45
MRR: 0.812
Hits @ 1 = 0.723
Hits @ 3 = 0.891
Hits @ 10 = 0.956
```

---

## Summary

**Khuyến nghị**:

1. **Quick test**: DistMult + MedLLaMA, 4K iterations (~30 mins)
2. **Baseline**: DistMult + PubMedBERT, 40K iterations (~3 hours)
3. **Best method**: ComplEx + MedLLaMA, 40K iterations (~4 hours)
4. **Ultimate**: ConvE + MedLLaMA, 60K iterations (~6 hours)

**Path format**:
```bash
--text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy
--knowledge_embedding_file ../engine/poincare_embeddings.npy
```

Good luck! 🚀

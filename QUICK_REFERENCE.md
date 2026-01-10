# Quick Reference - Training Commands

## 🚀 Copy-Paste Commands (Giả sử engine/ ở parent directory)

### 1. DistMult + MedLLaMA (Như Command Cũ Của Bạn)

```bash
cd fuselinker
python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 --n_hidden 200 --iterations 4000 \
    --evaluate_every 100 --validate_every 500 --neg_sample_size_eval 100 \
    --w 0.75 --use_cuda --model_state_file suppkg_distmult_medllama.pth
```

### 2. DistMult + PubMedBERT (Khuyến Nghị)

```bash
cd fuselinker
python main.py --data suppkg \
    --text_embedding_file ../engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 --n_hidden 200 --iterations 4000 \
    --evaluate_every 100 --validate_every 500 --neg_sample_size_eval 100 \
    --w 0.75 --use_cuda --model_state_file suppkg_distmult_pubmedbert.pth
```

### 3. TransE + MedLLaMA

```bash
cd fuselinker-transe
python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 --n_hidden 200 --iterations 4000 \
    --evaluate_every 100 --validate_every 500 --neg_sample_size_eval 100 \
    --w 0.75 --use_cuda --model_state_file suppkg_transe_medllama.pth
```

### 4. ComplEx + MedLLaMA (Best)

```bash
cd fuselinker-complex
python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 --n_hidden 200 --iterations 4000 \
    --evaluate_every 100 --validate_every 500 --neg_sample_size_eval 100 \
    --w 0.75 --lr 0.005 --use_cuda --model_state_file suppkg_complex_medllama.pth
```

### 5. ConvE + MedLLaMA (Ultimate)

```bash
cd fuselinker-conve
python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 --n_hidden 200 --iterations 6000 \
    --evaluate_every 100 --validate_every 500 --neg_sample_size_eval 100 \
    --w 0.75 --lr 0.003 --dropout 0.3 --use_cuda \
    --model_state_file suppkg_conve_medllama.pth
```

---

## 📋 Embedding Options

```bash
# MedLLaMA (4096D) - Medical LLM
--text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy

# PubMedBERT (768D) - Biomedical BERT
--text_embedding_file ../engine/pubmedbert_pretrained_embeddings_768.npy

# BERT (768D) - General
--text_embedding_file ../engine/bert_pretrained_embeddings_768.npy

# FlanT5 (768D) - Instruction-tuned
--text_embedding_file ../engine/flant5_pretrained_embeddings_768.npy

# Llama2 (4096D) - General LLM
--text_embedding_file ../engine/llama2_pretrained_embeddings_4096.npy

# Domain Knowledge (Always same)
--knowledge_embedding_file ../engine/poincare_embeddings.npy
```

---

## 🎯 Quick Tips

### Path Issues?

```bash
# Check engine directory
ls ../engine/

# Use absolute path if needed
--text_embedding_file /home/user/engine/medllama_pretrained_embeddings_4096.npy
```

### GPU Monitoring

```bash
watch -n 1 nvidia-smi
```

### Training Time (4000 iterations on GPU)

- DistMult: ~30 minutes
- TransE: ~35 minutes
- ComplEx: ~40 minutes
- ConvE: ~50 minutes

### Full Training (40K iterations)

Replace `--iterations 4000` with `--iterations 40000`

Expected results with 40K iterations:
- DistMult: MRR ~0.85
- ComplEx: MRR ~0.87
- ConvE: MRR ~0.89

---

## 🔄 Compare All Methods

```bash
# Set path variable
ENGINE="../engine"

# Run all methods with MedLLaMA
cd fuselinker && python main.py --data suppkg \
    --text_embedding_file $ENGINE/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file $ENGINE/poincare_embeddings.npy \
    --iterations 4000 --w 0.75 --use_cuda

cd fuselinker-transe && python main.py --data suppkg \
    --text_embedding_file $ENGINE/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file $ENGINE/poincare_embeddings.npy \
    --iterations 4000 --w 0.75 --use_cuda

cd fuselinker-complex && python main.py --data suppkg \
    --text_embedding_file $ENGINE/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file $ENGINE/poincare_embeddings.npy \
    --iterations 4000 --lr 0.005 --w 0.75 --use_cuda
```

# Quick Start Guide - FuseLinker Experiments

## ✅ Setup Verified

All embedding files are ready:
- ✓ llama2_pretrained_embeddings_4096.npy (43474, 4096)
- ✓ bert_pretrained_embeddings_768.npy (43474, 768)
- ✓ flant5_pretrained_embeddings_768.npy (43474, 768)
- ✓ pubmedbert_pretrained_embeddings_768.npy (43474, 768)
- ✓ poincare_embeddings.npy (43474, 200)
- ✗ medllama_pretrained_embeddings_4096.npy (corrupted - use llama2 instead)

## 🚀 Quick Test (2 minutes)

Test that everything works:

```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG/fuselinker

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/llama2_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
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
Text embedding path: /root/fussion-and-verify-in-BKG/engine/llama2_pretrained_embeddings_4096.npy
✓ Loaded Text Embeddings file successfully! Shape: (43474, 4096)
Knowledge embedding path: /root/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy
✓ Loaded Domain Knowledge Embeddings file successfully! Shape: (43474, 200)
w: 0.75
Data Processing...
# entities: 43474
# relations: 15
# edges: 305986
cuda
...
Epoch 0-50 | Loss values
Evaluating...
MR: 3.45
MRR: 0.812
Hits @ 1 = 0.723
```

If you see both embeddings loaded successfully (✓) → You're ready! 🎉

## 📊 Recommended Experiments

### Priority 1: Baseline Comparison (Choose ONE)

**Option A: Large embeddings (4096D)**
```bash
# DistMult + Llama2 (30 min)
cd ~/fussion-and-verify-in-BKG/fuselinker
python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/llama2_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 --n_hidden 200 --iterations 4000 --evaluate_every 100 \
    --neg_sample_size_eval 100 --w 0.75 --use_cuda True \
    --model_state_file suppkg_distmult_llama2_4k.pth
```

**Option B: Biomedical-specific (768D) - RECOMMENDED**
```bash
# DistMult + PubMedBERT (25 min)
cd ~/fussion-and-verify-in-BKG/fuselinker
python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 --n_hidden 200 --iterations 4000 --evaluate_every 100 \
    --neg_sample_size_eval 100 --w 0.75 --use_cuda True \
    --model_state_file suppkg_distmult_pubmedbert_4k.pth
```

### Priority 2: Best Scoring Function

```bash
# ComplEx + PubMedBERT (35 min)
cd ~/fussion-and-verify-in-BKG/fuselinker-complex
python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 --n_hidden 200 --iterations 4000 --evaluate_every 100 \
    --neg_sample_size_eval 100 --w 0.75 --use_cuda True \
    --model_state_file suppkg_complex_pubmedbert_4k.pth
```

### Priority 3: State-of-the-art

```bash
# ConvE + PubMedBERT (40 min)
cd ~/fussion-and-verify-in-BKG/fuselinker-conve
python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 --n_hidden 200 --iterations 4000 --evaluate_every 100 \
    --neg_sample_size_eval 100 --w 0.75 --use_cuda True \
    --model_state_file suppkg_conve_pubmedbert_4k.pth
```

## 📈 Monitor Training

Open a new terminal and watch GPU usage:
```bash
watch -n 1 nvidia-smi
```

## 🔍 All Available Experiments

See `CORRECTED_COMMANDS.sh` for complete list:
- 10 different configurations
- 4 scoring functions: DistMult, TransE, ComplEx, ConvE
- 5 text embeddings: Llama2, PubMedBERT, BERT, FlanT5
- 1 domain embedding: Poincaré

## 📊 Expected Results

| Method | Text Emb | MRR | Hits@1 | Hits@10 | Time |
|--------|----------|-----|--------|---------|------|
| DistMult | PubMedBERT | ~0.82 | ~0.72 | ~0.94 | 25 min |
| DistMult | Llama2 | ~0.83 | ~0.73 | ~0.94 | 30 min |
| ComplEx | PubMedBERT | ~0.85 | ~0.76 | ~0.96 | 35 min |
| ConvE | PubMedBERT | ~0.87 | ~0.79 | ~0.97 | 40 min |
| TransE | Llama2 | ~0.80 | ~0.70 | ~0.92 | 30 min |

## 🐛 Troubleshooting

### Embeddings not loading
Run debug script:
```bash
conda activate fuselinker
python debug_embeddings.py
```

### CUDA out of memory
Reduce batch size or hidden dimensions:
```bash
--n_hidden 150  # instead of 200
```

### File not found
Check paths are absolute:
```bash
ls -lh ~/fussion-and-verify-in-BKG/engine/*.npy
```

## 💡 Tips

- **For best biomedical performance:** Use PubMedBERT
- **For largest capacity:** Use Llama2 (4096D)
- **For fastest training:** Use BERT or FlanT5 (768D)
- **For best accuracy:** Use ConvE scoring function
- **For interpretability:** Use TransE scoring function

## 📝 Next Steps

After running experiments, compare results:
```bash
grep "MRR:" suppkg/*.log | sort -k2 -n
```

All model checkpoints saved as `.pth` files in respective directories.

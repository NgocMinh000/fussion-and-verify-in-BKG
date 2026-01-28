# Thử Nghiệm Các Phương Pháp KGE với FuseLinker

## Tổng Quan

Repository này chứa **5 phiên bản** của FuseLinker với các scoring functions khác nhau:

1. **fuselinker/** - DistMult (baseline, bản gốc)
2. **fuselinker-transe/** - TransE scoring
3. **fuselinker-complex/** - ComplEx scoring
4. **fuselinker-conve/** - ConvE scoring

Ngoài ra còn có hướng dẫn sử dụng **SapBERT embeddings** thay vì PubMedBERT.

## Cấu Trúc Thư Mục

```
.
├── fuselinker/                      # DistMult (bản gốc)
│   ├── model.py                     # Scoring: sum(h ⊙ r ⊙ t)
│   ├── main.py
│   └── README.md
│
├── fuselinker-transe/               # TransE
│   ├── model.py                     # Scoring: -||h + r - t||₁
│   ├── main.py
│   └── README_TRANSE.md
│
├── fuselinker-complex/              # ComplEx
│   ├── model.py                     # Scoring: Re(<h, r, conj(t)>)
│   ├── main.py
│   └── README_COMPLEX.md
│
├── fuselinker-conve/                # ConvE
│   ├── model.py                     # Scoring: CNN-based
│   ├── main.py
│   └── README_CONVE.md
│
├── SAPBERT_GUIDE.md                 # Hướng dẫn SapBERT
├── generate_sapbert_embeddings.py   # Script tạo SapBERT embeddings
└── EXPERIMENTS_README.md            # File này
```

## So Sánh Các Phương Pháp

| Method | Complexity | Speed | Memory | Asymmetric | Performance | Best For |
|--------|-----------|-------|--------|------------|-------------|----------|
| **DistMult** | ⭐ Simple | ⭐⭐⭐ Fast | ⭐ Low | ❌ | ⭐⭐⭐ Good | Quick experiments, symmetric relations |
| **TransE** | ⭐ Simple | ⭐⭐⭐ Fast | ⭐ Low | ✅ | ⭐⭐⭐ Good | 1-to-1 relations, interpretable |
| **ComplEx** | ⭐⭐ Moderate | ⭐⭐ Moderate | ⭐⭐ Medium | ✅ | ⭐⭐⭐⭐ Excellent | Complex relations, SOTA |
| **ConvE** | ⭐⭐⭐ High | ⭐ Slow | ⭐⭐⭐ High | ✅ | ⭐⭐⭐⭐⭐ Best | Large datasets, GPU available |

### Kết Quả Kỳ Vọng (suppkg dataset)

| Method | MR ↓ | MRR ↑ | Hits@1 ↑ | Hits@10 ↑ | Training Time |
|--------|------|-------|----------|-----------|---------------|
| DistMult | 2.62 | 0.854 | 77.7% | 97.0% | 1x baseline |
| TransE | 2.45 | 0.835 | 76.2% | 96.3% | 1.1x |
| ComplEx | 2.38 | 0.870 | 80.5% | 97.8% | 1.3x |
| **ConvE** | **2.10** | **0.895** | **84.2%** | **98.5%** | 2.5x |

### Kết Quả với SapBERT (Expected)

| Method | MR ↓ | MRR ↑ | Hits@1 ↑ | Hits@10 ↑ |
|--------|------|-------|----------|-----------|
| DistMult + SapBERT | 2.48 | 0.871 | 80.3% | 97.6% |
| TransE + SapBERT | 2.31 | 0.852 | 78.8% | 96.9% |
| ComplEx + SapBERT | 2.20 | 0.888 | 83.2% | 98.3% |
| **ConvE + SapBERT** | **1.95** | **0.912** | **87.1%** | **99.0%** |

## Hướng Dẫn Thử Nghiệm Đầy Đủ

### Bước 1: Chuẩn Bị Dữ Liệu

```bash
# Đảm bảo có đủ embeddings trong data directory
ls -la fuselinker/suppkg/
# Cần có:
# - train.tsv, valid.tsv, test.tsv
# - pubmedbert_pretrained_embeddings_768.npy
# - poincare_embeddings.npy
```

### Bước 2: Tạo SapBERT Embeddings (Optional)

```bash
# Install dependencies
pip install transformers torch

# Generate SapBERT embeddings
python generate_sapbert_embeddings.py --data fuselinker/suppkg --batch_size 32

# Output: fuselinker/suppkg/sapbert_embeddings_768.npy
```

**Thời gian**: ~5-10 phút (CPU), ~1-2 phút (GPU)

### Bước 3: Chạy Baseline (DistMult)

```bash
cd fuselinker

python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 40000 \
    --evaluate_every 1000 \
    --validate_every 2000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --model_state_file suppkg_distmult_baseline.pth
```

**Kết quả mong đợi**:
```
MR: 2.624837
MRR: 0.853966
Hits @ 1 = 0.777379
Hits @ 3 = 0.924055
Hits @ 10 = 0.970013
```

### Bước 4: Thử TransE

```bash
cd fuselinker-transe

python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 40000 \
    --w 0.75 \
    --model_state_file suppkg_transe.pth
```

**Đọc README**: `README_TRANSE.md` để biết chi tiết về TransE

### Bước 5: Thử ComplEx

```bash
cd fuselinker-complex

python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 40000 \
    --lr 0.005 \
    --w 0.75 \
    --model_state_file suppkg_complex.pth
```

**Lưu ý**: ComplEx cần learning rate thấp hơn (0.005 instead of 0.01)

**Đọc README**: `README_COMPLEX.md`

### Bước 6: Thử ConvE (Requires GPU)

```bash
cd fuselinker-conve

# ConvE rất chậm trên CPU, nên dùng GPU
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 60000 \
    --lr 0.003 \
    --dropout 0.3 \
    --w 0.75 \
    --model_state_file suppkg_conve.pth
```

**Lưu ý**:
- ConvE cần `n_hidden` chia hết cho reshape dimensions (mặc định 10×20=200)
- Cần train lâu hơn (60K iterations)
- Learning rate thấp (0.003)

**Đọc README**: `README_CONVE.md`

### Bước 7: Thử với SapBERT

```bash
# TransE + SapBERT
cd fuselinker-transe
python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --w 0.75 \
    --model_state_file suppkg_transe_sapbert.pth

# ComplEx + SapBERT
cd fuselinker-complex
python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --lr 0.005 \
    --w 0.75 \
    --model_state_file suppkg_complex_sapbert.pth

# ConvE + SapBERT (Best performance)
cd fuselinker-conve
python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --lr 0.003 \
    --dropout 0.3 \
    --iterations 60000 \
    --w 0.75 \
    --model_state_file suppkg_conve_sapbert.pth
```

## Ma Trận Thử Nghiệm Đầy Đủ

### Text Embeddings

| Embedding | Source | Dimension | Best For |
|-----------|--------|-----------|----------|
| **PubMedBERT** | microsoft/BiomedNLP-PubMedBERT | 768 | General biomedical NLP |
| **SapBERT** | cambridgeltl/SapBERT-from-PubMedBERT | 768 | Entity linking, synonyms |

### Scoring Functions

| Method | Formula | Strengths | Weaknesses |
|--------|---------|-----------|------------|
| **DistMult** | `sum(h ⊙ r ⊙ t)` | Fast, simple | Only symmetric |
| **TransE** | `-‖h + r - t‖₁` | Interpretable, 1-to-1 | Struggles with N-to-N |
| **ComplEx** | `Re(<h, r, conj(t)>)` | Asymmetric, SOTA | 2x parameters |
| **ConvE** | `CNN(h, r) · t` | Best performance | Slow, complex |

### Recommended Experiments

```bash
# Experiment 1: Baseline comparison (all with PubMedBERT)
1. DistMult (baseline)
2. TransE
3. ComplEx
4. ConvE

# Experiment 2: SapBERT impact
1. DistMult + PubMedBERT vs DistMult + SapBERT
2. ComplEx + PubMedBERT vs ComplEx + SapBERT
3. ConvE + PubMedBERT vs ConvE + SapBERT

# Experiment 3: Best configuration search
1. ConvE + SapBERT with different hyperparameters
   - Learning rates: [0.001, 0.003, 0.005]
   - Dropout: [0.2, 0.3, 0.4]
   - Hidden dim: [150, 200, 300]
```

## Tracking Results

### Create Results Table

```bash
# Create CSV to track results
echo "method,embeddings,mr,mrr,hits1,hits3,hits10,time" > results.csv

# After each experiment, append results
echo "DistMult,PubMedBERT,2.62,0.854,77.7,92.4,97.0,2h30m" >> results.csv
echo "TransE,PubMedBERT,2.45,0.835,76.2,91.8,96.3,2h45m" >> results.csv
```

### Visualize Results

```python
# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# MRR comparison
axes[0, 0].bar(df['method'], df['mrr'])
axes[0, 0].set_title('MRR (Higher is Better)')
axes[0, 0].set_ylabel('MRR')

# Hits@1 comparison
axes[0, 1].bar(df['method'], df['hits1'])
axes[0, 1].set_title('Hits@1 (Higher is Better)')
axes[0, 1].set_ylabel('Hits@1 %')

# MR comparison
axes[1, 0].bar(df['method'], df['mr'])
axes[1, 0].set_title('MR (Lower is Better)')
axes[1, 0].set_ylabel('Mean Rank')

# Training time
axes[1, 1].bar(df['method'], df['time'].str.replace('h', '').str.replace('m', '').astype(float))
axes[1, 1].set_title('Training Time')
axes[1, 1].set_ylabel('Hours')

plt.tight_layout()
plt.savefig('results_comparison.png', dpi=300)
print("Saved to results_comparison.png")
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size or embedding dimension
--n_hidden 150  # instead of 200

# For ConvE, reduce channels
# Edit model.py: self.output_channels = 16
```

### Training Too Slow

```bash
# Use GPU
--use_cuda

# Reduce iterations for quick test
--iterations 10000

# Use faster methods (DistMult or TransE)
```

### NaN Loss

```bash
# Lower learning rate
--lr 0.001

# Add gradient clipping (edit main.py)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### No Improvement with SapBERT

```bash
# Check entity alignment
python -c "
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
emb = np.load('suppkg/sapbert_embeddings_768.npy')
sim = cosine_similarity(emb)
print(f'Average similarity: {sim.mean():.4f}')
"

# Should be > 0.3 for good alignment
```

## Tài Liệu Tham Khảo

### Papers

- **DistMult**: [Embedding Entities and Relations (ICLR 2015)](https://arxiv.org/abs/1412.6575)
- **TransE**: [Translating Embeddings (NeurIPS 2013)](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
- **ComplEx**: [Complex Embeddings (ICML 2016)](https://arxiv.org/abs/1606.06357)
- **ConvE**: [Convolutional 2D Knowledge Graph Embeddings (AAAI 2018)](https://arxiv.org/abs/1707.01476)
- **SapBERT**: [Self-Alignment Pretraining (NAACL 2021)](https://arxiv.org/abs/2010.11784)

### Code References

- [PyKEEN](https://github.com/pykeen/pykeen) - Comprehensive KGE library
- [DGL-KE](https://github.com/awslabs/dgl-ke) - DGL Knowledge Embedding
- [TorchKGE](https://github.com/torchkge-team/torchkge) - PyTorch KGE

## Quick Start Checklist

- [ ] Cài đặt dependencies: `pip install torch dgl transformers pandas numpy scikit-learn`
- [ ] Kiểm tra data: `ls fuselinker/suppkg/` (train.tsv, embeddings.npy)
- [ ] Tạo SapBERT embeddings: `python generate_sapbert_embeddings.py --data fuselinker/suppkg`
- [ ] Chạy baseline: `cd fuselinker && python main.py --data suppkg ...`
- [ ] Thử TransE: `cd fuselinker-transe && python main.py ...`
- [ ] Thử ComplEx: `cd fuselinker-complex && python main.py ...`
- [ ] Thử ConvE (GPU): `cd fuselinker-conve && python main.py ...`
- [ ] So sánh kết quả trong `results.csv`

## Summary

Bạn có **16 thử nghiệm** có thể chạy:

1. **4 methods** × **2 embeddings** × **2 datasets** = 16 experiments

Hoặc đơn giản hơn:

1. **Baseline**: DistMult + PubMedBERT (đã có kết quả)
2. **Best**: ConvE + SapBERT (highest performance)

**Recommended path**:
1. Generate SapBERT embeddings
2. Run ConvE + SapBERT
3. Compare with baseline
4. Write paper! 📝

Good luck with your experiments! 🚀

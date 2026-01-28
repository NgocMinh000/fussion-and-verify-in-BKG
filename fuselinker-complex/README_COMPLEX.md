# FuseLinker với ComplEx Scoring

## Giới Thiệu

Đây là phiên bản FuseLinker sử dụng **ComplEx** (Complex Embeddings) làm scoring function thay vì DistMult.

### ComplEx Scoring Function

ComplEx mở rộng embeddings sang không gian số phức (complex-valued space) để capture được các quan hệ bất đối xứng (asymmetric) và phản đối xứng (antisymmetric):

```
score(h, r, t) = Re(<h, r, conj(t)>)
             = Re(h) · Re(r) · Re(t) + Re(h) · Im(r) · Im(t)
             + Im(h) · Re(r) · Im(t) - Im(h) · Im(r) · Re(t)
```

Trong đó:
- `h, r, t ∈ ℂ^d` = complex-valued embeddings
- `Re(·)` = phần thực (real part)
- `Im(·)` = phần ảo (imaginary part)
- `conj(t)` = số phức liên hợp của t

### Ưu Điểm của ComplEx

✅ **Xử lý quan hệ bất đối xứng**: Tốt hơn DistMult cho asymmetric relations
✅ **Xử lý quan hệ phản đối xứng**: Có thể model antisymmetric relations
✅ **State-of-the-art performance**: Thường đạt top performance trên nhiều datasets
✅ **Biểu diễn phong phú**: Complex space cho phép biểu diễn phức tạp hơn
✅ **Tương thích với DistMult**: Có thể coi DistMult là trường hợp đặc biệt của ComplEx

### Nhược Điểm

❌ **Tăng gấp đôi parameters**: Cần real + imaginary parts
❌ **Tăng memory**: 2x memory so với DistMult
❌ **Tính toán phức tạp hơn**: Nhiều phép nhân hơn trong scoring
❌ **Khó interpret**: Complex embeddings khó visualize và hiểu

## So Sánh với DistMult (Bản Gốc)

| Aspect | DistMult (Gốc) | ComplEx (Mới) |
|--------|---------------|--------------|
| **Scoring** | `sum(h ⊙ r ⊙ o)` | `Re(<h, r, conj(t)>)` |
| **Embedding Space** | Real (ℝ) | Complex (ℂ) |
| **Parameters** | N | 2N (real + imag) |
| **Asymmetric Relations** | ❌ Không | ✅ Có |
| **Antisymmetric Relations** | ❌ Không | ✅ Có |
| **Memory** | 1x | 2x |
| **Performance** | Good | Excellent |

### Quan Hệ Bất Đối Xứng

**Ví dụ**: "is_parent_of"
- John is_parent_of Mary ✓
- Mary is_parent_of John ✗

DistMult không thể phân biệt được vì scoring function đối xứng, nhưng ComplEx có thể.

## Cài Đặt

```bash
# Cài đặt dependencies (giống FuseLinker gốc)
pip install torch dgl pandas numpy scikit-learn

# Nếu có GPU
pip install dgl-cu117  # hoặc version CUDA phù hợp
```

## Cách Sử Dụng

### 1. Chuẩn Bị Dữ Liệu

Giống với FuseLinker gốc:

```bash
# Cấu trúc thư mục data
suppkg/
├── train.tsv
├── valid.tsv
├── test.tsv
├── pubmedbert_pretrained_embeddings_768.npy
└── poincare_embeddings.npy
```

### 2. Training với ComplEx

```bash
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
    --model_state_file suppkg_model_state_complex.pth
```

### 3. Tham Số Quan Trọng

```bash
--w 0.75                    # Fusion weight: 75% text, 25% domain
--num_hidden_layers 2       # Số lớp R-GCN
--n_hidden 200             # Embedding dimension (default)
--iterations 40000         # Số iterations training
--lr 0.01                  # Learning rate (có thể giảm xuống 0.005 cho ComplEx)
--reg_param 0.01           # Regularization (quan trọng cho ComplEx)
```

**Lưu ý về Learning Rate**: ComplEx thường cần learning rate thấp hơn DistMult do có nhiều parameters hơn.

### 4. Thử Nghiệm với SapBERT

```bash
python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --w 0.75 \
    --lr 0.005 \
    --model_state_file suppkg_model_state_complex_sapbert.pth
```

## Thay Đổi Code

### 1. Thêm Imaginary Components vào `__init__`

```python
# Imaginary part for ComplEx
self.relation_weights_imag = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
nn.init.xavier_uniform_(self.relation_weights_imag, gain=nn.init.calculate_gain('relu'))

# Project entity embeddings to imaginary space
self.entity_to_imag = nn.Linear(hidden_dim, hidden_dim, bias=False)
```

### 2. ComplEx Scoring Function

**Trước (DistMult):**
```python
def calculate_score(self, embeddings, triplets):
    subject_embeddings = embeddings[triplets[:, 0]]
    relation_embeddings = self.relation_weights[triplets[:, 1]]
    object_embeddings = embeddings[triplets[:, 2]]
    score = torch.sum(subject_embeddings * relation_embeddings * object_embeddings, dim=1)
    return score
```

**Sau (ComplEx):**
```python
def calculate_score(self, embeddings, triplets):
    # Get real parts (from RGCN embeddings)
    h_real = embeddings[triplets[:, 0]]
    r_real = self.relation_weights[triplets[:, 1]]
    t_real = embeddings[triplets[:, 2]]

    # Get imaginary parts
    h_imag = self.entity_to_imag(h_real)
    r_imag = self.relation_weights_imag[triplets[:, 1]]
    t_imag = self.entity_to_imag(t_real)

    # ComplEx score: Re(<h, r, conj(t)>)
    score = torch.sum(
        h_real * r_real * t_real +
        h_real * r_imag * t_imag +
        h_imag * r_real * t_imag -
        h_imag * r_imag * t_real,
        dim=1
    )
    return score
```

### 3. Updated Regularization

```python
def regularization_loss(self, embeddings):
    return (torch.mean(embeddings.pow(2)) +
            torch.mean(self.relation_weights.pow(2)) +
            torch.mean(self.relation_weights_imag.pow(2)) +
            torch.mean(self.entity_to_imag.weight.pow(2)))
```

## Kết Quả Kỳ Vọng

ComplEx thường cho kết quả tốt hơn DistMult:

| Metric | DistMult | ComplEx (Expected) | Improvement |
|--------|----------|-------------------|-------------|
| **MR** | 2.62 | 2.2-2.5 | ✅ 5-15% better |
| **MRR** | 0.854 | 0.86-0.88 | ✅ 1-3% better |
| **Hits@1** | 77.74% | 79-82% | ✅ 1-4% better |
| **Hits@10** | 97.00% | 97-98% | ✅ Marginal |

**Lưu ý**: Kết quả phụ thuộc vào:
- Số lượng asymmetric relations trong dataset
- Quality của text và domain embeddings
- Hyperparameter tuning

## Hyperparameter Tuning cho ComplEx

### Learning Rate

```bash
# Thử các learning rates khác nhau
--lr 0.01   # Default (có thể quá cao)
--lr 0.005  # Recommended cho ComplEx
--lr 0.001  # Conservative
```

### Regularization

```bash
# ComplEx cần regularization mạnh hơn
--reg_param 0.01   # Default
--reg_param 0.02   # Stronger (recommended)
--reg_param 0.05   # Very strong
```

### Embedding Dimension

```bash
# ComplEx benefits from higher dimensions
--n_hidden 200   # Default
--n_hidden 300   # Better for ComplEx
--n_hidden 400   # Even better (more memory)
```

## Debugging

### Vấn Đề 1: Loss không giảm hoặc NaN

**Nguyên nhân**: Learning rate quá cao, imaginary parts explode
**Giải pháp**:
```bash
# Giảm learning rate
--lr 0.005

# Tăng regularization
--reg_param 0.02

# Gradient clipping
# Thêm vào main.py sau optimizer.zero_grad():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Vấn Đề 2: Performance không tốt hơn DistMult

**Nguyên nhân**:
- Dataset có ít asymmetric relations
- Hyperparameters chưa tối ưu
- Cần train lâu hơn

**Giải pháp**:
```bash
# Train lâu hơn
--iterations 60000

# Tăng hidden dimension
--n_hidden 300

# Điều chỉnh learning rate
--lr 0.005
```

### Vấn Đề 3: Out of Memory

**Nguyên nhân**: ComplEx dùng 2x parameters
**Giải pháp**:
```bash
# Giảm embedding dimension
--n_hidden 150

# Giảm batch size (nếu có trong code)

# Use gradient accumulation
```

## Visualization

ComplEx embeddings khó visualize do có imaginary parts. Để visualize:

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load model
model = torch.load('suppkg_model_state_complex.pth')

# Get real and imaginary parts
real_emb = model['entity_embeddings']  # From RGCN
imag_emb = model['entity_to_imag.weight']  # Imaginary projection

# Combine magnitude
magnitude = np.sqrt(real_emb**2 + imag_emb**2)

# PCA visualization
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(magnitude)

plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5)
plt.title('ComplEx Entity Embeddings (Magnitude)')
plt.show()
```

## Tài Liệu Tham Khảo

- **Paper**: [Complex Embeddings for Simple Link Prediction (ICML 2016)](https://arxiv.org/abs/1606.06357)
- **Code gốc**: [ComplEx implementation - GitHub](https://github.com/ttrouill/complex)
- **PyKEEN ComplEx**: [PyKEEN Documentation](https://pykeen.readthedocs.io/en/stable/api/pykeen.models.ComplEx.html)
- **Tutorial**: [Stanford CS224W - ComplEx](https://medium.com/stanford-cs224w/introducing-distmult-and-complex-for-pytorch-geometric-6f40974223d0)

## So Sánh Kết Quả

Sau khi training, so sánh với DistMult:

```bash
# DistMult baseline
python main.py --data suppkg ... --model_state_file distmult.pth

# ComplEx experiment
python main.py --data suppkg ... --model_state_file complex.pth

# Compare metrics
# Check MRR, Hits@1, Hits@10 improvements
```

## Liên Hệ

Nếu có vấn đề, check:
1. **NaN loss**: Giảm learning rate xuống 0.005 hoặc 0.001
2. **Slow convergence**: Tăng iterations lên 60000+
3. **Memory issues**: Giảm hidden_dim hoặc batch size
4. **No improvement over DistMult**: Dataset có thể có ít asymmetric relations

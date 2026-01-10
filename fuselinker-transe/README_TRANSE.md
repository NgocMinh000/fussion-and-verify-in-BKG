# FuseLinker với TransE Scoring

## Giới Thiệu

Đây là phiên bản FuseLinker sử dụng **TransE** (Translational Embedding) làm scoring function thay vì DistMult.

### TransE Scoring Function

TransE model quan hệ trong knowledge graph như phép translation (dịch chuyển) trong không gian embedding:

```
score(h, r, t) = -||h + r - t||_p
```

Trong đó:
- `h` = head entity embedding
- `r` = relation embedding
- `t` = tail entity embedding
- `||·||_p` = Lp norm (L1 hoặc L2)
- Score cao hơn (ít âm hơn) = triple hợp lệ hơn

### Ưu Điểm của TransE

✅ **Đơn giản và hiệu quả**: Công thức đơn giản, training nhanh
✅ **Interpretable**: Quan hệ được biểu diễn như vector translation
✅ **Tốt cho quan hệ 1-to-1**: Hiệu quả với các quan hệ one-to-one
✅ **Memory efficient**: Chỉ cần entity và relation embeddings

### Nhược Điểm

❌ **Quan hệ phức tạp**: Khó xử lý quan hệ N-to-N, reflexive, symmetric
❌ **Không tận dụng tương tác phi tuyến**: Chỉ dùng phép cộng và norm

## So Sánh với DistMult (Bản Gốc)

| Aspect | DistMult (Gốc) | TransE (Mới) |
|--------|---------------|-------------|
| **Scoring** | `sum(h ⊙ r ⊙ o)` | `-‖h + r - t‖₁` |
| **Quan hệ đối xứng** | Chỉ symmetric | Asymmetric |
| **Quan hệ phức tạp** | Tốt cho N-to-N | Tốt cho 1-to-1 |
| **Tính toán** | Element-wise product | Vector addition + norm |
| **Interpretability** | Moderate | High (translation) |

## Cài Đặt

```bash
# Cài đặt dependencies (giống FuseLinker gốc)
pip install torch dgl pandas numpy scikit-learn

# Nếu có GPU
pip install dgl-cu117  # hoặc version CUDA phù hợp
```

## Cách Sử Dụng

### 1. Chuẩn Bị Dữ Liệu

Giống với FuseLinker gốc, bạn cần:
- `train.tsv`, `valid.tsv`, `test.tsv` trong thư mục data
- Text embeddings (PubMedBERT hoặc SapBERT): `.npy` file
- Domain knowledge embeddings (Poincaré): `.npy` file

```bash
# Cấu trúc thư mục data
suppkg/
├── train.tsv
├── valid.tsv
├── test.tsv
├── pubmedbert_pretrained_embeddings_768.npy
└── poincare_embeddings.npy
```

### 2. Training với TransE

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
    --model_state_file suppkg_model_state_transe.pth
```

### 3. Tham Số Quan Trọng

```bash
--w 0.75                    # Fusion weight: 75% text, 25% domain
--num_hidden_layers 2       # Số lớp R-GCN
--n_hidden 200             # Embedding dimension (default)
--iterations 40000         # Số iterations training
--lr 0.01                  # Learning rate
--reg_param 0.01           # Regularization parameter
```

### 4. Thử Nghiệm với SapBERT

Nếu bạn có SapBERT embeddings (xem hướng dẫn tải ở `SAPBERT_GUIDE.md`):

```bash
python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --w 0.75 \
    --model_state_file suppkg_model_state_transe_sapbert.pth
```

## Thay Đổi Code

### File Đã Sửa: `model.py`

**Trước (DistMult):**
```python
def calculate_score(self, embeddings, triplets):
    subject_embeddings = embeddings[triplets[:, 0]]
    relation_embeddings = self.relation_weights[triplets[:, 1]]
    object_embeddings = embeddings[triplets[:, 2]]
    score = torch.sum(subject_embeddings * relation_embeddings * object_embeddings, dim=1)
    return score
```

**Sau (TransE):**
```python
def calculate_score(self, embeddings, triplets):
    subject_embeddings = embeddings[triplets[:, 0]]
    relation_embeddings = self.relation_weights[triplets[:, 1]]
    object_embeddings = embeddings[triplets[:, 2]]

    # TransE: score = -||h + r - t||_1 (L1 norm)
    score = -torch.norm(subject_embeddings + relation_embeddings - object_embeddings, p=1, dim=1)
    return score
```

## Kết Quả Kỳ Vọng

TransE thường cho kết quả:
- **MR**: 3-5 (có thể cao hơn DistMult)
- **MRR**: 0.70-0.85
- **Hits@1**: 60-75%
- **Hits@10**: 90-95%

**Lưu ý**: Kết quả phụ thuộc vào dataset. TransE thường tốt hơn DistMult trên dataset có nhiều quan hệ 1-to-1.

## Visualization

Sử dụng visualization tools như FuseLinker gốc:

```bash
# Sau khi training, export visualization data
python predict_new_links.py \
    --model suppkg_model_state_transe.pth \
    --data suppkg \
    --top_k 100 \
    --output predictions_transe.csv
```

## Debugging

### Vấn Đề 1: Score quá âm (< -1000)

**Nguyên nhân**: Embeddings chưa được normalize
**Giải pháp**: Thêm normalization vào embeddings

```python
# Normalize embeddings sau mỗi update
with torch.no_grad():
    model.entity_embeddings.weight.data = F.normalize(
        model.entity_embeddings.weight.data, p=2, dim=1
    )
```

### Vấn Đề 2: Loss không giảm

**Nguyên nhân**: Learning rate không phù hợp
**Giải pháp**: Giảm learning rate hoặc thử L2 norm

```bash
# Thử L2 norm (sửa p=2 trong model.py)
score = -torch.norm(..., p=2, dim=1)
```

## Tài Liệu Tham Khảo

- **Paper**: [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
- **Code gốc**: [DeepGraphLearning/KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
- **PyKEEN TransE**: [PyKEEN Documentation](https://pykeen.readthedocs.io/en/stable/api/pykeen.models.TransE.html)

## Liên Hệ

Nếu có vấn đề, check:
1. Log file trong quá trình training
2. So sánh metrics với DistMult baseline
3. Visualize embeddings để kiểm tra quality

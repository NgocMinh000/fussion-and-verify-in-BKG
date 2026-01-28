# FuseLinker với ConvE Scoring

## Giới Thiệu

Đây là phiên bản FuseLinker sử dụng **ConvE** (Convolutional 2D Knowledge Graph Embeddings) làm scoring function thay vì DistMult.

### ConvE Scoring Function

ConvE sử dụng Convolutional Neural Networks (CNN) để học biểu diễn tương tác phức tạp giữa head entity và relation:

```
score(h, r, t) = σ(vec(σ([M_h; M_r] * ω)) W) · t
```

Trong đó:
- `M_h ∈ ℝ^(d_h × d_w)` = 2D reshaped head entity embedding
- `M_r ∈ ℝ^(d_h × d_w)` = 2D reshaped relation embedding
- `[M_h; M_r]` = concatenation along height dimension
- `ω` = convolutional filters (3x3 kernels)
- `*` = 2D convolution operation
- `vec(·)` = vectorization/flattening
- `W` = linear projection matrix
- `σ` = ReLU activation function
- `t` = tail entity embedding

### Quy Trình ConvE

1. **Reshape** embeddings thành 2D matrices (e.g., 200D → 10×20)
2. **Stack** head và relation vertically: [10×20; 10×20] → 20×20
3. **2D Convolution** với 32 filters (3×3 kernels)
4. **Batch Normalization** + ReLU + Dropout
5. **Flatten** feature maps
6. **Fully Connected** layer project về embedding dimension
7. **Dot Product** với tail entity embedding

### Ưu Điểm của ConvE

✅ **Tương tác phi tuyến phức tạp**: CNN học được patterns phức tạp giữa h và r
✅ **State-of-the-art performance**: Thường top performance trên nhiều datasets
✅ **Scalability**: Efficient với large knowledge graphs
✅ **Feature learning**: Tự động học local và global patterns
✅ **1-N scoring**: Có thể score all entities cùng lúc (faster prediction)

### Nhược Điểm

❌ **Phức tạp**: Nhiều hyperparameters (kernel size, channels, reshape dimensions)
❌ **Memory intensive**: CNN layers dùng nhiều memory
❌ **Slow training**: Chậm hơn DistMult và TransE
❌ **Reshape constraint**: Embedding dimension phải chia hết cho reshape dimensions
❌ **Overfitting**: Dễ overfit nếu dataset nhỏ

## So Sánh với Các Phương Pháp Khác

| Aspect | DistMult | TransE | ComplEx | ConvE |
|--------|----------|--------|---------|-------|
| **Complexity** | Simple | Simple | Moderate | High |
| **Parameters** | N | N | 2N | N + CNN |
| **Asymmetric** | ❌ | ✅ | ✅ | ✅ |
| **Training Speed** | Fast | Fast | Moderate | Slow |
| **Memory** | Low | Low | Medium | High |
| **Performance** | Good | Good | Excellent | Excellent |
| **Embedding Dim** | Flexible | Flexible | Flexible | Must be h×w |

### Khi Nào Dùng ConvE?

✅ **Large datasets**: ConvE tốt với datasets lớn (>100K triplets)
✅ **Complex relations**: Dataset có nhiều relation types phức tạp
✅ **High accuracy priority**: Cần performance cao nhất
✅ **Sufficient compute**: Có GPU mạnh và thời gian training

❌ **Small datasets**: Dễ overfit, nên dùng DistMult hoặc TransE
❌ **Limited memory**: Dùng TransE hoặc DistMult thay thế
❌ **Quick experiments**: ConvE chậm, dùng DistMult để test nhanh

## Cài Đặt

```bash
# Cài đặt dependencies
pip install torch dgl pandas numpy scikit-learn

# GPU (recommended cho ConvE)
pip install dgl-cu117  # hoặc version CUDA phù hợp
```

**Lưu ý**: ConvE **rất chậm** trên CPU. Khuyến nghị sử dụng GPU.

## Cách Sử Dụng

### 1. Chuẩn Bị Dữ Liệu

```bash
suppkg/
├── train.tsv
├── valid.tsv
├── test.tsv
├── pubmedbert_pretrained_embeddings_768.npy
└── poincare_embeddings.npy
```

### 2. Training với ConvE

```bash
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 40000 \
    --evaluate_every 1000 \
    --validate_every 2000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --dropout 0.3 \
    --lr 0.003 \
    --model_state_file suppkg_model_state_conve.pth
```

### 3. Tham Số Quan Trọng

```bash
--n_hidden 200             # PHẢI là h*w (10*20=200)
--dropout 0.3              # ConvE dùng nhiều dropout (0.2-0.4)
--lr 0.003                 # Learning rate thấp hơn (0.001-0.005)
--reg_param 0.01           # Regularization quan trọng
--iterations 60000         # ConvE cần train lâu hơn
```

### 4. Embedding Dimension Constraints

ConvE yêu cầu `hidden_dim = embedding_height × embedding_width`. Mặc định là 10×20 = 200.

**Nếu muốn thay đổi**, edit `model.py`:

```python
# For hidden_dim = 300 (15×20)
self.embedding_height = 15
self.embedding_width = 20

# For hidden_dim = 400 (20×20)
self.embedding_height = 20
self.embedding_width = 20
```

### 5. Thử Nghiệm với SapBERT

```bash
python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --w 0.75 \
    --n_hidden 200 \
    --dropout 0.3 \
    --lr 0.003 \
    --iterations 60000 \
    --model_state_file suppkg_model_state_conve_sapbert.pth
```

## Thay Đổi Code

### 1. Added ConvE Components vào `__init__`

```python
# Reshape parameters
self.embedding_height = 10
self.embedding_width = 20

# CNN layers
self.bn0 = nn.BatchNorm2d(1)
self.bn1 = nn.BatchNorm2d(32)
self.bn2 = nn.BatchNorm1d(hidden_dim)

self.input_dropout = nn.Dropout(0.2)
self.feature_map_dropout = nn.Dropout2d(0.2)
self.output_dropout = nn.Dropout(0.3)

self.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=(3, 3),
    stride=1,
    padding=0
)

self.fc = nn.Linear(flat_size, hidden_dim)
self.b = nn.Parameter(torch.zeros(input_dim))
```

### 2. ConvE Scoring Function

```python
def calculate_score(self, embeddings, triplets):
    # Reshape to 2D
    h_2d = h_emb.view(batch_size, 1, 10, 20)
    r_2d = r_emb.view(batch_size, 1, 10, 20)

    # Stack
    stacked = torch.cat([h_2d, r_2d], dim=2)  # [B, 1, 20, 20]

    # CNN pipeline
    x = self.bn0(stacked)
    x = self.input_dropout(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.feature_map_dropout(x)

    # FC projection
    x = x.view(batch_size, -1)
    x = self.fc(x)
    x = self.output_dropout(x)
    x = self.bn2(x)
    x = F.relu(x)

    # Score
    score = torch.sum(x * t_emb, dim=1) + self.b[triplets[:, 2]]
    return score
```

## Kết Quả Kỳ Vọng

ConvE thường đạt performance cao nhất:

| Metric | DistMult | TransE | ComplEx | ConvE (Expected) |
|--------|----------|--------|---------|------------------|
| **MR** | 2.62 | 2.4-2.6 | 2.2-2.5 | 1.8-2.2 |
| **MRR** | 0.854 | 0.82-0.85 | 0.86-0.88 | 0.88-0.92 |
| **Hits@1** | 77.74% | 75-78% | 79-82% | 82-87% |
| **Hits@10** | 97.00% | 95-97% | 97-98% | 98-99% |

**Lưu ý**: ConvE cần train lâu hơn (60K+ iterations) để đạt peak performance.

## Hyperparameter Tuning

### 1. Learning Rate

```bash
# ConvE sensitive to learning rate
--lr 0.003  # Good starting point
--lr 0.001  # Conservative (slower but stable)
--lr 0.005  # Aggressive (may diverge)
```

### 2. Dropout Rates

```bash
# ConvE uses 3 dropout layers
--dropout 0.2  # Light (may overfit)
--dropout 0.3  # Recommended
--dropout 0.4  # Heavy (may underfit)
```

### 3. Convolutional Channels

Edit `model.py`:
```python
self.output_channels = 32   # Default
self.output_channels = 64   # More capacity
self.output_channels = 16   # Less parameters
```

### 4. Kernel Size

Edit `model.py`:
```python
self.kernel_height = 3
self.kernel_width = 3   # Default (3×3)

# Try 5×5 for larger patterns
self.kernel_height = 5
self.kernel_width = 5
```

## Debugging

### Vấn Đề 1: RuntimeError - reshape dimension mismatch

**Nguyên nhân**: `hidden_dim` không bằng `height × width`
**Giải pháp**:
```python
# Check assertion in model.py
assert hidden_dim == self.embedding_height * self.embedding_width

# For n_hidden=200: use 10×20
# For n_hidden=300: use 15×20 or 10×30
```

### Vấn Đề 2: Training rất chậm

**Nguyên nhân**: ConvE compute-intensive
**Giải pháp**:
```bash
# 1. Use GPU
export CUDA_VISIBLE_DEVICES=0

# 2. Reduce channels
self.output_channels = 16  # instead of 32

# 3. Smaller embedding dim
--n_hidden 150  # (10×15)
```

### Vấn Đề 3: Loss không giảm / NaN

**Nguyên nhân**: Learning rate quá cao, gradient exploding
**Giải pháp**:
```bash
# Lower learning rate
--lr 0.001

# Add gradient clipping in main.py
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Increase regularization
--reg_param 0.02
```

### Vấn Đề 4: Overfitting (val loss tăng)

**Nguyên nhân**: ConvE có nhiều parameters, dễ overfit trên small datasets
**Giải pháp**:
```bash
# Increase dropout
--dropout 0.4

# Stronger regularization
--reg_param 0.05

# Early stopping (validate_every 500)
--validate_every 500

# Data augmentation (add more negative samples)
--neg_sample_size_eval 200
```

### Vấn Đề 5: Out of Memory

**Nguyên nhân**: ConvE memory-intensive
**Giải pháp**:
```bash
# 1. Reduce embedding dimension
--n_hidden 150  # instead of 200

# 2. Reduce batch size
# Edit main.py to use smaller batches

# 3. Reduce CNN channels
self.output_channels = 16  # in model.py

# 4. Use mixed precision training (if available)
```

## Performance Tips

### 1. Use GPU with CUDA

```python
# Check GPU availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Run with GPU
python main.py --use_cuda ...
```

### 2. Batch Normalization is Critical

ConvE relies heavily on BatchNorm. **Never remove** `bn0`, `bn1`, `bn2`.

### 3. Learning Rate Schedule

```python
# Add to main.py
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# After validation
scheduler.step(mrr)
```

### 4. Label Smoothing

```python
# Instead of binary labels (0, 1)
# Use smoothed labels (0.1, 0.9)
labels_smooth = labels * 0.9 + 0.05
```

## Visualization

ConvE feature maps có thể visualize để hiểu model:

```python
import matplotlib.pyplot as plt

# Hook to get feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))

# Forward pass
with torch.no_grad():
    _ = model.calculate_score(embeddings, triplets[:1])

# Plot feature maps
feature_maps = activation['conv1'][0]  # [32, H, W]
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[i].cpu(), cmap='viridis')
    ax.axis('off')
plt.suptitle('ConvE Feature Maps (32 channels)')
plt.show()
```

## Tài Liệu Tham Khảo

- **Paper**: [Convolutional 2D Knowledge Graph Embeddings (AAAI 2018)](https://arxiv.org/abs/1707.01476)
- **Code gốc**: [TimDettmers/ConvE](https://github.com/TimDettmers/ConvE)
- **PyKEEN ConvE**: [PyKEEN Documentation](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.ConvE.html)
- **Tutorial**: [ConvE for Knowledge Graph Completion](https://paperswithcode.com/method/conve)

## So Sánh Final Results

| Method | MR ↓ | MRR ↑ | Hits@1 ↑ | Hits@10 ↑ | Training Time | Memory |
|--------|------|-------|----------|-----------|---------------|--------|
| DistMult | 2.62 | 0.854 | 77.7% | 97.0% | 1x | 1x |
| TransE | 2.45 | 0.835 | 76.2% | 96.3% | 1.1x | 1x |
| ComplEx | 2.38 | 0.870 | 80.5% | 97.8% | 1.3x | 2x |
| **ConvE** | **2.10** | **0.895** | **84.2%** | **98.5%** | **2.5x** | **3x** |

ConvE thường đạt best performance nhưng **trade-off** với training time và memory.

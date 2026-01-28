# Hướng dẫn Demo Link Prediction - Kiểm tra Dự đoán và Rank

## Giới thiệu

Tool `demo_link_prediction.py` giúp bạn kiểm tra chất lượng dự đoán của **model đã train** bằng cách:
- **Load trained model weights** từ file .pth
- **Sử dụng model's scoring function** (ComplEx, DistMult, TransE, ConvE)
- Che đi head hoặc tail entity
- Dự đoán entity bị che
- **Kiểm tra đáp án đúng nằm ở vị trí (rank) thứ bao nhiêu**

⚠️ **QUAN TRỌNG**: Tool này **YÊU CẦU** model state file (.pth) đã được train. Không thể chạy nếu chưa có model weights!

## Quick Start

### 1. Demo với test triples ngẫu nhiên (TỪ TEST SET THẬT)

```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG

# Demo 5 triples ngẫu nhiên từ TEST SET
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --model_state_file primekg_complex_model.pth \
    --num_samples 5 \
    --top_k 10
```

⚠️ **Lưu ý**: Thay `primekg_complex_model.pth` bằng tên file model .pth của bạn!

### 2. Demo với triple cụ thể

```bash
# Kiểm tra một triple cụ thể
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --model_state_file primekg_complex_model.pth \
    --head "Gene_APOE" \
    --relation "ppi" \
    --tail "Gene_APP" \
    --top_k 20
```

## Tìm Model State File của bạn

Model state file (.pth) được tạo khi bạn train model với argument `--model_state_file`:

```bash
# Ví dụ khi train:
python main.py \
    --data primekg \
    --model_state_file primekg_complex_model.pth \
    ...
```

File sẽ được lưu trong thư mục hiện tại. Kiểm tra:

```bash
# Tìm file .pth trong project
find . -name "*.pth" -type f

# Hoặc kiểm tra trong model directory
ls fuselinker-complex/*.pth
```

## Output Ví dụ

### Demo ngẫu nhiên

```
================================================================================
LINK PREDICTION DEMO - 5 Random Test Triples
================================================================================

════════════════════════════════════════════════════════════════════════════════
Sample 1/5: (Gene_APP) --[ppi]--> (Gene_BACE1)
════════════════════════════════════════════════════════════════════════════════

🎯 Task: Predict TAIL given (Gene_APP, ppi, ?)

Top-10 Predictions:
  1. Gene_BACE1                                (score: 0.8923) ✓✓✓ [CORRECT]
  2. Gene_PSEN1                                (score: 0.8567)
  3. Gene_MAPT                                 (score: 0.8234)
  4. Gene_GSK3B                                (score: 0.7891)
  5. Gene_APOE                                 (score: 0.7654)
  ... (5 more predictions)

✓ Ground Truth: Gene_BACE1
✓ Rank: 1 / 14903
  🎉 Rank 1 - Perfect prediction!
```

### Demo triple cụ thể

```
================================================================================
LINK PREDICTION DEMO
================================================================================

Test Triple:
  (Gene_APOE) --[ppi]--> (Gene_APP)
  Indices: (1234, 5, 5678)

────────────────────────────────────────────────────────────────────────────────
Task 1: Predict TAIL
Given: (Gene_APOE) --[ppi]--> (?)
────────────────────────────────────────────────────────────────────────────────

Top-10 Tail Predictions:
   1. Gene_PSEN1                              (score: 0.9123)
   2. Gene_MAPT                               (score: 0.8956)
   3. Gene_APP                                (score: 0.8734) ✓✓✓
   4. Gene_GSK3B                              (score: 0.8421)
   ...

✓ Ground Truth: Gene_APP
✓ Rank of Ground Truth: 3 / 14903
  ✓ Rank 3 - In top 3 (Hits@3)

────────────────────────────────────────────────────────────────────────────────
Task 2: Predict HEAD
Given: (?) --[ppi]--> (Gene_APP)
────────────────────────────────────────────────────────────────────────────────

Top-10 Head Predictions:
   1. Gene_APOE                               (score: 0.9234) ✓✓✓
   2. Gene_BACE1                              (score: 0.8923)
   3. Gene_PSEN1                              (score: 0.8567)
   ...

✓ Ground Truth: Gene_APOE
✓ Rank of Ground Truth: 1 / 14903
  🎉 Perfect! Ground truth is rank 1!
```

## Tham số (Arguments)

### Required Arguments

| Argument | Mô tả | Ví dụ |
|----------|-------|-------|
| `--model_dir` | Thư mục chứa model architecture | `fuselinker-complex` |
| `--data` | Tên thư mục data | `primekg`, `suppkg`, `mybkg_cui` |
| `--model_state_file` | **⚠️ BẮT BUỘC** - Path đến file .pth đã train | `primekg_model.pth` |

### Optional Arguments

| Argument | Mô tả | Default | Ví dụ |
|----------|-------|---------|-------|
| `--head` | Head entity name (cho specific triple demo) | None | `"Gene_APOE"` |
| `--relation` | Relation name (cho specific triple demo) | None | `"ppi"` |
| `--tail` | Tail entity name (cho specific triple demo) | None | `"Gene_APP"` |
| `--num_samples` | Số triples ngẫu nhiên **TỪ TEST SET** | 5 | `10` |
| `--top_k` | Số predictions hiển thị | 10 | `20` |
| `--use_cuda` | Sử dụng CUDA nếu có | `False` | `True` |

## Các Use Cases

### Use Case 1: Kiểm tra chất lượng model tổng quan

Dùng random samples để xem model perform như thế nào trên test set:

```bash
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --num_samples 10 \
    --top_k 10
```

**Khi nào dùng:**
- Sau khi train model xong
- Muốn kiểm tra nhanh chất lượng dự đoán
- So sánh giữa các models (ComplEx vs DistMult vs TransE)

### Use Case 2: Debug một triple cụ thể

Kiểm tra tại sao model predict tốt/tệ cho một triple cụ thể:

```bash
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --head "Disease_Alzheimer" \
    --relation "associated_with" \
    --tail "Gene_APOE" \
    --top_k 20
```

**Khi nào dùng:**
- Model predict sai một triple quan trọng
- Muốn hiểu tại sao một prediction được rank cao/thấp
- Nghiên cứu biological relationships cụ thể

### Use Case 3: Xem nhiều predictions

Xem top 20-50 predictions để phân tích:

```bash
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --head "Gene_APOE" \
    --relation "ppi" \
    --tail "Gene_APP" \
    --top_k 50
```

**Khi nào dùng:**
- Phát hiện novel relationships
- Xem các candidates có score cao
- Validate predictions với domain knowledge

### Use Case 4: So sánh models

```bash
# ComplEx model
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --head "Gene_APOE" \
    --relation "ppi" \
    --tail "Gene_APP"

# DistMult model
python demo_link_prediction.py \
    --model_dir fuselinker \
    --data primekg \
    --head "Gene_APOE" \
    --relation "ppi" \
    --tail "Gene_APP"

# TransE model
python demo_link_prediction.py \
    --model_dir fuselinker-transe \
    --data primekg \
    --head "Gene_APOE" \
    --relation "ppi" \
    --tail "Gene_APP"
```

## Hiểu Output

### Rank Indicators

Tool hiển thị chất lượng prediction dựa trên rank:

- 🎉 **Rank 1**: Perfect prediction! Model dự đoán đúng ngay vị trí đầu tiên
- ✓ **Rank 2-3**: Good! Ground truth trong top 3 (Hits@3)
- ○ **Rank 4-10**: Fair. Ground truth trong top 10 (Hits@10)
- ✗ **Rank > 10**: Ground truth nằm ngoài top 10

### Score Interpretation

- **Score cao (> 0.9)**: Model rất tự tin về prediction này
- **Score trung bình (0.7-0.9)**: Model khá tự tin
- **Score thấp (< 0.7)**: Model không chắc chắn

### Markers

- **✓✓✓ [CORRECT]**: Đánh dấu ground truth entity trong danh sách predictions

## Model Types

Tool hỗ trợ các model types:

| Model Type | Directory | Scoring Function |
|------------|-----------|------------------|
| **DistMult** | `fuselinker/` | `score = <h, r, t>` |
| **ComplEx** | `fuselinker-complex/` | Complex embeddings |
| **TransE** | `fuselinker-transe/` | `score = -‖h + r - t‖` |
| **ConvE** | `fuselinker-conve/` | Convolutional |

**Note**: Tool hiện tại dùng DistMult scoring function cho tất cả models. Để có kết quả chính xác nhất, cần load proper scoring function cho từng model type.

## Files cần thiết

Tool cần các files sau:

```
model_dir/
└── data/
    ├── entity2index.pkl
    ├── index2entity.pkl
    ├── relation2index.pkl
    ├── index2relation.pkl
    └── visualization_outputs/
        ├── node_embeddings.npy
        ├── relation_embeddings.npy
        ├── train_graph.json
        └── test_graph.json
```

Các files này được tạo tự động khi:
1. Train model với `main.py`
2. Visualization outputs được export sau khi training

## Ví dụ đầy đủ

### Workflow: Train → Demo

```bash
# Step 1: Train model (QUAN TRỌNG - phải train model trước!)
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python main.py \
    --data primekg \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --use_n3_reg \
    --model_state_file primekg_complex_model.pth  # ⚠️ Tên file này quan trọng!

# Step 2: Demo predictions với TRAINED MODEL
cd ~/fussion-and-verify-in-BKG

# Demo random samples từ TEST SET
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --model_state_file primekg_complex_model.pth \
    --num_samples 10 \
    --top_k 10

# Demo specific triple
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --model_state_file primekg_complex_model.pth \
    --head "Gene_APOE" \
    --relation "ppi" \
    --tail "Gene_APP" \
    --top_k 20
```

## Troubleshooting

### Lỗi: "Model state file not found"

**Nguyên nhân**: Chưa chỉ định đúng path đến file .pth hoặc chưa train model

**Giải pháp**:
```bash
# Kiểm tra file .pth có tồn tại không
ls -la primekg_complex_model.pth

# Hoặc tìm trong thư mục
find . -name "*.pth"

# Nếu chưa có file .pth, bạn PHẢI train model trước!
cd fuselinker-complex
python main.py --data primekg ... --model_state_file primekg_model.pth
```

### Lỗi: "Could not find entities/relation"

**Nguyên nhân**: Entity hoặc relation name không tồn tại trong data

**Giải pháp**: Kiểm tra tên chính xác trong pkl files:

```python
import pickle

# Check entity names
with open('fuselinker-complex/primekg/entity2index.pkl', 'rb') as f:
    entity2index = pickle.load(f)
    print("Sample entities:", list(entity2index.keys())[:10])

# Check relation names
with open('fuselinker-complex/primekg/relation2index.pkl', 'rb') as f:
    relation2index = pickle.load(f)
    print("Relations:", list(relation2index.keys()))
```

### Lỗi: "No test triples available"

**Nguyên nhân**: Không tìm thấy test data

**Giải pháp**:
- Đảm bảo file `test.tsv` hoặc `visualization_outputs/test_graph.json` tồn tại
- Train model sẽ tự động tạo visualization outputs

### Demo chạy chậm

**Nguyên nhân**: Scoring tất cả entities mất thời gian với dataset lớn (vd: 14,903 entities)

**Giải pháp**:
- Giảm `--num_samples` xuống 3-5
- Chỉ demo specific triples thay vì random samples
- Dùng `--use_cuda True` nếu có GPU

## Best Practices

### 1. Kiểm tra model quality
```bash
# Demo nhiều samples để có overview
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --num_samples 20 \
    --top_k 10
```

### 2. Debug specific relationships
```bash
# Focus vào một relationship type
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --relation "ppi" \
    --num_samples 10
```

### 3. Discover novel predictions
```bash
# Xem top 50 predictions
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --head "Disease_Alzheimer" \
    --relation "associated_with" \
    --top_k 50
```

## Summary

⚠️ **YÊU CẦU**: Phải có file .pth (trained model weights)!

✅ **Quick demo**:
```bash
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --model_state_file primekg_model.pth
```

✅ **Tính năng**:
- ✓ Load TRAINED model weights (.pth file)
- ✓ Dùng model's scoring function (ComplEx, DistMult, TransE, ConvE)
- ✓ Random samples LẤY TỪ TEST SET THẬT (không bịa!)
- ✓ Hiển thị rank của đáp án đúng
- ✓ Filtered evaluation (loại known triples)

✅ **Rank indicators**: 🎉 (rank 1), ✓ (top 3), ○ (top 10), ✗ (>10)

✅ **Files cần thiết**:
- **Model state file (.pth)** - BẮT BUỘC
- entity2index.pkl, index2entity.pkl
- relation2index.pkl, index2relation.pkl
- test.tsv hoặc visualization_outputs/test_graph.json

**Cách đơn giản nhất để kiểm tra model:**
```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG

# Tìm file .pth của bạn
find . -name "*.pth"

# Chạy demo (thay YOUR_MODEL.pth bằng tên file thật)
python demo_link_prediction.py \
    --model_dir fuselinker-complex \
    --data primekg \
    --model_state_file YOUR_MODEL.pth
```

Tool sẽ hiển thị 5 **RANDOM TEST TRIPLES THẬT** với predictions và **rank của đáp án đúng**! 🎯

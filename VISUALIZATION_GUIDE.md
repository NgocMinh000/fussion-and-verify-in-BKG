# Hướng dẫn Visualize Link Predictions

## Quick Start - Cách nhanh nhất

### Dùng script tự động (Khuyến nghị)

```bash
# Activate environment
conda activate fuselinker

# Chạy visualization cho model của bạn
cd ~/fussion-and-verify-in-BKG

./visualize_predictions.sh primekg complex
```

Script sẽ tự động:
1. ✓ Kiểm tra files cần thiết
2. ✓ Extract link predictions
3. ✓ Launch dashboard tại http://localhost:5000

### Dùng lệnh manual

```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

# Bước 1: Extract predictions
python -m visualization.link_predictor \
    --model_dir . \
    --data_dir primekg/visualization_outputs \
    --output predictions_complex.json \
    --top_k 10

# Bước 2: Launch dashboard
python -m visualization.app \
    --data_dir primekg/visualization_outputs \
    --port 5000

# Bước 3: Mở browser tại http://localhost:5000
```

## Chi tiết từng bước

### Bước 1: Kiểm tra files đã có

Sau khi train model, kiểm tra xem visualization data đã được export:

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

ls -la primekg/visualization_outputs/
```

**Cần có các files:**
```
primekg/visualization_outputs/
├── node_embeddings.npy
├── relation_embeddings.npy
├── train_graph.json
└── test_graph.json

primekg/
├── entity2index.pkl
├── index2entity.pkl
├── relation2index.pkl
└── index2relation.pkl
```

### Bước 2: Extract Link Predictions

```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python -m visualization.link_predictor \
    --model_dir . \
    --data_dir primekg/visualization_outputs \
    --output predictions_complex.json \
    --top_k 10
```

**Output:**
```
Loading model and data...
✓ Loaded model from: suppkg_model_state.pth
✓ Loaded entity embeddings: (14903, 200)
✓ Loaded relation embeddings: (16, 200)
✓ Loaded train triples: 125488
✓ Loaded test triples: 15686

Extracting predictions...
Processing test triples: 100%|███████| 15686/15686 [02:15<00:00, 115.67it/s]

✓ Extracted predictions for 15686 test triples
✓ Saved predictions to: predictions_complex.json
```

### Bước 3: Launch Dashboard

```bash
python -m visualization.app \
    --data_dir primekg/visualization_outputs \
    --port 5000
```

**Output:**
```
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Bước 4: Mở Browser

Mở browser và truy cập:
```
http://localhost:5000
```

Hoặc nếu chạy trên server remote:
```
http://your-server-ip:5000
```

## Sử dụng Script Tự Động

### Cú pháp cơ bản

```bash
./visualize_predictions.sh <data_dir> <model_type> [top_k] [--dashboard-only]
```

### Ví dụ

#### 1. Visualize ComplEx model với primekg data
```bash
./visualize_predictions.sh primekg complex
```

#### 2. Visualize DistMult với suppkg data
```bash
./visualize_predictions.sh suppkg distmult
```

#### 3. Extract top 20 predictions
```bash
./visualize_predictions.sh primekg complex 20
```

#### 4. Chỉ launch dashboard (không extract lại predictions)
```bash
./visualize_predictions.sh primekg complex --dashboard-only
```

### Model types hỗ trợ

- `distmult` → Chạy trong `fuselinker/`
- `transe` → Chạy trong `fuselinker-transe/`
- `complex` → Chạy trong `fuselinker-complex/`
- `conve` → Chạy trong `fuselinker-conve/`

## Visualize nhiều models để so sánh

### Extract predictions từ tất cả models

```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG

# DistMult
./visualize_predictions.sh primekg distmult

# TransE
./visualize_predictions.sh primekg transe

# ComplEx
./visualize_predictions.sh primekg complex

# ConvE
./visualize_predictions.sh primekg conve
```

### So sánh predictions

```bash
# Xem predictions của từng model
cd fuselinker
python -c "
import json
with open('predictions_distmult_primekg.json', 'r') as f:
    data = json.load(f)
    print('DistMult Predictions:', len(data['predictions']))
    print('Sample:', data['predictions'][0])
"

cd ../fuselinker-complex
python -c "
import json
with open('predictions_complex_primekg.json', 'r') as f:
    data = json.load(f)
    print('ComplEx Predictions:', len(data['predictions']))
    print('Sample:', data['predictions'][0])
"
```

## Dashboard Features

Khi dashboard chạy, bạn sẽ thấy các tính năng:

### 1. Link Predictions View
- Xem top-K predictions cho mỗi test triple
- Filter theo entity, relation, score
- Xem ground truth và predicted entities

### 2. Embedding Visualization
- Visualize entity embeddings (t-SNE/UMAP)
- Cluster entities theo relations
- Interactive plot với zoom/pan

### 3. Graph Structure
- Xem knowledge graph structure
- Node và edge visualization
- Filter theo relation types

### 4. Prediction Explanations
- Tại sao model predict entity này?
- Attention weights
- Similar entities

## Options cho link_predictor

```bash
python -m visualization.link_predictor --help
```

**Arguments:**
- `--model_dir`: Thư mục chứa model state file
- `--data_dir`: Thư mục chứa visualization outputs
- `--output`: Output JSON file cho predictions
- `--top_k`: Số predictions cho mỗi triple (default: 10)
- `--batch_size`: Batch size (default: 50)

**Ví dụ:**
```bash
# Extract top 20 predictions với batch size 100
python -m visualization.link_predictor \
    --model_dir . \
    --data_dir primekg/visualization_outputs \
    --output predictions_top20.json \
    --top_k 20 \
    --batch_size 100
```

## Options cho app

```bash
python -m visualization.app --help
```

**Arguments:**
- `--data_dir`: Thư mục chứa visualization outputs (required)
- `--host`: Host address (default: 127.0.0.1)
- `--port`: Port number (default: 5000)
- `--debug`: Enable debug mode

**Ví dụ:**
```bash
# Chạy trên port 8080 với debug mode
python -m visualization.app \
    --data_dir primekg/visualization_outputs \
    --port 8080 \
    --debug

# Allow remote access
python -m visualization.app \
    --data_dir primekg/visualization_outputs \
    --host 0.0.0.0 \
    --port 5000
```

## Troubleshooting

### Lỗi: "No module named 'torch'"

**Nguyên nhân:** Chưa activate environment

**Giải pháp:**
```bash
conda activate fuselinker
# Sau đó chạy lại commands
```

### Lỗi: "visualization_outputs not found"

**Nguyên nhân:** Model chưa export visualization data

**Giải pháp:** Train lại model, visualization data sẽ được export tự động:
```bash
python main.py --data primekg ... --iterations 4000
```

### Lỗi: "app.py: error: the following arguments are required: --data_dir"

**Nguyên nhân:** Thiếu argument --data_dir

**Giải pháp:**
```bash
# Thêm --data_dir argument
python -m visualization.app --data_dir primekg/visualization_outputs
```

### Lỗi: "Address already in use"

**Nguyên nhân:** Port 5000 đang được dùng

**Giải pháp:** Dùng port khác:
```bash
python -m visualization.app \
    --data_dir primekg/visualization_outputs \
    --port 5001
```

### Dashboard không load được

**Giải pháp 1:** Kiểm tra firewall
```bash
# Nếu chạy trên server remote
sudo ufw allow 5000
```

**Giải pháp 2:** Allow external access
```bash
python -m visualization.app \
    --data_dir primekg/visualization_outputs \
    --host 0.0.0.0 \
    --port 5000
```

### Predictions extraction quá chậm

**Giải pháp:** Tăng batch size
```bash
python -m visualization.link_predictor \
    --model_dir . \
    --data_dir primekg/visualization_outputs \
    --output predictions.json \
    --batch_size 200  # Tăng từ 50 lên 200
```

## Workflow đầy đủ

### Từ training đến visualization

```bash
# Activate environment
conda activate fuselinker

# Step 1: Train model
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
    --model_state_file primekg_complex_model.pth

# Step 2: Extract predictions (tự động dùng script)
cd ~/fussion-and-verify-in-BKG
./visualize_predictions.sh primekg complex 10

# Step 3: Browser sẽ mở dashboard tại http://localhost:5000
```

### Chạy manual (không dùng script)

```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

# Extract predictions
python -m visualization.link_predictor \
    --model_dir . \
    --data_dir primekg/visualization_outputs \
    --output predictions_complex.json \
    --top_k 10

# Launch dashboard
python -m visualization.app \
    --data_dir primekg/visualization_outputs \
    --port 5000

# Mở browser: http://localhost:5000
```

## Output Files

### predictions_*.json format

```json
{
  "metadata": {
    "model_type": "ComplEx",
    "data_dir": "primekg",
    "num_test_triples": 15686,
    "top_k": 10,
    "timestamp": "2026-01-16T12:00:00"
  },
  "predictions": [
    {
      "test_triple": ["entity_123", "relation_5", "entity_456"],
      "test_triple_names": ["Gene_APOE", "ppi", "Gene_APP"],
      "predicted_heads": [
        {"entity_id": 789, "entity_name": "Gene_PSEN1", "score": 0.892},
        {"entity_id": 234, "entity_name": "Gene_MAPT", "score": 0.856},
        ...
      ],
      "predicted_tails": [
        {"entity_id": 567, "entity_name": "Gene_BACE1", "score": 0.923},
        {"entity_id": 890, "entity_name": "Gene_GSK3B", "score": 0.901},
        ...
      ]
    },
    ...
  ]
}
```

## Ví dụ với các datasets khác nhau

### Với suppkg (dataset nhỏ)

```bash
# Extract và visualize
./visualize_predictions.sh suppkg complex

# Manual
cd fuselinker-complex
python -m visualization.link_predictor \
    --model_dir . \
    --data_dir suppkg/visualization_outputs \
    --output predictions_suppkg.json \
    --top_k 10

python -m visualization.app --data_dir suppkg/visualization_outputs
```

### Với primekg (dataset lớn)

```bash
# Extract với batch size lớn hơn
cd fuselinker-complex
python -m visualization.link_predictor \
    --model_dir . \
    --data_dir primekg/visualization_outputs \
    --output predictions_primekg.json \
    --top_k 10 \
    --batch_size 200

python -m visualization.app --data_dir primekg/visualization_outputs
```

### Với mybkg (custom dataset)

```bash
# Sau khi train với mybkg data
./visualize_predictions.sh mybkg_converted complex

# Hoặc manual
cd fuselinker-complex
python -m visualization.link_predictor \
    --model_dir . \
    --data_dir mybkg_converted/visualization_outputs \
    --output predictions_mybkg.json \
    --top_k 10

python -m visualization.app --data_dir mybkg_converted/visualization_outputs
```

## Performance Tips

### Tăng tốc extraction

```bash
# Tăng batch size (cần GPU memory)
--batch_size 200

# Giảm top_k nếu không cần nhiều predictions
--top_k 5
```

### Dashboard performance

```bash
# Nếu dashboard chậm với dataset lớn
# Giảm số predictions được load

# Extract ít predictions hơn
python -m visualization.link_predictor ... --top_k 5

# Hoặc sample test triples trước khi extract
```

## Summary

✅ **Script tự động**: `./visualize_predictions.sh primekg complex`

✅ **Manual steps**:
1. `python -m visualization.link_predictor --model_dir . --data_dir primekg/visualization_outputs --output predictions.json`
2. `python -m visualization.app --data_dir primekg/visualization_outputs`
3. Mở http://localhost:5000

✅ **Requirements**:
- Activate `fuselinker` environment
- Có visualization_outputs/ từ training
- Có pkl files (entity2index, etc.)

✅ **Model types**: distmult, transe, complex, conve

**Cách đơn giản nhất:**
```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG
./visualize_predictions.sh primekg complex
```

Dashboard sẽ mở tại http://localhost:5000 🎉

# 📁 Cấu Trúc Thư Mục và Hướng Dẫn Visualization

## 🏗️ Cấu Trúc Thư Mục

```
fussion-and-verify-in-BKG/                    (ROOT)
│
├── fuselinker/                               (DistMult implementation)
│   ├── main.py
│   ├── model.py
│   ├── myutils.py
│   └── suppkg/                               (Data cho DistMult)
│       ├── train.tsv
│       ├── test.tsv
│       ├── entity2index.pkl
│       ├── relation2index.pkl
│       └── visualization_outputs/            (Tạo sau khi train)
│           ├── entity_embeddings.npy
│           ├── relation_embeddings.npy
│           └── ...
│
├── fuselinker-transe/                        (TransE implementation)
│   ├── main.py
│   └── suppkg/                               (Data cho TransE)
│       └── visualization_outputs/
│
├── fuselinker-complex/                       (ComplEx implementation)
│   ├── main.py
│   └── suppkg/                               (Data cho ComplEx)
│       └── visualization_outputs/
│
├── fuselinker-conve/                         (ConvE implementation)
│   ├── main.py
│   └── suppkg/                               (Data cho ConvE)
│       └── visualization_outputs/
│
└── visualization/                            (Shared visualization tools)
    ├── __init__.py
    ├── export_utils.py                       (Export từ model)
    ├── embedding_visualizer.py               (Visualize embeddings)
    ├── graph_visualizer.py                   (Visualize graph structure)
    ├── link_predictor.py                     (Extract predictions)
    ├── explainer.py                          (Explain predictions)
    ├── prediction_visualizer.py              (Visualize predictions)
    ├── prediction_app.py                     (Interactive dashboard)
    └── app.py                                (Embedding dashboard)
```

---

## 🔑 Nguyên Tắc Quan Trọng

### 1. **Mỗi Model Variant Có Data Riêng**

- **DistMult** → `fuselinker/suppkg/`
- **TransE** → `fuselinker-transe/suppkg/`
- **ComplEx** → `fuselinker-complex/suppkg/`
- **ConvE** → `fuselinker-conve/suppkg/`

### 2. **Visualization Tools Ở Root Level**

Tất cả scripts visualization ở `/visualization/` (shared cho tất cả models).

### 3. **Chạy Commands Từ ROOT Directory**

**✅ ĐÚNG:**
```bash
cd ~/fussion-and-verify-in-BKG                    # ROOT directory
python -m visualization.link_predictor \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs
```

**❌ SAI:**
```bash
cd ~/fussion-and-verify-in-BKG/fuselinker-complex  # Sai!
python -m visualization.link_predictor \
    --viz_dir suppkg/visualization_outputs         # Không tìm thấy module
```

### 4. **Paths Phải Chính Xác**

Khi specify paths, luôn dùng relative path từ ROOT:
- ✅ `fuselinker-complex/suppkg/visualization_outputs`
- ✅ `fuselinker-transe/suppkg/link_predictions.json`
- ❌ `suppkg/visualization_outputs` (thiếu prefix)

---

## 📝 Workflow Chuẩn

### Step 1: Train Model

```bash
cd ~/fussion-and-verify-in-BKG

# Train ComplEx model
cd fuselinker-complex
python main.py --data suppkg --iterations 4000 --w 0.75 --use_cuda True

# Model sẽ tự động export visualization data vào:
# fuselinker-complex/suppkg/visualization_outputs/
```

✅ **Check xem data đã được export chưa:**
```bash
ls fuselinker-complex/suppkg/visualization_outputs/
# Phải thấy: entity_embeddings.npy, relation_embeddings.npy, etc.
```

### Step 2: Quay Về ROOT Directory

```bash
cd ~/fussion-and-verify-in-BKG    # QUAN TRỌNG: Phải ở ROOT!
```

### Step 3: Run Visualization

#### A. Embedding Visualizations

```bash
python -m visualization.embedding_visualizer \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_dir fuselinker-complex/suppkg/visualization_plots
```

#### B. Graph Visualizations

```bash
python -m visualization.graph_visualizer \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_dir fuselinker-complex/suppkg/visualization_plots
```

#### C. Link Predictions

```bash
python -m visualization.link_predictor \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_file fuselinker-complex/suppkg/link_predictions.json \
    --model_type complex \
    --num_queries 100
```

#### D. Explanations

```bash
python -m visualization.explainer \
    --prediction_file fuselinker-complex/suppkg/link_predictions.json \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_file fuselinker-complex/suppkg/prediction_explanations.json
```

#### E. Prediction Visualizations

```bash
python -m visualization.prediction_visualizer \
    --prediction_file fuselinker-complex/suppkg/link_predictions.json \
    --output_dir fuselinker-complex/suppkg/prediction_plots
```

#### F. Interactive Dashboards

```bash
# Dashboard cho embeddings
streamlit run visualization/app.py

# Dashboard cho predictions
streamlit run visualization/prediction_app.py
```

---

## 🔄 So Sánh Nhiều Models

### Train Tất Cả Models

```bash
cd ~/fussion-and-verify-in-BKG

# Train DistMult
cd fuselinker
python main.py --data suppkg --iterations 39166 --w 0.75
cd ..

# Train TransE
cd fuselinker-transe
python main.py --data suppkg --iterations 39166 --w 0.75
cd ..

# Train ComplEx
cd fuselinker-complex
python main.py --data suppkg --iterations 39166 --w 0.75 --use_n3_reg
cd ..

# Train ConvE
cd fuselinker-conve
python main.py --data suppkg --iterations 39166 --w 0.75
cd ..
```

### Extract Predictions Từ Tất Cả

```bash
cd ~/fussion-and-verify-in-BKG    # ROOT directory

# DistMult
python -m visualization.link_predictor \
    --viz_dir fuselinker/suppkg/visualization_outputs \
    --output_file fuselinker/suppkg/link_predictions.json \
    --model_type distmult

# TransE
python -m visualization.link_predictor \
    --viz_dir fuselinker-transe/suppkg/visualization_outputs \
    --output_file fuselinker-transe/suppkg/link_predictions.json \
    --model_type transe

# ComplEx
python -m visualization.link_predictor \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_file fuselinker-complex/suppkg/link_predictions.json \
    --model_type complex

# ConvE
python -m visualization.link_predictor \
    --viz_dir fuselinker-conve/suppkg/visualization_outputs \
    --output_file fuselinker-conve/suppkg/link_predictions.json \
    --model_type conve
```

### So Sánh Kết Quả

```bash
# Compare metrics
cat fuselinker/suppkg/link_predictions.json | grep -A 5 "statistics"
cat fuselinker-transe/suppkg/link_predictions.json | grep -A 5 "statistics"
cat fuselinker-complex/suppkg/link_predictions.json | grep -A 5 "statistics"
cat fuselinker-conve/suppkg/link_predictions.json | grep -A 5 "statistics"
```

---

## 🐛 Common Mistakes

### ❌ Mistake 1: Chạy từ sai directory

```bash
cd fuselinker-complex    # SAI!
python -m visualization.link_predictor    # Module not found!
```

**✅ Fix:**
```bash
cd ~/fussion-and-verify-in-BKG    # ĐÚNG!
python -m visualization.link_predictor \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs
```

### ❌ Mistake 2: Sai path

```bash
python -m visualization.link_predictor \
    --viz_dir suppkg/visualization_outputs    # Không tìm thấy!
```

**✅ Fix:**
```bash
python -m visualization.link_predictor \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs    # Có prefix!
```

### ❌ Mistake 3: Model chưa train

```bash
python -m visualization.link_predictor \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs    # Directory not found!
```

**✅ Fix:**
```bash
# Train model trước
cd fuselinker-complex
python main.py --data suppkg --iterations 4000 --w 0.75
cd ..

# Bây giờ visualization_outputs đã tồn tại
python -m visualization.link_predictor \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs
```

---

## 📖 Quick Reference Commands

### Complete Workflow (ComplEx Example)

```bash
# 1. Go to ROOT
cd ~/fussion-and-verify-in-BKG

# 2. Train model (nếu chưa)
cd fuselinker-complex
python main.py --data suppkg --iterations 4000 --w 0.75 --use_cuda True
cd ..

# 3. Verify export
ls fuselinker-complex/suppkg/visualization_outputs/

# 4. Extract predictions
python -m visualization.link_predictor \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_file fuselinker-complex/suppkg/link_predictions.json \
    --model_type complex \
    --num_queries 100

# 5. Generate explanations
python -m visualization.explainer \
    --prediction_file fuselinker-complex/suppkg/link_predictions.json \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_file fuselinker-complex/suppkg/prediction_explanations.json \
    --num_queries 10

# 6. Create visualizations
python -m visualization.embedding_visualizer \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_dir fuselinker-complex/suppkg/visualization_plots

python -m visualization.graph_visualizer \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_dir fuselinker-complex/suppkg/visualization_plots

python -m visualization.prediction_visualizer \
    --prediction_file fuselinker-complex/suppkg/link_predictions.json \
    --output_dir fuselinker-complex/suppkg/prediction_plots

# 7. Launch dashboard
streamlit run visualization/prediction_app.py
# Nhập: fuselinker-complex/suppkg/link_predictions.json
```

---

## ✅ Checklist

- [ ] Hiểu cấu trúc thư mục (mỗi variant có `suppkg/` riêng)
- [ ] Luôn chạy commands từ ROOT directory (`~/fussion-and-verify-in-BKG`)
- [ ] Paths luôn có prefix (`fuselinker-complex/suppkg/...`)
- [ ] Model đã train và export visualization data
- [ ] Có thể extract predictions và explanations
- [ ] Có thể tạo visualizations và launch dashboards

---

## 🎯 TL;DR

**1. Cấu trúc:**
- ROOT: `/fussion-and-verify-in-BKG`
- Models: `fuselinker/`, `fuselinker-transe/`, `fuselinker-complex/`, `fuselinker-conve/`
- Data: `<model>/suppkg/`
- Tools: `visualization/` (shared)

**2. Workflow:**
- Train từ model directory: `cd fuselinker-complex && python main.py`
- Visualize từ ROOT directory: `cd ~/fussion-and-verify-in-BKG && python -m visualization.link_predictor`

**3. Paths:**
- Luôn dùng full relative path từ ROOT
- Example: `fuselinker-complex/suppkg/visualization_outputs`

**Happy Visualizing! 🎉**

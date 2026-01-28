# 🎨 Visualization Quick Start Guide

## ✅ Hệ Thống Visualization Đã Hoàn Thành

Tôi đã tạo một hệ thống visualization hoàn chỉnh để phân tích embeddings và cấu trúc knowledge graph từ FuseLinker.

---

## 📦 Các Thành Phần Đã Tạo

### 1. **Export Utilities** (`visualization/export_utils.py`)
- Tự động export dữ liệu visualization từ model sau khi training
- Export embeddings (entity, relation, text, domain, learned)
- Export graph structure và statistics
- **Tích hợp sẵn trong main.py** - tự động chạy sau evaluation

### 2. **Embedding Visualizer** (`visualization/embedding_visualizer.py`)
- Visualize embeddings bằng t-SNE và PCA
- So sánh các thành phần fusion (text, domain, learned, fused)
- Tạo plots tĩnh (PNG files)
- Hỗ trợ sampling cho large datasets

### 3. **Graph Visualizer** (`visualization/graph_visualizer.py`)
- Visualize graph structure và topology
- Relation distribution (bar charts)
- Entity degree distribution
- Sample subgraph visualization
- Train/test split statistics

### 4. **Interactive Dashboard** (`visualization/app.py`)
- Web interface với Streamlit
- Explore embeddings interactively
- Compare fusion components
- View statistics và metrics
- Real-time filtering và sampling

### 5. **Documentation** (`visualization/README.md`)
- Hướng dẫn sử dụng chi tiết
- Command-line examples
- Troubleshooting guide
- Interpretation guidelines

---

## 🚀 Cách Sử Dụng Nhanh

### Bước 1: Train Model (Đã Có Kết Quả Rồi!)

Model của bạn đã train và đã export visualization data vào thư mục tương ứng:
```
fuselinker-complex/suppkg/visualization_outputs/    (ComplEx model)
├── entity_embeddings.npy
├── relation_embeddings.npy
├── learned_embeddings.npy
├── text_embeddings.npy
├── domain_embeddings.npy
├── graph_structure.json
├── statistics.json
└── model_config.json
```

**Lưu ý:** Mỗi variant có data riêng:
- `fuselinker/suppkg/` → DistMult
- `fuselinker-transe/suppkg/` → TransE
- `fuselinker-complex/suppkg/` → ComplEx
- `fuselinker-conve/suppkg/` → ConvE

✓ **Data đã sẵn sàng để visualize!**

### Bước 2: Tạo Static Visualizations

```bash
# Activate conda environment
conda activate fuselinker

# Visualize embeddings (t-SNE + PCA) cho ComplEx model
cd ~/fussion-and-verify-in-BKG
python -m visualization.embedding_visualizer \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_dir fuselinker-complex/suppkg/visualization_plots \
    --method both \
    --sample_size 1000

# Visualize graph structure
python -m visualization.graph_visualizer \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_dir fuselinker-complex/suppkg/visualization_plots \
    --sample_nodes 50 \
    --sample_edges 100
```

✓ **Plots được lưu tại: `fuselinker-complex/suppkg/visualization_plots/`**

### Bước 3: Xem Interactive Dashboard

```bash
# Install Streamlit (nếu chưa có)
pip install streamlit plotly

# Launch dashboard
streamlit run visualization/app.py
```

✓ **Dashboard mở tại: http://localhost:8501**

---

## 📊 Output Files

### Embedding Plots (từ embedding_visualizer.py):

1. **entity_embeddings_tsne.png** - Entity embeddings (t-SNE projection)
2. **entity_embeddings_pca.png** - Entity embeddings (PCA projection)
3. **relation_embeddings_tsne.png** - Relation embeddings (t-SNE)
4. **relation_embeddings_pca.png** - Relation embeddings (PCA)

**Fusion Component Comparison:**
5. **text_embeddings_pca.png** - Text embeddings (PubMedBERT)
6. **domain_embeddings_pca.png** - Domain embeddings (Poincaré)
7. **learned_embeddings_pca.png** - Learned embeddings
8. **fused_embeddings_pca.png** - Final fused embeddings

### Graph Plots (từ graph_visualizer.py):

1. **relation_distribution.png** - Relation frequency distribution
2. **entity_degree_distribution.png** - Entity connectivity
3. **train_test_split.png** - Data split statistics
4. **graph_sample.png** - Sample subgraph visualization

---

## 💡 Ý Nghĩa Các Visualizations

### 1. Entity Embeddings (t-SNE/PCA)
**Xem để:**
- Kiểm tra xem entities có cluster tốt không
- Các loại entity khác nhau (drugs, diseases) có tách biệt không
- Outliers (entities bất thường)

**Ví dụ tốt:**
- Drug entities cluster gần nhau
- Disease entities cluster gần nhau
- Có separation rõ ràng giữa các nhóm

### 2. Fusion Component Comparison
**Xem để:**
- So sánh text vs domain embeddings
- Xem fusion có kết hợp tốt cả hai không
- Kiểm tra contribution của từng component

**Ý nghĩa:**
- **Text embeddings**: Semantic similarity (từ PubMedBERT)
- **Domain embeddings**: Hierarchical knowledge (từ Poincaré)
- **Fused embeddings**: Kết hợp cả hai → should be better!

### 3. Relation Distribution
**Xem để:**
- Kiểm tra data balanced hay skewed
- Relations nào xuất hiện nhiều nhất
- Với `--use_reciprocal`: forward/inverse có balanced không

**Ideal:**
- Không có relation nào quá dominant
- Forward và inverse relations có counts gần bằng nhau

### 4. Entity Degree Distribution
**Xem để:**
- Kiểm tra graph topology
- Tìm hub entities (degree cao)
- Hiểu connectivity pattern

**Typical:**
- Power-law distribution (một vài hubs, nhiều low-degree nodes)
- Hubs là entities quan trọng

---

## 🎯 Use Cases

### Use Case 1: Kiểm Tra Fusion Quality

```bash
# Generate fusion comparison plots
python -m visualization.embedding_visualizer \
    --viz_dir suppkg/visualization_outputs \
    --output_dir suppkg/fusion_analysis \
    --method pca

# Xem các plots:
# - text_embeddings_pca.png
# - domain_embeddings_pca.png
# - fused_embeddings_pca.png

# Kiểm tra: Fused embeddings có cluster tốt hơn text/domain riêng lẻ không?
```

### Use Case 2: So Sánh Các Models

```bash
# Train DistMult
cd fuselinker
python main.py --data suppkg --iterations 39166 ...
mv suppkg/visualization_outputs suppkg/viz_distmult

# Train ComplEx
cd ../fuselinker-complex
python main.py --data suppkg --iterations 39166 --use_n3_reg ...
mv suppkg/visualization_outputs suppkg/viz_complex

# Visualize both
python -m visualization.embedding_visualizer --viz_dir suppkg/viz_distmult --output_dir plots/distmult
python -m visualization.embedding_visualizer --viz_dir suppkg/viz_complex --output_dir plots/complex

# So sánh: Model nào có embeddings cluster tốt hơn?
```

### Use Case 3: Debug Poor Performance

Nếu model metrics thấp:

1. **Check entity embeddings** → Có clusters rõ ràng không?
2. **Check relation distribution** → Data có balanced không?
3. **Check degree distribution** → Có hubs quá dominant không?
4. **Check fusion components** → Text/domain có contribute đều không?

---

## 🔧 Advanced Options

### Fast Preview (PCA Only, Small Sample)

```bash
python -m visualization.embedding_visualizer \
    --method pca \
    --sample_size 500
```

⚡ **Nhanh nhất** (~10 seconds)

### Detailed Analysis (t-SNE, Large Sample)

```bash
python -m visualization.embedding_visualizer \
    --method tsne \
    --sample_size 2000
```

🐢 **Chậm hơn** (~2-5 minutes), nhưng chi tiết hơn

### Custom Paths

```bash
# Specify custom directories
python -m visualization.embedding_visualizer \
    --viz_dir path/to/visualization_outputs \
    --output_dir path/to/save/plots
```

---

## 📋 Checklist Sau Khi Train Model

- [ ] Model đã train xong
- [ ] Visualization data đã export (`suppkg/visualization_outputs/` tồn tại)
- [ ] Install dependencies: `pip install streamlit plotly sklearn`
- [ ] Generate static plots: `python -m visualization.embedding_visualizer`
- [ ] Generate graph plots: `python -m visualization.graph_visualizer`
- [ ] Launch dashboard: `streamlit run visualization/app.py`
- [ ] Analyze results và compare với baseline

---

## 🐛 Troubleshooting

### "No module named 'streamlit'"
```bash
pip install streamlit plotly
```

### "Directory not found: suppkg/visualization_outputs"
→ Model chưa train hoặc export failed. Re-train model.

### "Out of memory"
→ Giảm `--sample_size`:
```bash
python -m visualization.embedding_visualizer --sample_size 500
```

### t-SNE quá chậm
→ Dùng PCA thay vì t-SNE:
```bash
python -m visualization.embedding_visualizer --method pca
```

---

## 📚 Next Steps

### 1. Tạo Visualizations Ngay

```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG

# Generate all visualizations
python -m visualization.embedding_visualizer --viz_dir suppkg/visualization_outputs --output_dir suppkg/plots
python -m visualization.graph_visualizer --viz_dir suppkg/visualization_outputs --output_dir suppkg/plots

# Launch dashboard
streamlit run visualization/app.py
```

### 2. Phân Tích Kết Quả

Xem các plots và trả lời:
- Embeddings có cluster tốt không?
- Fusion có hiệu quả không?
- Relations có balanced không?
- Degree distribution như thế nào?

### 3. Compare Models

Train các models khác (DistMult, TransE, ConvE) và so sánh visualizations.

### 4. Optimize

Dựa vào visualizations để quyết định:
- Có nên tăng embedding dimension không?
- Có nên thay đổi fusion weight `--w` không?
- Có nên dùng `--use_reciprocal` không?

---

## 🎯 TÓM TẮT

**✅ Đã tạo:**
- Export utilities (tích hợp trong main.py)
- Embedding visualizer (t-SNE + PCA)
- Graph visualizer (structure + relations)
- Interactive dashboard (Streamlit)
- Comprehensive documentation

**✅ Có thể làm:**
- Visualize embeddings và graph structure
- Compare fusion components
- Analyze relation distribution
- Interactive exploration
- Compare different models

**✅ Bước tiếp theo:**
```bash
# Generate visualizations
python -m visualization.embedding_visualizer --viz_dir suppkg/visualization_outputs --output_dir suppkg/plots
python -m visualization.graph_visualizer --viz_dir suppkg/visualization_outputs --output_dir suppkg/plots

# Launch dashboard
streamlit run visualization/app.py
```

**🎨 Happy Visualizing!**

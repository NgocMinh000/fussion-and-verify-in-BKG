# 🚀 Complete Workflow: Từ Training đến Visualize Fused Links

## 📋 Tổng Quan

Document này mô tả **complete end-to-end workflow** từ khi train FuseLinker model cho đến khi visualize và analyze các liên kết đã được fuse.

---

## 🔄 Workflow Overview

```
┌─────────────────┐
│  1. TRAINING    │  Train FuseLinker model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. EXPORT      │  Export embeddings & graphs (tự động)
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│ 3a. VISUALIZE   │  │ 3b. PREDICT     │
│   Embeddings    │  │   New Links     │
│   & Graphs      │  │                 │
└─────────────────┘  └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ 4. ANALYZE      │
                     │   Fused Links   │
                     └─────────────────┘
```

---

## 📝 Bước 1: Training Model

### Command

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
    --model_state_file suppkg_model_state.pth
```

### Output Files

Sau khi training xong, bạn sẽ có:

```
fuselinker/
├── suppkg_model_state.pth          # ✓ Trained model
└── suppkg/
    ├── entity2index.pkl
    ├── index2entity.pkl
    ├── relation2index.pkl
    ├── index2relation.pkl
    └── visualization_outputs/      # ✓ Auto-exported data
        ├── node_embeddings.npy
        ├── node_embeddings.meta.json
        ├── relation_embeddings.npy
        ├── train_graph.json
        ├── test_graph.json
        └── mappings.json
```

### Training Output

```
Start training...
Epoch 1000 | Loss 0.12345
Epoch 2000 | Loss 0.11234
...
Epoch 40000 | Loss 0.08765

Evaluating...
MR: 125.34
MRR: 0.6789
Hits @ 1 = 0.5234
Hits @ 3 = 0.7891
Hits @ 10 = 0.9012

============================================================
Exporting data for visualization...
============================================================
[1/5] Exporting node embeddings...
✓ Saved embeddings to suppkg/visualization_outputs/node_embeddings.npy
...
✓ All visualization data exported to: suppkg/visualization_outputs

To visualize:
  1. Graph structure: python -m visualization.graph_visualizer
  2. Embeddings: python -m visualization.embedding_visualizer
  3. Dashboard: python -m visualization.app

Training done!
```

---

## 📊 Bước 2: Visualize Embeddings & Graphs

### Option 2A: Interactive Dashboard

```bash
python -m visualization.app --data_dir suppkg/visualization_outputs
```

Mở browser: **http://localhost:8050**

**Features:**
- Tab 1: Interactive graph structure
- Tab 2: 2D/3D embedding visualization
- Filters: Semantic types, relations, confidence
- Real-time updates

### Option 2B: Quick Graph Visualization

```python
from visualization.graph_visualizer import visualize_graph

# Visualize training graph
viz = visualize_graph(
    json_path='suppkg/visualization_outputs/train_graph.json',
    max_nodes=1000,
    semantic_types=['dsyn', 'phsu'],  # diseases & drugs
    relations=['TREATS', 'CAUSES'],
    show=True
)
```

Output: **graph.html** - Interactive network graph

### Option 2C: Quick Embedding Visualization

```python
from visualization.embedding_visualizer import visualize_embeddings

# Visualize embeddings in 2D
viz = visualize_embeddings(
    embedding_path='suppkg/visualization_outputs/node_embeddings.npy',
    method='umap',
    n_components=2,
    color_by='semantic_type',
    show=True
)
```

Output: **embeddings.html** - Interactive scatter plot

---

## 🔗 Bước 3: Predict NEW Fused Links

**Đây là bước quan trọng để trả lời câu hỏi: "Liên kết nào đã được fuse?"**

### Method 1: CLI Script (Khuyến Nghị)

```bash
python predict_new_links.py \
    --model suppkg_model_state.pth \
    --data suppkg \
    --top_k 100 \
    --min_score 0.7 \
    --output predicted_links.csv
```

### Output

```
============================================================
FUSELINKER LINK PREDICTION
============================================================
Model: suppkg_model_state.pth
Data: suppkg
Top K per relation: 100
Min score: 0.7
============================================================

Loading FuseLinker Model...
✓ Entities: 9000
✓ Relations: 15
✓ Existing links: 305986

Predicting links for relation 0...
100%|███████████████████████| 9000/9000

...

============================================================
✓ Generated 1500 new link predictions
============================================================

TOP 10 PREDICTED LINKS (New Fused Links)
============================================================

[1] Score: 0.9234
    C0003250_aapp (aapp)
      --[INTERACTS_WITH]-->
    C0011503_nnon (nnon)

[2] Score: 0.9156
    C0035179_orch (orch)
      --[TREATS]-->
    C0233488_dsyn (dsyn)

...

✓ Exported 1500 predictions to predicted_links.csv
```

### File Output: predicted_links.csv

| subject | relation | object | score | subject_type | object_type |
|---------|----------|--------|-------|--------------|-------------|
| C0003250_aapp | INTERACTS_WITH | C0011503_nnon | 0.9234 | aapp | nnon |
| C0035179_orch | TREATS | C0233488_dsyn | 0.9156 | orch | dsyn |
| C0033554_bacs | AFFECTS | C0011603_dsyn | 0.8987 | bacs | dsyn |
| ... | ... | ... | ... | ... | ... |

**Đây chính là các FUSED LINKS mới!** - Các liên kết KHÔNG có trong training/test data nhưng model predict với high confidence.

### Method 2: Python Script

```python
from visualization.link_predictor import LinkPredictor

predictor = LinkPredictor()

# Load model
predictor.load_model(
    model_state_path='suppkg_model_state.pth',
    data_dir='suppkg',
    text_embedding_path='suppkg/pubmedbert_pretrained_embeddings_768.npy',
    knowledge_embedding_path='suppkg/poincare_embeddings.npy',
    n_hidden=200,
    num_hidden_layers=2,
    w=0.75
)

# Predict
predictions = predictor.predict_new_links(
    top_k_per_relation=100,
    min_score=0.7
)

# Export
predictor.export_predictions('predicted_links.csv')

# Analyze
stats = predictor.analyze_predictions()
print(f"Total new links: {stats['total_predictions']}")
print(f"Score range: {stats['score_min']:.4f} - {stats['score_max']:.4f}")
```

---

## 🔍 Bước 4: Analyze & Filter Predicted Links

### Filter by Disease-Drug Treatment

```python
from visualization.link_predictor import LinkPredictor

predictor = LinkPredictor()
predictor.load_model(...)
predictions = predictor.predict_new_links(top_k_per_relation=200)

# Filter: diseases và drugs với TREATS relation
treatments = predictor.filter_predictions(
    semantic_types=['dsyn', 'phsu'],
    relations=['TREATS'],
    min_score=0.85
)

print(f"Found {len(treatments)} high-confidence treatment links")
treatments.to_csv('disease_drug_treatments.csv', index=False)
```

### Visualize Predicted Links

```python
from visualization.graph_visualizer import GraphVisualizer

# Load predictions
viz = GraphVisualizer()
viz.load_from_dataframe(treatments)

# Visualize
viz.create_pyvis_network(layout='hierarchical')
viz.show('predicted_treatments.html')
```

### Compare Existing vs Predicted

```python
# Existing links
viz_existing = GraphVisualizer()
viz_existing.load_from_json('suppkg/visualization_outputs/train_graph.json')
viz_existing.filter_by_relations(['TREATS'])
viz_existing.sample_nodes(300)
viz_existing.show('existing_treats.html')

# Predicted links
viz_predicted = GraphVisualizer()
viz_predicted.load_from_dataframe(treatments)
viz_predicted.sample_nodes(300)
viz_predicted.show('predicted_treats.html')

# Open both HTML files to compare!
```

---

## 📈 Use Cases

### Use Case 1: Drug Discovery

**Goal**: Tìm drug candidates cho một disease

```python
predictor = LinkPredictor()
predictor.load_model(...)
predictions = predictor.predict_new_links(top_k_per_relation=200)

# Filter treatments
treatments = predictor.filter_predictions(
    semantic_types=['dsyn', 'phsu'],
    relations=['TREATS'],
    min_score=0.9  # Very high confidence
)

# Find drugs for diabetes (C0011849)
diabetes_drugs = treatments[
    treatments['subject'].str.contains('C0011849')
].sort_values('score', ascending=False)

print("Top predicted drugs for diabetes:")
print(diabetes_drugs[['object', 'score']].head(10))
```

### Use Case 2: Protein Interaction Discovery

**Goal**: Discover new protein-protein interactions

```python
# Filter protein interactions
protein_links = predictor.filter_predictions(
    semantic_types=['aapp'],  # proteins
    relations=['INTERACTS_WITH'],
    min_score=0.85
)

# Analyze
print(f"Discovered {len(protein_links)} new protein interactions")

# Export for wet-lab validation
protein_links.to_csv('novel_protein_interactions.csv', index=False)
```

### Use Case 3: Knowledge Graph Completion

**Goal**: Complete missing links in KG

```python
# Get all high-confidence predictions
all_new_links = predictor.predict_new_links(
    top_k_per_relation=500,
    min_score=0.8
)

# Analyze by relation type
for relation in all_new_links['relation'].unique():
    rel_links = all_new_links[all_new_links['relation'] == relation]
    print(f"{relation}: {len(rel_links)} new links")

# Visualize completed graph
from visualization.graph_visualizer import GraphVisualizer

viz = GraphVisualizer()
viz.load_from_dataframe(all_new_links)
viz.sample_nodes(1000, strategy='pagerank')
viz.create_pyvis_network()
viz.show('completed_kg.html')
```

---

## 📁 File Organization

Sau khi hoàn thành workflow:

```
fuselinker/
├── suppkg_model_state.pth              # Trained model
│
├── suppkg/
│   ├── train.tsv, valid.tsv, test.tsv  # Original data
│   ├── *.pkl                            # Mappings
│   ├── *.npy                            # Embeddings
│   └── visualization_outputs/           # Auto-exported
│       ├── node_embeddings.npy
│       ├── relation_embeddings.npy
│       ├── train_graph.json
│       ├── test_graph.json
│       └── mappings.json
│
├── predicted_links.csv                  # ✨ NEW FUSED LINKS ✨
├── disease_drug_treatments.csv          # Filtered predictions
│
├── predicted_treatments.html            # Visualizations
├── existing_treats.html
└── completed_kg.html
```

---

## ⚡ Quick Reference Commands

### 1. Train Model
```bash
python main.py --data suppkg --w 0.75 --iterations 40000 \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --model_state_file suppkg_model_state.pth
```

### 2. Predict Fused Links
```bash
python predict_new_links.py \
    --model suppkg_model_state.pth \
    --data suppkg \
    --top_k 100 \
    --min_score 0.7
```

### 3. Visualize Dashboard
```bash
python -m visualization.app --data_dir suppkg/visualization_outputs
```

### 4. Quick Graph Viz
```python
from visualization.graph_visualizer import visualize_graph
visualize_graph('suppkg/visualization_outputs/train_graph.json', show=True)
```

### 5. Quick Embedding Viz
```python
from visualization.embedding_visualizer import visualize_embeddings
visualize_embeddings('suppkg/visualization_outputs/node_embeddings.npy', show=True)
```

---

## 🎯 Key Takeaways

### ✅ What You Get

1. **Trained Model** (`suppkg_model_state.pth`)
   - Learned embeddings fusing text + domain knowledge
   - Learned relation patterns

2. **Auto-Exported Data** (`suppkg/visualization_outputs/`)
   - Node embeddings (fused)
   - Relation embeddings
   - Graph structures (JSON)
   - Mappings

3. **NEW Fused Links** (`predicted_links.csv`)
   - Links NOT in training data
   - Predicted by model with confidence scores
   - **This is the answer to "liên kết nào đã được fuse?"**

4. **Interactive Visualizations**
   - Dashboard for exploration
   - Graph HTML files
   - Embedding plots
   - Comparison views

### 🔑 Understanding "Fused Links"

**Fused Links** có 2 meanings:

1. **During Training**: Model fuses embeddings từ 2 sources
   - Text embeddings (PubMedBERT)
   - Domain knowledge embeddings (Poincaré)
   - Creates combined representation

2. **After Training**: Model predicts NEW links
   - Links không có trong original data
   - Based on learned patterns từ fused embeddings
   - High score = high confidence

→ **`predicted_links.csv` chứa các NEW links được discover nhờ fusion process!**

---

## 🚀 Next Steps

After completing workflow:

1. **Validate Predictions**
   - Check top predictions với domain experts
   - Verify against literature
   - Wet-lab validation nếu applicable

2. **Iterate**
   - Adjust `min_score` threshold
   - Focus on specific relations
   - Retrain với different `w` (fusion weight)

3. **Export & Share**
   - Share CSV files với team
   - Create presentation với visualizations
   - Publish findings

4. **Production Use**
   - Integrate vào knowledge base
   - Build API for prediction service
   - Deploy dashboard for users

---

## 📚 Documentation Index

- **VISUALIZATION_PLAN.md** - Detailed architecture & plan
- **IMPLEMENTATION_SUMMARY.md** - What was built
- **HOW_TO_USE_PREDICTIONS.md** - Link prediction guide
- **visualization/README.md** - Visualization module docs
- **visualization/QUICKSTART.py** - Quick start guide

---

## 🎓 Additional Resources

### Research Papers Referenced
1. GNNExplainer - GNN explainability
2. Visual Analytics of Knowledge Graphs
3. UMAP dimensionality reduction
4. Interactive network visualization

### Code Examples
- `visualization/example_usage.py` - 8 examples
- `predict_new_links.py` - CLI tool
- `visualization/app.py` - Dashboard

---

## ❓ FAQ

**Q: Tại sao prediction script chậm?**
A: Predicting all possible pairs cho all relations. Giảm `top_k` hoặc specify specific relations.

**Q: Score bao nhiêu là tốt?**
A: >= 0.8 là high confidence, >= 0.9 là very high. Tùy use case.

**Q: Làm sao biết prediction đúng?**
A: Validate với domain knowledge, literature search, hoặc experiments.

**Q: Có thể retrain với predicted links không?**
A: Có! Add high-confidence predictions vào training data và retrain.

**Q: Visualization quá nhiều nodes?**
A: Dùng sampling, filtering, hoặc focus vào specific relations.

---

**🎉 Chúc bạn thành công với FuseLinker! 🎉**

For questions or issues, check the documentation or open a GitHub issue.

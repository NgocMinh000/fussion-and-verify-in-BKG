# Kế Hoạch Trực Quan Hóa Fused Links trong BKG

## 📋 Tổng Quan

Dự án này nhằm trực quan hóa các liên kết (links) được tạo ra sau khi FuseLinker thực hiện fusion embeddings và huấn luyện trên Background Knowledge Graph (BKG).

## 🎯 Mục Tiêu

1. **Hiển thị các liên kết mới được dự đoán** sau quá trình huấn luyện
2. **Trực quan hóa embedding space** để thấy được sự phân bố của entities
3. **Tạo interactive dashboard** để explore và phân tích các fused links
4. **Hỗ trợ explainability** - hiểu tại sao model dự đoán một liên kết cụ thể

## 📊 Phân Tích Output của FuseLinker

### Dữ Liệu Cần Visualize

1. **Node Embeddings** (sau fusion)
   - Combined embeddings từ PubMedBERT (text) và Poincaré (domain knowledge)
   - Dimension: `num_nodes × hidden_dim` (mặc định: 200)
   - Format: PyTorch tensor

2. **Relation Embeddings**
   - Learned relation weights từ DistMult
   - Dimension: `num_relations × hidden_dim`
   - 15 relation types trong suppkg dataset

3. **Predicted Links**
   - Triplets: (subject, relation, object)
   - Scores: Confidence score cho mỗi prediction (0-1)
   - Rankings: MR, MRR, Hits@1/3/10

4. **Graph Structure**
   - Original graph: ~9000 entities, 15 relations, ~305K links
   - Training/validation/test splits
   - Entity metadata: UMLS concept IDs, semantic types

## 🏗️ Kiến Trúc Hệ Thống Visualization

### Module 1: Data Export & Storage
```
fuselinker/
├── visualization/
│   ├── __init__.py
│   ├── export_utils.py          # Export embeddings, predictions, graphs
│   ├── data_processor.py        # Process data for visualization
│   └── config.py                # Configuration settings
```

**Chức năng:**
- Extend `main.py` để save embeddings và predictions
- Export data formats: JSON (graph), NPY (embeddings), CSV (predictions)
- Incremental saving during training để track evolution

### Module 2: Graph Visualization
```
fuselinker/
├── visualization/
│   ├── graph_visualizer.py      # Interactive graph visualization
│   └── static/
│       └── styles.css           # Styling cho graph display
```

**Công nghệ:**
- **PyVis**: Interactive network visualization
- **NetworkX**: Graph manipulation
- **Plotly**: Advanced interactive plots

**Features:**
- Node coloring theo semantic types (aapp, dsyn, etc.)
- Edge coloring theo relation types
- Filter by confidence score threshold
- Highlight new predicted links vs existing links
- Zoom, pan, search nodes

### Module 3: Embedding Space Visualization
```
fuselinker/
├── visualization/
│   ├── embedding_visualizer.py  # t-SNE/UMAP visualization
│   └── dimension_reduction.py   # Dimensionality reduction utils
```

**Công nghệ:**
- **UMAP**: Fast, high-quality dimensionality reduction
- **t-SNE**: Alternative visualization method
- **Plotly**: 2D/3D interactive scatter plots

**Features:**
- Reduce embeddings từ `hidden_dim` → 2D/3D
- Color points theo:
  - Semantic types
  - Cluster IDs (từ clustering algorithm)
  - Prediction confidence
- Hover to show entity info
- Link visualization trong embedding space

### Module 4: Interactive Dashboard
```
fuselinker/
├── visualization/
│   ├── dashboard.py             # Main Dash application
│   ├── components/
│   │   ├── graph_panel.py       # Graph visualization panel
│   │   ├── embedding_panel.py   # Embedding space panel
│   │   ├── stats_panel.py       # Statistics panel
│   │   └── filter_panel.py      # Filtering controls
│   └── app.py                   # Run dashboard server
```

**Công nghệ:**
- **Dash by Plotly**: Python framework cho web apps
- **Dash Bootstrap Components**: Professional UI
- **Plotly Dash**: Interactive components

**Features:**
- **Multi-panel layout:**
  - Panel 1: Graph structure view
  - Panel 2: Embedding space view
  - Panel 3: Statistics & metrics
  - Panel 4: Filters & controls

- **Interactive filters:**
  - Filter by relation type
  - Filter by confidence threshold
  - Filter by semantic type
  - Filter by time (training iteration)

- **Link analysis:**
  - Click on link to see details
  - Show scores, rankings
  - Compare with ground truth
  - Neighborhood exploration

- **Training evolution:**
  - Slider to view different training iterations
  - Animate embedding changes over time
  - Track metric improvements

### Module 5: Explainability Tools
```
fuselinker/
├── visualization/
│   ├── explainer.py             # GNN explainability tools
│   └── attention_viz.py         # Attention weights visualization
```

**Công nghệ:**
- **Custom implementation**: Analyze contribution of each embedding component
- **Gradient-based methods**: Feature importance

**Features:**
- Decompose prediction score: text vs domain contribution
- Show which neighbors influence a prediction
- Visualize attention/importance weights
- Compare fusion weights (w parameter effects)

## 🔧 Implementation Plan - Chi Tiết Từng Bước

### **Bước 1: Setup Environment & Dependencies** ✅
```bash
# Install required packages
pip install pyvis networkx plotly dash dash-bootstrap-components
pip install umap-learn scikit-learn pandas
pip install torch-geometric  # Optional: for advanced GNN viz
```

### **Bước 2: Modify main.py - Export Data**
**File:** `fuselinker/main.py`

**Thay đổi:**
1. Thêm callback để save embeddings mỗi `validate_every` iterations
2. Save final predictions với scores
3. Export graph structure và metadata

**Code additions:**
```python
# After line 145 (torch.save model state)
# Export embeddings
with torch.no_grad():
    final_embeddings = model(test_graph.to(device),
                            test_node_id.to(device),
                            test_rel.to(device),
                            test_norm.to(device))
    np.save(f'{args.data}/final_embeddings.npy',
            final_embeddings.cpu().numpy())
    np.save(f'{args.data}/final_relation_weights.npy',
            model.relation_weights.detach().cpu().numpy())

# Export predictions (modify evaluation code)
# Save predicted links with scores
```

### **Bước 3: Create Data Export Module**
**File:** `fuselinker/visualization/export_utils.py`

**Chức năng:**
- `export_embeddings()`: Save embeddings to NPY
- `export_graph_json()`: Export graph to JSON format cho PyVis
- `export_predictions()`: Save predictions với scores
- `export_metadata()`: Save entity/relation info

### **Bước 4: Implement Graph Visualizer**
**File:** `fuselinker/visualization/graph_visualizer.py`

**Class:** `GraphVisualizer`
- `load_graph()`: Load từ JSON/TSV
- `create_pyvis_network()`: Tạo PyVis network
- `add_nodes()`: Add nodes với properties
- `add_edges()`: Add edges với weights
- `apply_filters()`: Filter theo conditions
- `render()`: Generate HTML output
- `show()`: Display trong browser

### **Bước 5: Implement Embedding Visualizer**
**File:** `fuselinker/visualization/embedding_visualizer.py`

**Class:** `EmbeddingVisualizer`
- `load_embeddings()`: Load NPY files
- `reduce_dimensions()`: UMAP/t-SNE reduction
- `create_scatter_plot()`: Plotly scatter plot
- `add_metadata()`: Add hover info
- `highlight_clusters()`: Clustering và coloring
- `render()`: Generate interactive plot

### **Bước 6: Implement Dashboard**
**File:** `fuselinker/visualization/dashboard.py`

**Structure:**
```python
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout with 4 panels
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([html.H1("FuseLinker Visualization Dashboard")], width=12)
    ]),
    dbc.Row([
        dbc.Col([filter_panel], width=3),
        dbc.Col([graph_panel], width=9)
    ]),
    dbc.Row([
        dbc.Col([embedding_panel], width=6),
        dbc.Col([stats_panel], width=6)
    ])
])

# Callbacks for interactivity
@app.callback(...)
def update_graph(filters):
    ...
```

### **Bước 7: Create Individual Panels**
**Files:**
- `components/filter_panel.py`: Dropdowns, sliders, checkboxes
- `components/graph_panel.py`: PyVis iframe embedding
- `components/embedding_panel.py`: Plotly scatter plot
- `components/stats_panel.py`: Tables, metrics, charts

### **Bước 8: Integration & Testing**
1. Test với suppkg dataset
2. Verify tất cả filters hoạt động
3. Check performance với large graphs
4. Optimize rendering speed
5. Add loading indicators
6. Error handling

### **Bước 9: Documentation**
1. User guide: Cách sử dụng dashboard
2. Developer guide: Cách extend
3. README.md cho visualization module
4. Example notebooks

### **Bước 10: Advanced Features** (Optional)
1. **Comparison mode**: So sánh different training runs
2. **Export features**: Save filtered views
3. **Animation**: Training evolution over time
4. **Subgraph exploration**: Focus on specific regions
5. **Link prediction interface**: Interactive query

## 📚 Tham Khảo Các Bài Báo & Resources

### Papers về Knowledge Graph Visualization:
1. **"A Survey on the Visual Analytics of Knowledge Graph"** - ResearchGate 2024
   - Overview về các phương pháp visual analytics cho KG

2. **"GNNExplainer: Generating Explanations for Graph Neural Networks"** - NeurIPS 2019
   - Phương pháp explain GNN predictions

3. **"Interactive GNNExplainer"** - ArXiv 2024
   - Framework kết hợp visualization và explainability

### Tools & Libraries:
1. **PyVis** - [https://pyvis.readthedocs.io/](https://pyvis.readthedocs.io/)
2. **Plotly Dash** - [https://dash.plotly.com/](https://dash.plotly.com/)
3. **UMAP** - [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
4. **NetworkX** - [https://networkx.org/](https://networkx.org/)

### Tutorials:
1. **"Python Interactive Network Visualization Using NetworkX, Plotly, and Dash"** - Medium
2. **"Visualizing Neural Networks using t-SNE and UMAP"** - Kaggle
3. **"T-SNE and UMAP Projections"** - Plotly Documentation

## 🎨 Visualization Examples & Use Cases

### Use Case 1: Explore New Predicted Links
**Scenario:** User muốn xem top 100 predicted links với highest confidence

**Steps:**
1. Set confidence threshold slider → 0.9
2. Toggle "Show only new predictions" checkbox
3. Graph hiển thị chỉ predicted links (màu khác vs ground truth)
4. Click vào link để xem details: score, ranking, neighboring entities

### Use Case 2: Analyze Embedding Space
**Scenario:** User muốn hiểu entities cluster như thế nào

**Steps:**
1. Switch to "Embedding View" tab
2. Select coloring: "By Semantic Type"
3. UMAP plot hiển thị entities trong 2D
4. Observe clusters: diseases, drugs, proteins, etc.
5. Hover để xem entity info

### Use Case 3: Compare Fusion Strategies
**Scenario:** So sánh w=0.5 vs w=0.75 (text vs domain weight)

**Steps:**
1. Load two trained models
2. Side-by-side embedding visualization
3. Compare prediction differences
4. Analyze which fusion works better cho specific relation types

### Use Case 4: Training Evolution
**Scenario:** Xem embeddings thay đổi như thế nào qua training

**Steps:**
1. Load saved embeddings from different iterations
2. Use slider to animate từ iteration 0 → 40000
3. Observe embedding space stabilization
4. Track metrics improvement timeline

## 📈 Expected Outputs

### 1. Static Visualizations
- **Graph PNG/SVG**: High-quality network diagrams
- **Embedding plots**: 2D/3D scatter plots
- **Heatmaps**: Relation type distributions, confusion matrices

### 2. Interactive HTML
- **Standalone graph HTML**: Shareable PyVis networks
- **Embedding explorer HTML**: Interactive Plotly plots
- **Full dashboard**: Multi-panel Dash application

### 3. Data Exports
- **Filtered predictions CSV**: Predicted links với scores
- **Cluster assignments**: Entities grouped by clusters
- **Statistics JSON**: Metrics, counts, distributions

## 🚀 Deployment Options

### Local Development
```bash
cd fuselinker/visualization
python app.py
# Access at http://localhost:8050
```

### Production Deployment
- **Heroku**: Free tier cho demo apps
- **AWS EC2**: Full control
- **Google Colab**: Notebooks với ngrok tunneling
- **Streamlit Cloud**: Alternative to Dash

## ⏱️ Estimated Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Setup & Dependencies | 0.5h |
| 2 | Modify main.py for export | 1h |
| 3 | Create export_utils.py | 1h |
| 4 | Implement graph_visualizer.py | 2h |
| 5 | Implement embedding_visualizer.py | 2h |
| 6 | Create dashboard structure | 2h |
| 7 | Implement panels & components | 3h |
| 8 | Integration & Testing | 2h |
| 9 | Documentation | 1h |
| **Total** | | **~14-15 hours** |

## 🎯 Success Criteria

✅ **Criteria 1:** Có thể visualize toàn bộ graph với 9000 entities
✅ **Criteria 2:** Interactive filters hoạt động mượt mà
✅ **Criteria 3:** Embedding visualization rõ ràng, có clusters
✅ **Criteria 4:** Dashboard loading time < 5 giây
✅ **Criteria 5:** Có thể export visualizations ra file
✅ **Criteria 6:** Documentation đầy đủ, dễ hiểu

## 🔄 Future Enhancements

1. **Real-time training monitoring**: Stream metrics during training
2. **Comparison tools**: Multi-model comparison side-by-side
3. **Advanced analytics**: Path finding, centrality measures
4. **Mobile-responsive**: Optimize cho mobile browsers
5. **API integration**: REST API cho programmatic access
6. **Cloud storage**: Save/load configurations from cloud

---

## 📝 Notes

- Optimization quan trọng cho large graphs (>10K nodes)
- Consider sampling strategies cho visualization
- Cache computed layouts để tăng tốc độ
- Progressive loading cho large datasets
- WebGL rendering cho smooth interactions

**Last Updated:** 2026-01-03
**Author:** Claude Code Assistant

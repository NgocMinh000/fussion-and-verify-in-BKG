# FuseLinker Visualization Module

Công cụ trực quan hóa interactive cho FuseLinker - giúp khám phá và phân tích các liên kết được fusion trong Background Knowledge Graph (BKG).

## 🎯 Tính Năng

### 1. Graph Structure Visualization
- **Interactive network graph** với PyVis
- Filter theo semantic types và relation types
- Node coloring theo loại thực thể
- Edge coloring theo loại quan hệ
- Zoom, pan, search nodes
- Sample strategy cho large graphs

### 2. Embedding Space Visualization
- **Dimensionality reduction**: UMAP, t-SNE, PCA
- **2D/3D scatter plots** với Plotly
- Color by semantic type hoặc clusters
- Interactive hover với entity details
- Clustering analysis (K-Means, DBSCAN)

### 3. Interactive Dashboard
- **Multi-panel layout** với Dash
- Real-time filtering
- Statistics và analytics
- Export visualizations
- Responsive UI

## 📦 Installation

### Bước 1: Install Dependencies

```bash
cd fuselinker
pip install -r visualization/requirements.txt
```

Hoặc install từng package:

```bash
pip install pyvis networkx plotly dash dash-bootstrap-components
pip install umap-learn scikit-learn pandas numpy matplotlib
```

### Bước 2: Verify Installation

```python
python -c "import pyvis, plotly, dash, umap; print('✓ All packages installed')"
```

## 🚀 Quick Start

### Option 1: Automatic Export (Recommended)

Khi train model, data sẽ tự động được export:

```bash
cd fuselinker
python main.py --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 40000 \
    --w 0.75 \
    --model_state_file suppkg_model_state.pth
```

Sau khi training xong, visualization data sẽ được save vào `suppkg/visualization_outputs/`

### Option 2: Manual Export

Nếu đã có trained model:

```python
import torch
from visualization.export_utils import export_full_visualization_data
from data_loader import Data
import pandas as pd

# Load data
train = pd.read_csv('suppkg/train.tsv', sep='\t', header=None)
valid = pd.read_csv('suppkg/valid.tsv', sep='\t', header=None)
test = pd.read_csv('suppkg/test.tsv', sep='\t', header=None)
graph = pd.concat([train, valid, test])

knowledge_graph = Data(graph, train, valid, test)

# Load model
checkpoint = torch.load('suppkg_model_state.pth')
model.load_state_dict(checkpoint['state_dict'])

# Export
export_full_visualization_data(
    model=model,
    graph=test_graph,
    node_ids=test_node_id,
    rel_ids=test_rel,
    norm=test_norm,
    train_data=train_data_np,
    test_data=test_data_np,
    entity2index=knowledge_graph.entity2index,
    index2entity=knowledge_graph.index2entity,
    relation2index=knowledge_graph.relation2index,
    index2relation=knowledge_graph.index2relation,
    output_dir='suppkg/visualization_outputs',
    device=device
)
```

## 📊 Usage Examples

### Example 1: Visualize Graph Structure

```python
from visualization.graph_visualizer import visualize_graph

# Quick visualization
viz = visualize_graph(
    json_path='suppkg/visualization_outputs/train_graph.json',
    output_path='graph_viz.html',
    max_nodes=1000,
    semantic_types=['dsyn', 'phsu'],  # Diseases and drugs
    relations=['TREATS', 'CAUSES'],
    show=True
)
```

### Example 2: Visualize Embedding Space

```python
from visualization.embedding_visualizer import visualize_embeddings

# Quick visualization
viz = visualize_embeddings(
    embedding_path='suppkg/visualization_outputs/node_embeddings.npy',
    output_path='embedding_viz.html',
    method='umap',
    n_components=2,
    color_by='semantic_type',
    n_clusters=8,  # Enable clustering
    show=True
)
```

### Example 3: Advanced Graph Customization

```python
from visualization.graph_visualizer import GraphVisualizer

# Load and customize
viz = GraphVisualizer(width="1200px", height="900px")
viz.load_from_json('suppkg/visualization_outputs/train_graph.json')

# Apply multiple filters
viz.filter_by_semantic_types(['dsyn', 'phsu', 'orch'])
viz.filter_by_relations(['TREATS', 'INHIBITS'])
viz.sample_nodes(500, strategy='pagerank')

# Create and customize network
viz.create_pyvis_network(physics_enabled=True, layout='hierarchical')

# Get statistics
viz.print_statistics()

# Render
viz.show('custom_graph.html')
```

### Example 4: Advanced Embedding Analysis

```python
from visualization.embedding_visualizer import EmbeddingVisualizer

# Load embeddings
viz = EmbeddingVisualizer()
viz.load_embeddings('suppkg/visualization_outputs/node_embeddings.npy')

# Reduce dimensions with custom parameters
viz.reduce_dimensions(
    method='umap',
    n_components=3,  # 3D
    n_neighbors=30,
    min_dist=0.05
)

# Cluster
viz.cluster_embeddings(method='dbscan', eps=0.3, min_samples=10)

# Create multiple plots
scatter_3d = viz.create_scatter_plot(color_by='cluster', title="3D Embedding Space")
density = viz.create_density_plot()
centroids = viz.plot_cluster_centroids()

# Save all
viz.save_plot(scatter_3d, 'embedding_3d.html')
viz.save_plot(density, 'embedding_density.html')
viz.save_plot(centroids, 'cluster_centroids.html')

# Export reduced embeddings
viz.export_reduced_embeddings('reduced_embeddings.csv', format='csv')
```

## 🎨 Interactive Dashboard

### Launch Dashboard

```bash
cd fuselinker
python -m visualization.app --data_dir suppkg/visualization_outputs
```

Hoặc:

```bash
python visualization/app.py --data_dir suppkg/visualization_outputs --port 8050
```

Truy cập: http://localhost:8050

### Dashboard Features

**Left Sidebar - Filters:**
- Semantic type filter (multi-select)
- Relation type filter (multi-select)
- Max nodes slider
- Apply/Reset buttons

**Tab 1 - Graph Structure:**
- Interactive PyVis network
- Physics simulation
- Node/edge details on hover
- Zoom, pan, search

**Tab 2 - Embedding Space:**
- Choose reduction method (UMAP/t-SNE/PCA)
- 2D or 3D visualization
- Color by semantic type or cluster
- Interactive Plotly scatter plot

**Tab 3 - About:**
- Dashboard documentation
- Usage instructions

## 📁 Output Structure

Sau khi export, data structure sẽ như sau:

```
suppkg/visualization_outputs/
├── node_embeddings.npy              # Node embeddings (num_nodes x hidden_dim)
├── node_embeddings.meta.json        # Metadata và entity mapping
├── relation_embeddings.npy          # Relation embeddings (num_rels x hidden_dim)
├── relation_embeddings.mapping.json # Relation mapping
├── train_graph.json                 # Training graph structure
├── test_graph.json                  # Test graph structure
└── mappings.json                    # All mappings (entities, relations)
```

## 🎨 Customization

### Modify Colors

Edit `visualization/config.py`:

```python
VIZ_CONFIG = {
    'semantic_type_colors': {
        'dsyn': '#YOUR_COLOR',  # Disease
        'phsu': '#YOUR_COLOR',  # Drug
        # ...
    },
    'relation_colors': {
        'TREATS': '#YOUR_COLOR',
        # ...
    }
}
```

### Modify Graph Layout

```python
viz.create_pyvis_network(
    physics_enabled=True,
    layout='hierarchical'  # Options: force_directed, hierarchical, circular
)
```

### Modify UMAP Parameters

```python
viz.reduce_dimensions(
    method='umap',
    n_neighbors=50,      # Larger = more global structure
    min_dist=0.01,       # Smaller = tighter clusters
    metric='euclidean'   # Options: cosine, euclidean, manhattan
)
```

## 🔧 Troubleshooting

### Issue: "No module named 'visualization'"

**Solution:** Chạy từ thư mục `fuselinker/`:

```bash
cd fuselinker
python -m visualization.app --data_dir suppkg/visualization_outputs
```

### Issue: Dashboard không load graph

**Solution:** Kiểm tra data path:

```python
import os
print(os.path.exists('suppkg/visualization_outputs/train_graph.json'))
```

### Issue: UMAP chậm với large embeddings

**Solution:** Sample trước khi visualize:

```python
# Sample 2000 nodes
indices = np.random.choice(len(embeddings), 2000, replace=False)
sampled_embeddings = embeddings[indices]
```

### Issue: PyVis graph quá chậm

**Solution:** Giảm số nodes:

```python
viz.sample_nodes(500, strategy='degree')
viz.create_pyvis_network(physics_enabled=False)  # Disable physics
```

## 📚 API Reference

### GraphVisualizer

```python
class GraphVisualizer:
    def load_from_json(json_path)
    def load_from_dataframe(df, entity2index, index2entity)
    def filter_by_semantic_types(semantic_types)
    def filter_by_relations(relations)
    def sample_nodes(max_nodes, strategy='degree')
    def create_pyvis_network(physics_enabled=True, layout='force_directed')
    def render(output_path)
    def show(output_path)
    def get_statistics()
    def print_statistics()
```

### EmbeddingVisualizer

```python
class EmbeddingVisualizer:
    def load_embeddings(embedding_path, entity_mapping)
    def reduce_dimensions(method='umap', n_components=2, **kwargs)
    def cluster_embeddings(method='kmeans', n_clusters=8, **kwargs)
    def create_scatter_plot(color_by='semantic_type', ...)
    def create_density_plot()
    def plot_cluster_centroids()
    def save_plot(fig, output_path)
    def export_reduced_embeddings(output_path, format='csv')
```

## 🎓 Examples & Tutorials

Xem file `VISUALIZATION_PLAN.md` để biết:
- Detailed architecture
- Use cases
- Best practices
- Future enhancements

## 🤝 Contributing

Contributions welcome! Areas to improve:
- [ ] Add animation for training evolution
- [ ] Implement subgraph exploration
- [ ] Add export to various formats (PDF, SVG)
- [ ] Improve performance for very large graphs
- [ ] Add more clustering algorithms
- [ ] Implement link prediction interface

## 📄 License

Same as parent project.

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review `VISUALIZATION_PLAN.md`
3. Open an issue on GitHub

---

**Last Updated:** 2026-01-03
**Version:** 0.1.0
**Author:** Claude Code Assistant

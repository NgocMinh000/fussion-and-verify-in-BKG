"""
QUICK START GUIDE
FuseLinker Visualization - Hướng Dẫn Nhanh

Hướng dẫn nhanh để bắt đầu visualize các fused links trong BKG
"""

# ============================================================================
# BƯỚC 1: INSTALL DEPENDENCIES
# ============================================================================

print("""
BƯỚC 1: Cài Đặt Dependencies
────────────────────────────────────────────────────────────────────────

Chạy lệnh sau trong terminal:
""")

print("""
cd fuselinker
pip install -r visualization/requirements.txt
""")

print("""
Hoặc cài từng package:
pip install pyvis networkx plotly dash dash-bootstrap-components
pip install umap-learn scikit-learn pandas numpy

Verify installation:
python -c "import pyvis, plotly, dash, umap; print('✓ All installed!')"
""")

# ============================================================================
# BƯỚC 2: TRAIN MODEL (Nếu chưa có)
# ============================================================================

print("""
BƯỚC 2: Train FuseLinker Model
────────────────────────────────────────────────────────────────────────

Nếu chưa train model, chạy:
""")

print("""
python main.py --data suppkg \\
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \\
    --knowledge_embedding_file poincare_embeddings.npy \\
    --num_hidden_layers 2 \\
    --iterations 40000 \\
    --evaluate_every 1000 \\
    --w 0.75 \\
    --model_state_file suppkg_model_state.pth

Sau khi training xong, visualization data sẽ tự động được export vào:
suppkg/visualization_outputs/
""")

# ============================================================================
# BƯỚC 3: VISUALIZE GRAPH
# ============================================================================

print("""
BƯỚC 3: Visualize Graph Structure
────────────────────────────────────────────────────────────────────────

Option A - Quick visualization:
""")

print("""python
from visualization.graph_visualizer import visualize_graph

# Visualize toàn bộ graph (sample 1000 nodes)
viz = visualize_graph(
    json_path='suppkg/visualization_outputs/train_graph.json',
    output_path='graph.html',
    max_nodes=1000,
    show=True
)
""")

print("""
Option B - Filtered visualization (chỉ diseases và drugs):
""")

print("""python
from visualization.graph_visualizer import GraphVisualizer

viz = GraphVisualizer()
viz.load_from_json('suppkg/visualization_outputs/train_graph.json')

# Filter theo semantic types
viz.filter_by_semantic_types(['dsyn', 'phsu'])  # diseases, drugs

# Filter theo relations
viz.filter_by_relations(['TREATS', 'PREVENTS'])

# Sample top nodes
viz.sample_nodes(500, strategy='degree')

# Tạo và hiển thị
viz.create_pyvis_network(layout='hierarchical')
viz.show('disease_drug_network.html')
""")

# ============================================================================
# BƯỚC 4: VISUALIZE EMBEDDINGS
# ============================================================================

print("""
BƯỚC 4: Visualize Embedding Space
────────────────────────────────────────────────────────────────────────

Option A - Quick 2D UMAP:
""")

print("""python
from visualization.embedding_visualizer import visualize_embeddings

viz = visualize_embeddings(
    embedding_path='suppkg/visualization_outputs/node_embeddings.npy',
    output_path='embeddings_2d.html',
    method='umap',
    n_components=2,
    color_by='semantic_type',
    show=True
)
""")

print("""
Option B - 3D với clustering:
""")

print("""python
from visualization.embedding_visualizer import EmbeddingVisualizer

viz = EmbeddingVisualizer()
viz.load_embeddings('suppkg/visualization_outputs/node_embeddings.npy')

# Reduce to 3D
viz.reduce_dimensions(method='umap', n_components=3)

# Cluster
viz.cluster_embeddings(method='kmeans', n_clusters=10)

# Create plot
fig = viz.create_scatter_plot(color_by='cluster', title="3D Clusters")
viz.show_plot(fig)
""")

# ============================================================================
# BƯỚC 5: LAUNCH DASHBOARD
# ============================================================================

print("""
BƯỚC 5: Launch Interactive Dashboard
────────────────────────────────────────────────────────────────────────

Chạy dashboard application:
""")

print("""
python -m visualization.app --data_dir suppkg/visualization_outputs

Hoặc:
cd fuselinker
python visualization/app.py --data_dir suppkg/visualization_outputs --port 8050

Sau đó mở browser: http://localhost:8050

Dashboard có 3 tabs:
  📊 Graph Structure - Interactive network graph
  🎯 Embedding Space - 2D/3D embeddings với UMAP/t-SNE
  ℹ️ About - Documentation
""")

# ============================================================================
# BƯỚC 6: RUN EXAMPLES
# ============================================================================

print("""
BƯỚC 6: Run Example Scripts
────────────────────────────────────────────────────────────────────────

Chạy tất cả 8 examples:
""")

print("""
cd fuselinker
python visualization/example_usage.py --all

Hoặc chạy example cụ thể:
python visualization/example_usage.py --example 1

Available examples:
  1. Basic graph visualization
  2. Filtered graph (diseases and drugs)
  3. 2D embedding visualization (UMAP)
  4. 3D embedding visualization (t-SNE)
  5. Clustering analysis
  6. Compare semantic type networks
  7. Analyze specific relation
  8. Embedding density plot

Results được save vào outputs/ directory.
""")

# ============================================================================
# COMMON USE CASES
# ============================================================================

print("""
────────────────────────────────────────────────────────────────────────
COMMON USE CASES
────────────────────────────────────────────────────────────────────────

Use Case 1: Tìm các drug treat một disease cụ thể
""")

print("""python
viz = GraphVisualizer()
viz.load_from_json('suppkg/visualization_outputs/train_graph.json')

# Filter diseases và drugs, chỉ TREATS relation
viz.filter_by_semantic_types(['dsyn', 'phsu'])
viz.filter_by_relations(['TREATS'])

# Visualize
viz.create_pyvis_network()
viz.show('drug_disease_treatment.html')
""")

print("""
Use Case 2: Xem entities cluster như thế nào
""")

print("""python
viz = EmbeddingVisualizer()
viz.load_embeddings('suppkg/visualization_outputs/node_embeddings.npy')

# Reduce và cluster
viz.reduce_dimensions(method='umap', n_components=2)
viz.cluster_embeddings(n_clusters=8)

# Visualize clusters
fig = viz.create_scatter_plot(color_by='cluster')
viz.show_plot(fig)

# Export cluster assignments
viz.export_reduced_embeddings('clusters.csv', format='csv')
""")

print("""
Use Case 3: So sánh different semantic types
""")

print("""python
semantic_groups = {
    'diseases': ['dsyn', 'sosy'],
    'chemicals': ['orch', 'phsu'],
    'biological': ['aapp', 'gngm']
}

for name, types in semantic_groups.items():
    viz = GraphVisualizer()
    viz.load_from_json('suppkg/visualization_outputs/train_graph.json')
    viz.filter_by_semantic_types(types)
    viz.sample_nodes(400)
    viz.create_pyvis_network()
    viz.show(f'{name}_network.html')
""")

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

print("""
────────────────────────────────────────────────────────────────────────
TROUBLESHOOTING
────────────────────────────────────────────────────────────────────────

Problem: "No module named 'visualization'"
Solution: Chạy từ fuselinker/ directory:
  cd fuselinker
  python -m visualization.app ...

Problem: Dashboard không load
Solution: Check data path tồn tại:
  ls suppkg/visualization_outputs/train_graph.json

Problem: UMAP quá chậm
Solution: Sample embeddings trước:
  indices = np.random.choice(len(emb), 2000, replace=False)
  sampled = emb[indices]

Problem: PyVis graph lag
Solution: Giảm nodes và disable physics:
  viz.sample_nodes(500)
  viz.create_pyvis_network(physics_enabled=False)

Problem: ImportError for packages
Solution: Reinstall dependencies:
  pip install -r visualization/requirements.txt --force-reinstall
""")

# ============================================================================
# NEXT STEPS
# ============================================================================

print("""
────────────────────────────────────────────────────────────────────────
NEXT STEPS
────────────────────────────────────────────────────────────────────────

1. Explore interactive dashboard
   → Thử các filters, zoom, pan
   → Export visualizations

2. Analyze clusters
   → Xem entities nào cluster together
   → So sánh với domain knowledge

3. Identify new predictions
   → Compare train vs test graphs
   → Analyze high-confidence predictions

4. Customize visualizations
   → Edit colors in config.py
   → Modify layouts, parameters
   → Create custom views

5. Share results
   → Export HTML files
   → Create presentations
   → Generate reports

────────────────────────────────────────────────────────────────────────

For detailed documentation, see:
  - visualization/README.md
  - VISUALIZATION_PLAN.md

For API reference:
  - Check docstrings in each module
  - Run help(GraphVisualizer) or help(EmbeddingVisualizer)

Happy visualizing! 🎨📊
""")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FUSELINKER VISUALIZATION - QUICK START GUIDE")
    print("="*70 + "\n")

"""
Interactive dashboard for FuseLinker visualizations.
Displays embeddings, graph structure, and model statistics.

Requirements: streamlit, plotly
Install: pip install streamlit plotly

Run: streamlit run visualization/app.py
"""

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: streamlit and plotly required for dashboard")
    print("Install: pip install streamlit plotly")
    exit(1)

import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@st.cache_data
def load_data(viz_dir):
    """Load all visualization data."""
    data = {}

    # Load embeddings
    data['entity_embeddings'] = np.load(f'{viz_dir}/entity_embeddings.npy')
    data['relation_embeddings'] = np.load(f'{viz_dir}/relation_embeddings.npy')

    # Load optional components
    for name in ['learned_embeddings', 'text_embeddings', 'domain_embeddings']:
        path = Path(f'{viz_dir}/{name}.npy')
        if path.exists():
            data[name] = np.load(path)

    # Load mappings
    with open(f'{viz_dir}/index2entity.pkl', 'rb') as f:
        data['index2entity'] = pickle.load(f)
    with open(f'{viz_dir}/index2relation.pkl', 'rb') as f:
        data['index2relation'] = pickle.load(f)

    # Load statistics
    with open(f'{viz_dir}/statistics.json', 'r') as f:
        data['statistics'] = json.load(f)

    # Load graph structure
    with open(f'{viz_dir}/graph_structure.json', 'r') as f:
        data['graph'] = json.load(f)

    # Load model config
    with open(f'{viz_dir}/model_config.json', 'r') as f:
        data['config'] = json.load(f)

    return data


def plot_embeddings_interactive(embeddings, labels, title, method='PCA', sample_size=1000):
    """Create interactive embedding plot."""
    # Sample if needed
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]

    # Dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        var_explained = reducer.explained_variance_ratio_
        axis_labels = [f'PC1 ({var_explained[0]:.1%})', f'PC2 ({var_explained[1]:.1%})']
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = reducer.fit_transform(embeddings)
        axis_labels = ['t-SNE 1', 't-SNE 2']

    # Create interactive plot
    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hover_name=labels,
        title=f'{title} ({method})',
        labels={'x': axis_labels[0], 'y': axis_labels[1]}
    )

    fig.update_traces(marker=dict(size=8, opacity=0.6))
    fig.update_layout(height=600, hovermode='closest')

    return fig


def show_statistics(data):
    """Display model and graph statistics."""
    st.header("📊 Model Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of Entities", f"{data['config']['num_entities']:,}")
        st.metric("Number of Relations", f"{data['config']['num_relations']:,}")

    with col2:
        st.metric("Embedding Dimension", data['config']['embedding_dim'])
        st.metric("Model Type", data['config']['model_type'])

    with col3:
        st.metric("Train Triples", f"{len(data['graph']['train_triples']):,}")
        st.metric("Test Triples", f"{len(data['graph']['test_triples']):,}")

    # Embedding statistics
    st.subheader("Embedding Statistics")
    emb_stats = data['statistics']['embedding_stats']

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Entity Embeddings**")
        st.write(f"- Mean: {emb_stats['entity_emb_mean']:.4f}")
        st.write(f"- Std: {emb_stats['entity_emb_std']:.4f}")

    with col2:
        st.write("**Relation Embeddings**")
        st.write(f"- Mean: {emb_stats['relation_emb_mean']:.4f}")
        st.write(f"- Std: {emb_stats['relation_emb_std']:.4f}")

    # Entity degree statistics
    st.subheader("Entity Degree Statistics")
    degree_stats = data['statistics']['entity_degree_stats']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Degree", f"{degree_stats['mean']:.1f}")
    with col2:
        st.metric("Median Degree", f"{degree_stats['median']:.0f}")
    with col3:
        st.metric("Max Degree", degree_stats['max'])


def show_relation_distribution(data):
    """Display relation distribution."""
    st.header("📈 Relation Distribution")

    relation_counts = data['statistics']['relation_distribution']

    # Sort by count
    relations = list(relation_counts.keys())
    counts = list(relation_counts.values())
    sorted_data = sorted(zip(relations, counts), key=lambda x: x[1], reverse=True)
    relations_sorted, counts_sorted = zip(*sorted_data)

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(x=relations_sorted, y=counts_sorted, marker_color='steelblue')
    ])

    fig.update_layout(
        title='Number of Triples per Relation',
        xaxis_title='Relation',
        yaxis_title='Count',
        height=500,
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig, use_container_width=True)


def show_embeddings(data):
    """Display embedding visualizations."""
    st.header("🎨 Embedding Visualizations")

    # Sidebar options
    with st.sidebar:
        st.subheader("Visualization Options")

        embedding_type = st.selectbox(
            "Embedding Type",
            ["Entity Embeddings", "Relation Embeddings",
             "Text Embeddings", "Domain Embeddings", "Learned Embeddings"]
        )

        method = st.radio("Method", ["PCA", "t-SNE"])
        sample_size = st.slider("Sample Size", 100, 2000, 1000, 100)

    # Get selected embeddings
    emb_map = {
        "Entity Embeddings": ("entity_embeddings", "index2entity"),
        "Relation Embeddings": ("relation_embeddings", "index2relation"),
        "Text Embeddings": ("text_embeddings", "index2entity"),
        "Domain Embeddings": ("domain_embeddings", "index2entity"),
        "Learned Embeddings": ("learned_embeddings", "index2entity")
    }

    emb_key, label_key = emb_map[embedding_type]

    if emb_key not in data:
        st.warning(f"{embedding_type} not available in the data")
        return

    embeddings = data[emb_key]
    labels = [str(data[label_key].get(i, f'idx_{i}'))[:30] for i in range(len(embeddings))]

    # Generate plot
    with st.spinner(f'Generating {method} visualization...'):
        fig = plot_embeddings_interactive(embeddings, labels, embedding_type, method, sample_size)
        st.plotly_chart(fig, use_container_width=True)


def show_fusion_comparison(data):
    """Compare fused embedding components."""
    st.header("🔀 Fusion Component Comparison")

    if 'text_embeddings' not in data or 'domain_embeddings' not in data:
        st.warning("Text or domain embeddings not available")
        return

    st.write("""
    This section compares the different embedding components:
    - **Text Embeddings**: From PubMedBERT (contextual language understanding)
    - **Domain Embeddings**: From Poincaré (hierarchical domain knowledge)
    - **Learned Embeddings**: Task-specific learned during training
    - **Fused Embeddings**: Final combined embeddings used for prediction
    """)

    # Compute similarities or other comparisons
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Text Embedding Dim", data['text_embeddings'].shape[1])
        st.metric("Domain Embedding Dim", data['domain_embeddings'].shape[1])

    with col2:
        if 'learned_embeddings' in data:
            st.metric("Learned Embedding Dim", data['learned_embeddings'].shape[1])
        st.metric("Fused Embedding Dim", data['entity_embeddings'].shape[1])

    # Show norms
    st.subheader("Embedding Norms (Average)")

    text_norm = np.mean(np.linalg.norm(data['text_embeddings'], axis=1))
    domain_norm = np.mean(np.linalg.norm(data['domain_embeddings'], axis=1))
    fused_norm = np.mean(np.linalg.norm(data['entity_embeddings'], axis=1))

    norms = {
        'Text': text_norm,
        'Domain': domain_norm,
        'Fused': fused_norm
    }

    if 'learned_embeddings' in data:
        learned_norm = np.mean(np.linalg.norm(data['learned_embeddings'], axis=1))
        norms['Learned'] = learned_norm

    fig = go.Figure(data=[
        go.Bar(x=list(norms.keys()), y=list(norms.values()), marker_color='darkorange')
    ])

    fig.update_layout(
        title='Average Embedding Norms',
        xaxis_title='Embedding Type',
        yaxis_title='L2 Norm',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="FuseLinker Visualization Dashboard",
        page_icon="🔗",
        layout="wide"
    )

    st.title("🔗 FuseLinker Visualization Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        viz_dir = st.text_input("Visualization Data Directory", "suppkg/visualization_outputs")

        if not Path(viz_dir).exists():
            st.error(f"Directory not found: {viz_dir}")
            st.stop()

        if st.button("🔄 Reload Data"):
            st.cache_data.clear()

    # Load data
    try:
        with st.spinner("Loading data..."):
            data = load_data(viz_dir)
        st.sidebar.success(f"✓ Loaded {len(data['entity_embeddings'])} entities")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Navigation
    page = st.sidebar.radio(
        "📖 Navigation",
        ["Statistics", "Embeddings", "Relation Distribution", "Fusion Comparison"]
    )

    # Display selected page
    if page == "Statistics":
        show_statistics(data)
    elif page == "Embeddings":
        show_embeddings(data)
    elif page == "Relation Distribution":
        show_relation_distribution(data)
    elif page == "Fusion Comparison":
        show_fusion_comparison(data)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <small>FuseLinker Knowledge Graph Embedding Visualization |
        Model: {model_type} |
        {num_entities:,} entities, {num_relations} relations</small>
        </div>
        """.format(
            model_type=data['config']['model_type'],
            num_entities=data['config']['num_entities'],
            num_relations=data['config']['num_relations']
        ),
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()

"""
Embedding space visualization using UMAP/t-SNE and Plotly

This module provides the EmbeddingVisualizer class for creating interactive
2D/3D visualizations of the embedding space.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

# Dimensionality reduction
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

from .config import (
    VIZ_CONFIG,
    get_semantic_type,
    get_color_for_semantic_type
)


class EmbeddingVisualizer:
    """
    Interactive embedding space visualization using UMAP/t-SNE and Plotly

    Features:
    - Dimensionality reduction (UMAP, t-SNE, PCA)
    - 2D/3D scatter plots
    - Color by semantic type or cluster
    - Interactive hover with entity info
    - Cluster analysis
    """

    def __init__(self):
        """Initialize EmbeddingVisualizer"""
        self.embeddings = None
        self.reduced_embeddings = None
        self.entity_ids = []
        self.metadata = {}
        self.clusters = None

    def load_embeddings(
        self,
        embedding_path: Union[str, Path],
        entity_mapping: Optional[Dict] = None
    ):
        """
        Load embeddings from NPY file

        Args:
            embedding_path: Path to NPY file
            entity_mapping: Optional dict mapping indices to entity IDs
        """
        embedding_path = Path(embedding_path)

        # Load embeddings
        self.embeddings = np.load(embedding_path)

        # Load metadata if exists
        meta_path = embedding_path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
                if 'entity_mapping' in self.metadata and entity_mapping is None:
                    entity_mapping = {int(k): v for k, v in self.metadata['entity_mapping'].items()}

        # Set entity IDs
        if entity_mapping:
            self.entity_ids = [entity_mapping.get(i, f"Entity_{i}")
                              for i in range(len(self.embeddings))]
        else:
            self.entity_ids = [f"Entity_{i}" for i in range(len(self.embeddings))]

        print(f"✓ Loaded embeddings from {embedding_path}")
        print(f"  - Shape: {self.embeddings.shape}")
        print(f"  - Entities: {len(self.entity_ids)}")

        return self

    def reduce_dimensions(
        self,
        method: str = 'umap',
        n_components: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        """
        Reduce embedding dimensionality for visualization

        Args:
            method: Reduction method ('umap', 'tsne', 'pca')
            n_components: Number of dimensions (2 or 3)
            random_state: Random seed
            **kwargs: Additional arguments for the reduction method
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings() first.")

        print(f"Reducing dimensions using {method.upper()}...")

        if method == 'umap':
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                metric=kwargs.get('metric', 'cosine'),
                random_state=random_state
            )
        elif method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                perplexity=kwargs.get('perplexity', 30),
                n_iter=kwargs.get('n_iter', 1000),
                random_state=random_state
            )
        elif method == 'pca':
            reducer = PCA(
                n_components=n_components,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.reduced_embeddings = reducer.fit_transform(self.embeddings)

        print(f"✓ Reduced to {n_components}D")
        print(f"  - Output shape: {self.reduced_embeddings.shape}")

        return self

    def cluster_embeddings(
        self,
        method: str = 'kmeans',
        n_clusters: int = 8,
        **kwargs
    ):
        """
        Cluster embeddings using K-Means or DBSCAN

        Args:
            method: Clustering method ('kmeans', 'dbscan')
            n_clusters: Number of clusters (for K-Means)
            **kwargs: Additional arguments for clustering
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings() first.")

        print(f"Clustering embeddings using {method.upper()}...")

        if method == 'kmeans':
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=kwargs.get('random_state', 42),
                n_init=kwargs.get('n_init', 10)
            )
        elif method == 'dbscan':
            clusterer = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5),
                metric=kwargs.get('metric', 'cosine')
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        self.clusters = clusterer.fit_predict(self.embeddings)

        n_clusters_found = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        print(f"✓ Found {n_clusters_found} clusters")

        return self

    def create_scatter_plot(
        self,
        color_by: str = 'semantic_type',
        title: str = "Embedding Space Visualization",
        show_legend: bool = True,
        point_size: int = 5,
        opacity: float = 0.7
    ) -> go.Figure:
        """
        Create interactive scatter plot

        Args:
            color_by: Coloring scheme ('semantic_type', 'cluster', 'none')
            title: Plot title
            show_legend: Whether to show legend
            point_size: Size of points
            opacity: Point opacity

        Returns:
            Plotly Figure object
        """
        if self.reduced_embeddings is None:
            raise ValueError("No reduced embeddings. Call reduce_dimensions() first.")

        n_dims = self.reduced_embeddings.shape[1]

        # Prepare data
        data = {
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'entity_id': self.entity_ids,
            'semantic_type': [get_semantic_type(eid) for eid in self.entity_ids]
        }

        if n_dims == 3:
            data['z'] = self.reduced_embeddings[:, 2]

        if self.clusters is not None:
            data['cluster'] = [f"Cluster {c}" if c >= 0 else "Noise"
                              for c in self.clusters]

        df = pd.DataFrame(data)

        # Create hover text
        df['hover_text'] = df.apply(
            lambda row: f"<b>{row['entity_id']}</b><br>"
                       f"Type: {row['semantic_type']}"
                       + (f"<br>{row.get('cluster', '')}" if 'cluster' in row else ""),
            axis=1
        )

        # Determine color column
        if color_by == 'semantic_type':
            color_col = 'semantic_type'
            color_discrete_map = VIZ_CONFIG['semantic_type_colors']
        elif color_by == 'cluster' and 'cluster' in df.columns:
            color_col = 'cluster'
            color_discrete_map = None
        else:
            color_col = None
            color_discrete_map = None

        # Create plot
        if n_dims == 2:
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color=color_col,
                hover_data={'hover_text': True, 'x': False, 'y': False,
                           'entity_id': False, 'semantic_type': False},
                color_discrete_map=color_discrete_map,
                title=title,
                labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
            )

            fig.update_traces(
                marker=dict(size=point_size, opacity=opacity),
                hovertemplate='%{customdata[0]}<extra></extra>'
            )

        else:  # 3D
            fig = px.scatter_3d(
                df,
                x='x',
                y='y',
                z='z',
                color=color_col,
                hover_data={'hover_text': True, 'x': False, 'y': False, 'z': False,
                           'entity_id': False, 'semantic_type': False},
                color_discrete_map=color_discrete_map,
                title=title,
                labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3'}
            )

            fig.update_traces(
                marker=dict(size=point_size, opacity=opacity),
                hovertemplate='%{customdata[0]}<extra></extra>'
            )

        # Update layout
        fig.update_layout(
            width=VIZ_CONFIG['embedding']['plot_width'],
            height=VIZ_CONFIG['embedding']['plot_height'],
            showlegend=show_legend,
            template='plotly_dark',
            hovermode='closest'
        )

        return fig

    def create_density_plot(self) -> go.Figure:
        """
        Create 2D density plot of embeddings

        Returns:
            Plotly Figure object
        """
        if self.reduced_embeddings is None or self.reduced_embeddings.shape[1] != 2:
            raise ValueError("Need 2D reduced embeddings for density plot")

        fig = go.Figure()

        fig.add_trace(go.Histogram2dContour(
            x=self.reduced_embeddings[:, 0],
            y=self.reduced_embeddings[:, 1],
            colorscale='Blues',
            showscale=True
        ))

        fig.update_layout(
            title="Embedding Density Plot",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=VIZ_CONFIG['embedding']['plot_width'],
            height=VIZ_CONFIG['embedding']['plot_height'],
            template='plotly_dark'
        )

        return fig

    def plot_cluster_centroids(self) -> go.Figure:
        """
        Plot cluster centroids in reduced space

        Returns:
            Plotly Figure object
        """
        if self.clusters is None:
            raise ValueError("No clusters. Call cluster_embeddings() first.")

        if self.reduced_embeddings is None:
            raise ValueError("No reduced embeddings. Call reduce_dimensions() first.")

        # Calculate centroids
        unique_clusters = set(self.clusters) - {-1}  # Exclude noise
        centroids = []

        for cluster_id in unique_clusters:
            mask = self.clusters == cluster_id
            centroid = self.reduced_embeddings[mask].mean(axis=0)
            centroids.append({
                'cluster_id': cluster_id,
                'x': centroid[0],
                'y': centroid[1],
                'size': np.sum(mask)
            })

        centroids_df = pd.DataFrame(centroids)

        fig = px.scatter(
            centroids_df,
            x='x',
            y='y',
            size='size',
            text='cluster_id',
            title="Cluster Centroids",
            labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'size': 'Cluster Size'}
        )

        fig.update_traces(
            textposition='top center',
            marker=dict(symbol='diamond', line=dict(width=2, color='white'))
        )

        fig.update_layout(
            width=VIZ_CONFIG['embedding']['plot_width'],
            height=VIZ_CONFIG['embedding']['plot_height'],
            template='plotly_dark'
        )

        return fig

    def save_plot(self, fig: go.Figure, output_path: Union[str, Path]):
        """
        Save plot to HTML file

        Args:
            fig: Plotly Figure object
            output_path: Path to save HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(str(output_path))
        print(f"✓ Saved plot to {output_path}")

    def show_plot(self, fig: go.Figure):
        """
        Display plot in browser or notebook

        Args:
            fig: Plotly Figure object
        """
        fig.show()

    def export_reduced_embeddings(
        self,
        output_path: Union[str, Path],
        format: str = 'csv'
    ):
        """
        Export reduced embeddings to file

        Args:
            output_path: Path to save file
            format: Export format ('csv', 'npy')
        """
        if self.reduced_embeddings is None:
            raise ValueError("No reduced embeddings. Call reduce_dimensions() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            n_dims = self.reduced_embeddings.shape[1]
            df = pd.DataFrame(self.reduced_embeddings,
                            columns=[f'dim_{i}' for i in range(n_dims)])
            df.insert(0, 'entity_id', self.entity_ids)
            df.insert(1, 'semantic_type',
                     [get_semantic_type(eid) for eid in self.entity_ids])

            if self.clusters is not None:
                df['cluster'] = self.clusters

            df.to_csv(output_path, index=False)

        elif format == 'npy':
            np.save(output_path, self.reduced_embeddings)

        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"✓ Exported reduced embeddings to {output_path}")


# Convenience function
def visualize_embeddings(
    embedding_path: Union[str, Path],
    output_path: Union[str, Path] = "embedding_visualization.html",
    method: str = 'umap',
    n_components: int = 2,
    color_by: str = 'semantic_type',
    n_clusters: Optional[int] = None,
    entity_mapping: Optional[Dict] = None,
    show: bool = True
) -> EmbeddingVisualizer:
    """
    Quick visualization of embeddings

    Args:
        embedding_path: Path to embeddings NPY file
        output_path: Path to save HTML output
        method: Reduction method ('umap', 'tsne', 'pca')
        n_components: Number of dimensions (2 or 3)
        color_by: Coloring scheme ('semantic_type', 'cluster')
        n_clusters: Optional number of clusters (enables clustering)
        entity_mapping: Optional entity ID mapping
        show: Whether to display in browser

    Returns:
        EmbeddingVisualizer instance
    """
    viz = EmbeddingVisualizer()
    viz.load_embeddings(embedding_path, entity_mapping)
    viz.reduce_dimensions(method=method, n_components=n_components)

    if n_clusters:
        viz.cluster_embeddings(n_clusters=n_clusters)
        color_by = 'cluster'

    fig = viz.create_scatter_plot(color_by=color_by)

    viz.save_plot(fig, output_path)

    if show:
        viz.show_plot(fig)

    return viz

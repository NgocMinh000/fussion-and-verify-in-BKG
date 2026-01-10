"""
Graph structure visualization using PyVis and NetworkX

This module provides the GraphVisualizer class for creating interactive
network visualizations of the knowledge graph.
"""

import json
import networkx as nx
from pyvis.network import Network
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Tuple
from collections import Counter

from .config import (
    VIZ_CONFIG,
    get_semantic_type,
    get_color_for_semantic_type,
    get_color_for_relation
)


class GraphVisualizer:
    """
    Interactive graph visualization using PyVis

    Features:
    - Node coloring by semantic type
    - Edge coloring by relation type
    - Filtering by confidence scores
    - Highlight new predictions vs existing links
    - Interactive exploration (zoom, pan, search)
    """

    def __init__(
        self,
        width: str = "100%",
        height: str = "800px",
        bgcolor: str = "#222222",
        font_color: str = "white",
        notebook: bool = False
    ):
        """
        Initialize GraphVisualizer

        Args:
            width: Width of the visualization canvas
            height: Height of the visualization canvas
            bgcolor: Background color
            font_color: Font color for labels
            notebook: Whether running in Jupyter notebook
        """
        self.width = width
        self.height = height
        self.bgcolor = bgcolor
        self.font_color = font_color
        self.notebook = notebook

        self.nx_graph = None
        self.pyvis_net = None
        self.nodes_data = []
        self.edges_data = []
        self.metadata = {}

    def load_from_json(self, json_path: Union[str, Path]):
        """
        Load graph from JSON file exported by export_graph_json()

        Args:
            json_path: Path to JSON file
        """
        json_path = Path(json_path)

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.nodes_data = data.get('nodes', [])
        self.edges_data = data.get('edges', [])
        self.metadata = data.get('metadata', {})

        print(f"✓ Loaded graph from {json_path}")
        print(f"  - Nodes: {len(self.nodes_data)}")
        print(f"  - Edges: {len(self.edges_data)}")

        return self

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        entity2index: Optional[Dict] = None,
        index2entity: Optional[Dict] = None
    ):
        """
        Load graph from pandas DataFrame

        Args:
            df: DataFrame with columns [subject, relation, object] or indices
            entity2index: Optional entity to index mapping
            index2entity: Optional index to entity mapping
        """
        # Extract unique nodes
        if 'subject' in df.columns and 'object' in df.columns:
            unique_entities = set(df['subject'].unique()) | set(df['object'].unique())
            self.nodes_data = [
                {
                    "id": entity,
                    "label": entity.split('_')[0] if '_' in entity else entity,
                    "type": get_semantic_type(entity)
                }
                for entity in unique_entities
            ]

            # Create edges
            self.edges_data = [
                {
                    "source": row['subject'],
                    "target": row['object'],
                    "relation": row['relation'] if 'relation' in row else "UNKNOWN"
                }
                for _, row in df.iterrows()
            ]

        else:
            # Assume index-based format
            if index2entity is None:
                raise ValueError("index2entity mapping required for index-based data")

            unique_indices = set(df.iloc[:, 0].unique()) | set(df.iloc[:, 2].unique())
            self.nodes_data = [
                {
                    "id": index2entity[idx],
                    "index": idx,
                    "label": index2entity[idx].split('_')[0] if '_' in index2entity[idx] else index2entity[idx],
                    "type": get_semantic_type(index2entity[idx])
                }
                for idx in unique_indices
            ]

            self.edges_data = [
                {
                    "source": index2entity[int(row.iloc[0])],
                    "target": index2entity[int(row.iloc[2])],
                    "relation": str(row.iloc[1])
                }
                for _, row in df.iterrows()
            ]

        print(f"✓ Loaded graph from DataFrame")
        print(f"  - Nodes: {len(self.nodes_data)}")
        print(f"  - Edges: {len(self.edges_data)}")

        return self

    def filter_by_semantic_types(self, semantic_types: List[str]):
        """
        Filter graph to only show nodes of specific semantic types

        Args:
            semantic_types: List of semantic types to include (e.g., ['dsyn', 'phsu'])
        """
        # Filter nodes
        filtered_node_ids = {
            node['id'] for node in self.nodes_data
            if node.get('type') in semantic_types
        }

        self.nodes_data = [
            node for node in self.nodes_data
            if node['id'] in filtered_node_ids
        ]

        # Filter edges to only include edges between remaining nodes
        self.edges_data = [
            edge for edge in self.edges_data
            if edge['source'] in filtered_node_ids and edge['target'] in filtered_node_ids
        ]

        print(f"✓ Filtered to {len(semantic_types)} semantic types")
        print(f"  - Remaining nodes: {len(self.nodes_data)}")
        print(f"  - Remaining edges: {len(self.edges_data)}")

        return self

    def filter_by_relations(self, relations: List[str]):
        """
        Filter graph to only show specific relation types

        Args:
            relations: List of relation types to include
        """
        # Filter edges
        self.edges_data = [
            edge for edge in self.edges_data
            if edge.get('relation') in relations
        ]

        # Get nodes that are part of remaining edges
        active_node_ids = set()
        for edge in self.edges_data:
            active_node_ids.add(edge['source'])
            active_node_ids.add(edge['target'])

        # Filter nodes
        self.nodes_data = [
            node for node in self.nodes_data
            if node['id'] in active_node_ids
        ]

        print(f"✓ Filtered to {len(relations)} relation types")
        print(f"  - Remaining nodes: {len(self.nodes_data)}")
        print(f"  - Remaining edges: {len(self.edges_data)}")

        return self

    def sample_nodes(self, max_nodes: int = 1000, strategy: str = 'degree'):
        """
        Sample a subset of nodes for visualization (for large graphs)

        Args:
            max_nodes: Maximum number of nodes to include
            strategy: Sampling strategy ('random', 'degree', 'pagerank')
        """
        if len(self.nodes_data) <= max_nodes:
            print(f"Graph already has <= {max_nodes} nodes, no sampling needed")
            return self

        # Build NetworkX graph for sampling
        G = nx.DiGraph()
        for node in self.nodes_data:
            G.add_node(node['id'])
        for edge in self.edges_data:
            G.add_edge(edge['source'], edge['target'])

        # Select nodes based on strategy
        if strategy == 'random':
            import random
            selected_nodes = set(random.sample(list(G.nodes()), max_nodes))

        elif strategy == 'degree':
            # Select nodes with highest degree
            degrees = dict(G.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            selected_nodes = set([node for node, _ in sorted_nodes[:max_nodes]])

        elif strategy == 'pagerank':
            # Select nodes with highest PageRank
            pr = nx.pagerank(G)
            sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            selected_nodes = set([node for node, _ in sorted_nodes[:max_nodes]])

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Filter nodes and edges
        self.nodes_data = [
            node for node in self.nodes_data
            if node['id'] in selected_nodes
        ]

        self.edges_data = [
            edge for edge in self.edges_data
            if edge['source'] in selected_nodes and edge['target'] in selected_nodes
        ]

        print(f"✓ Sampled graph using '{strategy}' strategy")
        print(f"  - Sampled nodes: {len(self.nodes_data)}")
        print(f"  - Sampled edges: {len(self.edges_data)}")

        return self

    def create_pyvis_network(
        self,
        physics_enabled: bool = True,
        layout: str = 'force_directed'
    ):
        """
        Create PyVis Network from loaded data

        Args:
            physics_enabled: Enable physics simulation
            layout: Layout algorithm ('force_directed', 'hierarchical', 'circular')
        """
        # Initialize PyVis network
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            notebook=self.notebook,
            directed=True
        )

        # Add nodes
        for node in self.nodes_data:
            node_id = node['id']
            label = node.get('label', node_id)
            semantic_type = node.get('type', 'default')
            color = get_color_for_semantic_type(semantic_type)

            net.add_node(
                node_id,
                label=label,
                title=f"{node_id}\nType: {semantic_type}",  # Hover text
                color=color,
                size=20,
                font={'color': self.font_color}
            )

        # Add edges
        for edge in self.edges_data:
            source = edge['source']
            target = edge['target']
            relation = edge.get('relation', 'UNKNOWN')
            color = get_color_for_relation(relation)

            net.add_edge(
                source,
                target,
                title=relation,  # Hover text
                color=color,
                arrows='to',
                width=2
            )

        # Configure physics
        if physics_enabled:
            if layout == 'force_directed':
                net.barnes_hut(
                    gravity=-8000,
                    central_gravity=0.3,
                    spring_length=200,
                    spring_strength=0.001,
                    damping=0.09,
                    overlap=0
                )
            elif layout == 'hierarchical':
                net.set_options("""
                var options = {
                  "layout": {
                    "hierarchical": {
                      "enabled": true,
                      "direction": "UD",
                      "sortMethod": "directed"
                    }
                  }
                }
                """)
        else:
            net.toggle_physics(False)

        self.pyvis_net = net
        print(f"✓ Created PyVis network")
        print(f"  - Layout: {layout}")
        print(f"  - Physics: {'enabled' if physics_enabled else 'disabled'}")

        return self

    def render(self, output_path: Union[str, Path]) -> Path:
        """
        Render the visualization to HTML file

        Args:
            output_path: Path to save HTML file

        Returns:
            Path to saved HTML file
        """
        if self.pyvis_net is None:
            self.create_pyvis_network()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.pyvis_net.save_graph(str(output_path))

        print(f"✓ Rendered graph to {output_path}")
        return output_path

    def show(self, output_path: Union[str, Path] = "graph.html"):
        """
        Render and display the visualization in browser

        Args:
            output_path: Path to save HTML file
        """
        output_path = self.render(output_path)

        if self.notebook:
            from IPython.display import IFrame, display
            display(IFrame(str(output_path), width='100%', height=self.height))
        else:
            import webbrowser
            webbrowser.open(f'file://{output_path.absolute()}')

        return output_path

    def get_statistics(self) -> Dict:
        """
        Get graph statistics

        Returns:
            Dictionary of statistics
        """
        # Build NetworkX graph
        G = nx.DiGraph()
        for node in self.nodes_data:
            G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
        for edge in self.edges_data:
            G.add_edge(edge['source'], edge['target'],
                      relation=edge.get('relation'))

        # Calculate statistics
        stats = {
            "num_nodes": len(self.nodes_data),
            "num_edges": len(self.edges_data),
            "density": nx.density(G),
            "avg_degree": sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0,
        }

        # Semantic type distribution
        type_counts = Counter(node.get('type', 'unknown') for node in self.nodes_data)
        stats['semantic_type_distribution'] = dict(type_counts)

        # Relation type distribution
        relation_counts = Counter(edge.get('relation', 'unknown') for edge in self.edges_data)
        stats['relation_distribution'] = dict(relation_counts)

        # Connected components
        if len(G.nodes()) > 0:
            stats['num_connected_components'] = nx.number_weakly_connected_components(G)
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            stats['largest_component_size'] = len(largest_cc)

        return stats

    def print_statistics(self):
        """Print graph statistics"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("Graph Statistics")
        print("=" * 60)
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Edges: {stats['num_edges']}")
        print(f"Density: {stats['density']:.4f}")
        print(f"Average Degree: {stats['avg_degree']:.2f}")

        if 'num_connected_components' in stats:
            print(f"Connected Components: {stats['num_connected_components']}")
            print(f"Largest Component: {stats['largest_component_size']} nodes")

        print(f"\nSemantic Type Distribution:")
        for stype, count in sorted(stats['semantic_type_distribution'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {stype}: {count}")

        print(f"\nRelation Distribution:")
        for rel, count in sorted(stats['relation_distribution'].items(),
                                key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {rel}: {count}")

        print("=" * 60 + "\n")


# Convenience function
def visualize_graph(
    json_path: Union[str, Path],
    output_path: Union[str, Path] = "graph_visualization.html",
    max_nodes: int = 1000,
    semantic_types: Optional[List[str]] = None,
    relations: Optional[List[str]] = None,
    layout: str = 'force_directed',
    show: bool = True
) -> GraphVisualizer:
    """
    Quick visualization of a graph from JSON file

    Args:
        json_path: Path to graph JSON file
        output_path: Path to save HTML output
        max_nodes: Maximum nodes to display (will sample if more)
        semantic_types: Optional list of semantic types to include
        relations: Optional list of relations to include
        layout: Layout algorithm
        show: Whether to open in browser

    Returns:
        GraphVisualizer instance
    """
    viz = GraphVisualizer()
    viz.load_from_json(json_path)

    # Apply filters
    if semantic_types:
        viz.filter_by_semantic_types(semantic_types)
    if relations:
        viz.filter_by_relations(relations)

    # Sample if too many nodes
    if len(viz.nodes_data) > max_nodes:
        viz.sample_nodes(max_nodes, strategy='degree')

    # Create and render
    viz.create_pyvis_network(layout=layout)
    viz.print_statistics()

    if show:
        viz.show(output_path)
    else:
        viz.render(output_path)

    return viz

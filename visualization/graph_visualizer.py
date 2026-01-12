"""
Knowledge graph structure visualization.
Visualizes the graph topology, relation distribution, and entity connectivity.
"""

import argparse
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import networkx as nx


def load_graph_data(viz_dir):
    """Load graph structure data."""
    print(f"Loading graph data from {viz_dir}...")

    data = {}

    # Load graph structure
    with open(f'{viz_dir}/graph_structure.json', 'r') as f:
        data['graph'] = json.load(f)

    # Load mappings
    with open(f'{viz_dir}/index2entity.pkl', 'rb') as f:
        data['index2entity'] = pickle.load(f)
    with open(f'{viz_dir}/index2relation.pkl', 'rb') as f:
        data['index2relation'] = pickle.load(f)

    # Load statistics
    with open(f'{viz_dir}/statistics.json', 'r') as f:
        data['statistics'] = json.load(f)

    print(f"  ✓ Loaded graph with {data['graph']['num_nodes']} nodes and {data['graph']['num_edges']} edges")

    return data


def plot_relation_distribution(data, output_path):
    """Plot distribution of relations in the knowledge graph."""
    print("\nGenerating relation distribution plot...")

    relation_counts = data['statistics']['relation_distribution']

    # Sort by count
    relations = list(relation_counts.keys())
    counts = list(relation_counts.values())
    sorted_indices = np.argsort(counts)[::-1]
    relations_sorted = [relations[i] for i in sorted_indices]
    counts_sorted = [counts[i] for i in sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Truncate relation names for display
    relation_labels = [rel[:40] + '...' if len(rel) > 40 else rel for rel in relations_sorted]

    bars = ax.bar(range(len(counts_sorted)), counts_sorted, color='steelblue', alpha=0.7)

    ax.set_xlabel('Relations', fontsize=12)
    ax.set_ylabel('Number of Triples', fontsize=12)
    ax.set_title('Relation Distribution in Knowledge Graph', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(relation_labels)))
    ax.set_xticklabels(relation_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def plot_entity_degree_distribution(data, output_path):
    """Plot entity degree distribution."""
    print("\nGenerating entity degree distribution plot...")

    train_triples = np.array(data['graph']['train_triples'])

    # Calculate degrees
    entity_degrees = {}
    for triple in train_triples:
        head_idx = int(triple[0])
        tail_idx = int(triple[2])

        entity_degrees[head_idx] = entity_degrees.get(head_idx, 0) + 1
        entity_degrees[tail_idx] = entity_degrees.get(tail_idx, 0) + 1

    degrees = list(entity_degrees.values())
    degree_counts = Counter(degrees)

    # Sort by degree
    degrees_sorted = sorted(degree_counts.keys())
    counts_sorted = [degree_counts[d] for d in degrees_sorted]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax1.bar(degrees_sorted, counts_sorted, color='darkorange', alpha=0.7, width=0.8)
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('Number of Entities', fontsize=12)
    ax1.set_title('Entity Degree Distribution (Linear Scale)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Log scale
    ax2.bar(degrees_sorted, counts_sorted, color='darkorange', alpha=0.7, width=0.8)
    ax2.set_xlabel('Degree', fontsize=12)
    ax2.set_ylabel('Number of Entities (log scale)', fontsize=12)
    ax2.set_title('Entity Degree Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text
    stats = data['statistics']['entity_degree_stats']
    stats_text = f"Mean: {stats['mean']:.1f}\nMedian: {stats['median']:.1f}\nStd: {stats['std']:.1f}\nMax: {stats['max']}"
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def plot_graph_sample(data, output_path, num_nodes=50, num_edges=100):
    """Plot a sample of the knowledge graph."""
    print(f"\nGenerating graph sample visualization ({num_nodes} nodes, {num_edges} edges)...")

    train_triples = np.array(data['graph']['train_triples'])

    # Sample triples
    if len(train_triples) > num_edges:
        sample_indices = np.random.choice(len(train_triples), num_edges, replace=False)
        triples_sample = train_triples[sample_indices]
    else:
        triples_sample = train_triples

    # Build NetworkX graph
    G = nx.DiGraph()

    for triple in triples_sample:
        head_idx = int(triple[0])
        tail_idx = int(triple[2])
        rel_idx = int(triple[1])

        head_name = str(data['index2entity'].get(head_idx, f'E{head_idx}'))[:20]
        tail_name = str(data['index2entity'].get(tail_idx, f'E{tail_idx}'))[:20]
        rel_name = str(data['index2relation'].get(rel_idx, f'R{rel_idx}'))[:15]

        G.add_edge(head_name, tail_name, relation=rel_name)

    # Limit nodes if too many
    if len(G.nodes()) > num_nodes:
        # Keep nodes with highest degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:num_nodes]
        G = G.subgraph(top_nodes).copy()

    print(f"  Graph sample: {len(G.nodes())} nodes, {len(G.edges())} edges")

    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Plot
    plt.figure(figsize=(16, 12))

    # Draw nodes
    node_sizes = [300 + G.degree(node) * 50 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color='lightblue', alpha=0.7,
                          edgecolors='darkblue', linewidths=1.5)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrows=True, arrowsize=15, alpha=0.5,
                          connectionstyle='arc3,rad=0.1')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Draw edge labels (relations)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, alpha=0.7)

    plt.title(f'Knowledge Graph Sample\n({len(G.nodes())} entities, {len(G.edges())} relations)',
              fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def plot_train_test_split(data, output_path):
    """Plot train/test split statistics."""
    print("\nGenerating train/test split visualization...")

    train_size = len(data['graph']['train_triples'])
    test_size = len(data['graph']['test_triples'])

    fig, ax = plt.subplots(figsize=(8, 6))

    splits = ['Train', 'Test']
    sizes = [train_size, test_size]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(splits, sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Number of Triples', fontsize=12)
    ax.set_title('Train/Test Data Split', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:,}\n({size/(train_size+test_size)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def visualize_all(viz_dir, output_dir, sample_nodes=50, sample_edges=100):
    """Generate all graph visualizations."""
    print("="*60)
    print("GRAPH STRUCTURE VISUALIZATION")
    print("="*60)

    # Load data
    data = load_graph_data(viz_dir)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    plot_relation_distribution(data, f'{output_dir}/relation_distribution.png')
    plot_entity_degree_distribution(data, f'{output_dir}/entity_degree_distribution.png')
    plot_train_test_split(data, f'{output_dir}/train_test_split.png')
    plot_graph_sample(data, f'{output_dir}/graph_sample.png', sample_nodes, sample_edges)

    print("\n" + "="*60)
    print(f"✓ ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Visualize FuseLinker knowledge graph structure')
    parser.add_argument(
        '--viz_dir',
        default='fuselinker-complex/suppkg/visualization_outputs',
        help='Directory containing exported visualization data (e.g., fuselinker-complex/suppkg/visualization_outputs)'
    )
    parser.add_argument(
        '--output_dir',
        default='fuselinker-complex/suppkg/visualization_plots',
        help='Directory to save visualization plots'
    )
    parser.add_argument(
        '--sample_nodes',
        type=int,
        default=50,
        help='Number of nodes to include in graph sample'
    )
    parser.add_argument(
        '--sample_edges',
        type=int,
        default=100,
        help='Number of edges to sample from graph'
    )

    args = parser.parse_args()

    visualize_all(args.viz_dir, args.output_dir, args.sample_nodes, args.sample_edges)


if __name__ == '__main__':
    main()

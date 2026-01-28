"""
Example usage scripts for FuseLinker Visualization

This file contains practical examples for using the visualization module.
Run these examples after training FuseLinker and exporting visualization data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.graph_visualizer import GraphVisualizer, visualize_graph
from visualization.embedding_visualizer import EmbeddingVisualizer, visualize_embeddings
from visualization.export_utils import load_mappings


def example_1_basic_graph_viz():
    """
    Example 1: Basic graph visualization
    Visualize the training graph with default settings
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Graph Visualization")
    print("=" * 60 + "\n")

    viz = visualize_graph(
        json_path='suppkg/visualization_outputs/train_graph.json',
        output_path='outputs/basic_graph.html',
        max_nodes=1000,
        show=True
    )

    print("\n✓ Graph visualization saved to outputs/basic_graph.html")


def example_2_filtered_graph():
    """
    Example 2: Filtered graph visualization
    Show only diseases and their treatments
    """
    print("\n" + "=" * 60)
    print("Example 2: Disease-Drug Treatment Network")
    print("=" * 60 + "\n")

    viz = GraphVisualizer()
    viz.load_from_json('suppkg/visualization_outputs/train_graph.json')

    # Filter to diseases and drugs
    viz.filter_by_semantic_types(['dsyn', 'phsu'])

    # Filter to treatment relations
    viz.filter_by_relations(['TREATS', 'PREVENTS'])

    # Sample top 300 nodes by degree
    viz.sample_nodes(300, strategy='degree')

    # Create and show
    viz.create_pyvis_network(layout='hierarchical')
    viz.print_statistics()
    viz.show('outputs/disease_drug_network.html')

    print("\n✓ Disease-drug network saved to outputs/disease_drug_network.html")


def example_3_embedding_viz_2d():
    """
    Example 3: 2D embedding visualization with UMAP
    """
    print("\n" + "=" * 60)
    print("Example 3: 2D Embedding Space (UMAP)")
    print("=" * 60 + "\n")

    viz = visualize_embeddings(
        embedding_path='suppkg/visualization_outputs/node_embeddings.npy',
        output_path='outputs/embeddings_2d_umap.html',
        method='umap',
        n_components=2,
        color_by='semantic_type',
        show=True
    )

    print("\n✓ 2D UMAP visualization saved to outputs/embeddings_2d_umap.html")


def example_4_embedding_viz_3d():
    """
    Example 4: 3D embedding visualization with t-SNE
    """
    print("\n" + "=" * 60)
    print("Example 4: 3D Embedding Space (t-SNE)")
    print("=" * 60 + "\n")

    viz = EmbeddingVisualizer()
    viz.load_embeddings('suppkg/visualization_outputs/node_embeddings.npy')

    # Reduce to 3D with t-SNE
    viz.reduce_dimensions(method='tsne', n_components=3, perplexity=50)

    # Create 3D scatter plot
    fig = viz.create_scatter_plot(
        color_by='semantic_type',
        title="3D Embedding Space (t-SNE)",
        point_size=4,
        opacity=0.8
    )

    viz.save_plot(fig, 'outputs/embeddings_3d_tsne.html')
    viz.show_plot(fig)

    print("\n✓ 3D t-SNE visualization saved to outputs/embeddings_3d_tsne.html")


def example_5_clustering_analysis():
    """
    Example 5: Clustering analysis of embeddings
    """
    print("\n" + "=" * 60)
    print("Example 5: Clustering Analysis")
    print("=" * 60 + "\n")

    viz = EmbeddingVisualizer()
    viz.load_embeddings('suppkg/visualization_outputs/node_embeddings.npy')

    # Reduce dimensions
    viz.reduce_dimensions(method='umap', n_components=2, n_neighbors=30, min_dist=0.1)

    # Cluster with K-Means
    viz.cluster_embeddings(method='kmeans', n_clusters=10)

    # Create visualizations
    scatter = viz.create_scatter_plot(color_by='cluster', title="Clusters in Embedding Space")
    centroids = viz.plot_cluster_centroids()

    # Save
    viz.save_plot(scatter, 'outputs/clusters_scatter.html')
    viz.save_plot(centroids, 'outputs/cluster_centroids.html')

    # Export clustered data
    viz.export_reduced_embeddings('outputs/clustered_embeddings.csv', format='csv')

    print("\n✓ Cluster visualizations saved to outputs/")
    print("  - clusters_scatter.html")
    print("  - cluster_centroids.html")
    print("  - clustered_embeddings.csv")


def example_6_compare_semantic_types():
    """
    Example 6: Compare different semantic types in separate graphs
    """
    print("\n" + "=" * 60)
    print("Example 6: Compare Semantic Type Networks")
    print("=" * 60 + "\n")

    semantic_groups = {
        'diseases': ['dsyn', 'sosy'],
        'chemicals': ['orch', 'phsu', 'bacs'],
        'biological': ['aapp', 'gngm', 'nnon']
    }

    for group_name, types in semantic_groups.items():
        viz = GraphVisualizer()
        viz.load_from_json('suppkg/visualization_outputs/train_graph.json')

        viz.filter_by_semantic_types(types)
        viz.sample_nodes(400, strategy='degree')
        viz.create_pyvis_network()

        output_path = f'outputs/{group_name}_network.html'
        viz.render(output_path)

        print(f"✓ {group_name.capitalize()} network saved to {output_path}")
        viz.print_statistics()


def example_7_analyze_specific_relation():
    """
    Example 7: Analyze a specific relation type
    """
    print("\n" + "=" * 60)
    print("Example 7: Analyze TREATS Relation")
    print("=" * 60 + "\n")

    viz = GraphVisualizer()
    viz.load_from_json('suppkg/visualization_outputs/train_graph.json')

    # Focus on TREATS relation
    viz.filter_by_relations(['TREATS'])

    # Get statistics
    stats = viz.get_statistics()

    print(f"\nTREATS Relation Statistics:")
    print(f"  Total edges: {stats['num_edges']}")
    print(f"  Unique nodes: {stats['num_nodes']}")
    print(f"  Semantic type distribution:")
    for stype, count in stats['semantic_type_distribution'].items():
        print(f"    {stype}: {count}")

    # Visualize
    viz.sample_nodes(500, strategy='degree')
    viz.create_pyvis_network(layout='hierarchical')
    viz.show('outputs/treats_relation_network.html')

    print("\n✓ TREATS network saved to outputs/treats_relation_network.html")


def example_8_embedding_density_plot():
    """
    Example 8: Create density plot of embeddings
    """
    print("\n" + "=" * 60)
    print("Example 8: Embedding Density Heatmap")
    print("=" * 60 + "\n")

    viz = EmbeddingVisualizer()
    viz.load_embeddings('suppkg/visualization_outputs/node_embeddings.npy')

    # Reduce to 2D
    viz.reduce_dimensions(method='umap', n_components=2)

    # Create density plot
    density_fig = viz.create_density_plot()

    viz.save_plot(density_fig, 'outputs/embedding_density.html')
    viz.show_plot(density_fig)

    print("\n✓ Density plot saved to outputs/embedding_density.html")


def run_all_examples():
    """Run all examples sequentially"""
    print("\n" + "=" * 60)
    print("Running All Visualization Examples")
    print("=" * 60)

    # Create output directory
    Path('outputs').mkdir(exist_ok=True)

    try:
        example_1_basic_graph_viz()
        example_2_filtered_graph()
        example_3_embedding_viz_2d()
        example_4_embedding_viz_3d()
        example_5_clustering_analysis()
        example_6_compare_semantic_types()
        example_7_analyze_specific_relation()
        example_8_embedding_density_plot()

        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("Check the 'outputs/' directory for results")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have:")
        print("  1. Trained the FuseLinker model")
        print("  2. Exported visualization data to suppkg/visualization_outputs/")
        print("  3. Installed all required packages (pip install -r visualization/requirements.txt)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FuseLinker Visualization Examples')
    parser.add_argument('--example', type=int, choices=range(1, 9),
                       help='Run specific example (1-8)')
    parser.add_argument('--all', action='store_true',
                       help='Run all examples')

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        example_func = globals()[f'example_{args.example}_{["basic_graph_viz", "filtered_graph", "embedding_viz_2d", "embedding_viz_3d", "clustering_analysis", "compare_semantic_types", "analyze_specific_relation", "embedding_density_plot"][args.example-1]}']
        example_func()
    else:
        print("Usage:")
        print("  python example_usage.py --all           # Run all examples")
        print("  python example_usage.py --example 1     # Run specific example")
        print("\nAvailable examples:")
        print("  1. Basic graph visualization")
        print("  2. Filtered graph (diseases and drugs)")
        print("  3. 2D embedding visualization (UMAP)")
        print("  4. 3D embedding visualization (t-SNE)")
        print("  5. Clustering analysis")
        print("  6. Compare semantic type networks")
        print("  7. Analyze specific relation")
        print("  8. Embedding density plot")

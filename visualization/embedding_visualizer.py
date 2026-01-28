"""
Embedding visualization tool.
Visualizes entity and relation embeddings using t-SNE and PCA.
"""

import argparse
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def load_visualization_data(viz_dir):
    """Load exported visualization data."""
    print(f"Loading data from {viz_dir}...")

    data = {}

    # Load embeddings
    data['entity_embeddings'] = np.load(f'{viz_dir}/entity_embeddings.npy')
    data['relation_embeddings'] = np.load(f'{viz_dir}/relation_embeddings.npy')

    # Load optional fused components
    learned_path = Path(f'{viz_dir}/learned_embeddings.npy')
    if learned_path.exists():
        data['learned_embeddings'] = np.load(learned_path)

    text_path = Path(f'{viz_dir}/text_embeddings.npy')
    if text_path.exists():
        data['text_embeddings'] = np.load(text_path)

    domain_path = Path(f'{viz_dir}/domain_embeddings.npy')
    if domain_path.exists():
        data['domain_embeddings'] = np.load(domain_path)

    # Load mappings
    with open(f'{viz_dir}/index2entity.pkl', 'rb') as f:
        data['index2entity'] = pickle.load(f)
    with open(f'{viz_dir}/index2relation.pkl', 'rb') as f:
        data['index2relation'] = pickle.load(f)

    # Load stats
    with open(f'{viz_dir}/statistics.json', 'r') as f:
        data['statistics'] = json.load(f)

    print(f"  ✓ Loaded {len(data['entity_embeddings'])} entity embeddings")
    print(f"  ✓ Loaded {len(data['relation_embeddings'])} relation embeddings")

    return data


def visualize_embeddings_tsne(embeddings, labels, title, output_path, sample_size=1000):
    """Visualize embeddings using t-SNE."""
    print(f"\nGenerating t-SNE visualization: {title}")

    # Sample if too many points
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[indices]
        labels_sample = [labels[i] for i in indices]
        print(f"  Sampling {sample_size} points from {len(embeddings)} total")
    else:
        embeddings_sample = embeddings
        labels_sample = labels

    # Apply t-SNE
    print(f"  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1))
    embeddings_2d = tsne.fit_transform(embeddings_sample)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)

    # Annotate a few points
    num_labels = min(20, len(labels_sample))
    for i in np.random.choice(len(labels_sample), num_labels, replace=False):
        plt.annotate(
            labels_sample[i][:30],  # Truncate long labels
            xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7
        )

    plt.title(f'{title} (t-SNE)', fontsize=14)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def visualize_embeddings_pca(embeddings, labels, title, output_path, sample_size=1000):
    """Visualize embeddings using PCA."""
    print(f"\nGenerating PCA visualization: {title}")

    # Sample if too many points
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[indices]
        labels_sample = [labels[i] for i in indices]
        print(f"  Sampling {sample_size} points from {len(embeddings)} total")
    else:
        embeddings_sample = embeddings
        labels_sample = labels

    # Apply PCA
    print(f"  Running PCA...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_sample)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)

    # Annotate a few points
    num_labels = min(20, len(labels_sample))
    for i in np.random.choice(len(labels_sample), num_labels, replace=False):
        plt.annotate(
            labels_sample[i][:30],
            xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7
        )

    plt.title(f'{title} (PCA - {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def compare_fused_components(data, output_dir):
    """Compare text, domain, and fused embeddings."""
    if 'text_embeddings' not in data or 'domain_embeddings' not in data:
        print("\n⚠ Text or domain embeddings not found, skipping fusion comparison")
        return

    print("\n" + "="*60)
    print("COMPARING FUSED EMBEDDING COMPONENTS")
    print("="*60)

    # Get entity labels
    entity_labels = [str(data['index2entity'].get(i, f'entity_{i}'))[:30]
                     for i in range(len(data['entity_embeddings']))]

    # Visualize each component
    sample_size = min(500, len(data['entity_embeddings']))

    # Text embeddings
    visualize_embeddings_pca(
        data['text_embeddings'],
        entity_labels,
        'Text Embeddings (PubMedBERT)',
        f'{output_dir}/text_embeddings_pca.png',
        sample_size=sample_size
    )

    # Domain embeddings
    visualize_embeddings_pca(
        data['domain_embeddings'],
        entity_labels,
        'Domain Knowledge Embeddings (Poincaré)',
        f'{output_dir}/domain_embeddings_pca.png',
        sample_size=sample_size
    )

    # Learned embeddings
    if 'learned_embeddings' in data:
        visualize_embeddings_pca(
            data['learned_embeddings'],
            entity_labels,
            'Learned Embeddings (Task-specific)',
            f'{output_dir}/learned_embeddings_pca.png',
            sample_size=sample_size
        )

    # Final fused embeddings
    visualize_embeddings_pca(
        data['entity_embeddings'],
        entity_labels,
        'Fused Embeddings (Final)',
        f'{output_dir}/fused_embeddings_pca.png',
        sample_size=sample_size
    )

    print("\n✓ Fusion comparison complete!")


def visualize_all(viz_dir, output_dir, method='both', sample_size=1000):
    """Generate all embedding visualizations."""
    print("="*60)
    print("EMBEDDING VISUALIZATION")
    print("="*60)

    # Load data
    data = load_visualization_data(viz_dir)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get labels
    entity_labels = [str(data['index2entity'].get(i, f'entity_{i}'))[:30]
                     for i in range(len(data['entity_embeddings']))]
    relation_labels = [str(data['index2relation'].get(i, f'rel_{i}'))[:50]
                       for i in range(len(data['relation_embeddings']))]

    # Visualize entity embeddings
    if method in ['tsne', 'both']:
        visualize_embeddings_tsne(
            data['entity_embeddings'],
            entity_labels,
            'Entity Embeddings',
            f'{output_dir}/entity_embeddings_tsne.png',
            sample_size=sample_size
        )

    if method in ['pca', 'both']:
        visualize_embeddings_pca(
            data['entity_embeddings'],
            entity_labels,
            'Entity Embeddings',
            f'{output_dir}/entity_embeddings_pca.png',
            sample_size=sample_size
        )

    # Visualize relation embeddings
    if method in ['tsne', 'both']:
        visualize_embeddings_tsne(
            data['relation_embeddings'],
            relation_labels,
            'Relation Embeddings',
            f'{output_dir}/relation_embeddings_tsne.png',
            sample_size=min(sample_size, len(data['relation_embeddings']))
        )

    if method in ['pca', 'both']:
        visualize_embeddings_pca(
            data['relation_embeddings'],
            relation_labels,
            'Relation Embeddings',
            f'{output_dir}/relation_embeddings_pca.png',
            sample_size=min(sample_size, len(data['relation_embeddings']))
        )

    # Compare fused components if available
    compare_fused_components(data, output_dir)

    print("\n" + "="*60)
    print(f"✓ ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Visualize FuseLinker embeddings')
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
        '--method',
        choices=['tsne', 'pca', 'both'],
        default='both',
        help='Dimensionality reduction method'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=1000,
        help='Number of points to sample for visualization'
    )

    args = parser.parse_args()

    visualize_all(args.viz_dir, args.output_dir, args.method, args.sample_size)


if __name__ == '__main__':
    main()

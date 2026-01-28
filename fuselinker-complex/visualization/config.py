"""
Configuration settings for FuseLinker visualization
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "suppkg"
VIZ_OUTPUT_DIR = BASE_DIR / "visualization" / "outputs"

# Create output directory if it doesn't exist
VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
VIZ_CONFIG = {
    # Graph visualization
    'graph': {
        'width': '100%',
        'height': '800px',
        'bgcolor': '#222222',
        'font_color': 'white',
        'node_size_range': (10, 50),
        'edge_width_range': (1, 5),
        'layout': 'force_directed',  # Options: force_directed, hierarchical, circular
        'physics_enabled': True,
        'smooth_edges': True,
    },

    # Embedding visualization
    'embedding': {
        'reduction_method': 'umap',  # Options: umap, tsne, pca
        'n_components': 2,  # 2D or 3D
        'n_neighbors': 15,  # UMAP parameter
        'min_dist': 0.1,  # UMAP parameter
        'perplexity': 30,  # t-SNE parameter
        'random_state': 42,
        'plot_width': 1000,
        'plot_height': 800,
    },

    # Dashboard settings
    'dashboard': {
        'host': '0.0.0.0',
        'port': 8050,
        'debug': True,
        'theme': 'bootstrap',
        'update_interval': 1000,  # ms
    },

    # Colors for semantic types
    'semantic_type_colors': {
        'aapp': '#FF6B6B',  # Amino Acid, Peptide, or Protein - Red
        'nnon': '#4ECDC4',  # Nucleic Acid, Nucleoside, or Nucleotide - Teal
        'orch': '#95E1D3',  # Organic Chemical - Light green
        'bacs': '#F38181',  # Biologically Active Substance - Pink
        'phsu': '#AA96DA',  # Pharmacologic Substance - Purple
        'gngm': '#FCBAD3',  # Gene or Genome - Light pink
        'dsyn': '#FFA07A',  # Disease or Syndrome - Orange
        'sosy': '#FFD93D',  # Sign or Symptom - Yellow
        'default': '#A8DADC'  # Default color
    },

    # Colors for relation types
    'relation_colors': {
        'INTERACTS_WITH': '#1f77b4',
        'CAUSES': '#ff7f0e',
        'AFFECTS': '#2ca02c',
        'COEXISTS_WITH': '#d62728',
        'TREATS': '#9467bd',
        'INHIBITS': '#8c564b',
        'ASSOCIATED_WITH': '#e377c2',
        'AUGMENTS': '#7f7f7f',
        'PRODUCES': '#bcbd22',
        'PREVENTS': '#17becf',
        'STIMULATES': '#ff9896',
        'DISRUPTS': '#98df8a',
        'PREDISPOSES': '#c5b0d5',
        'MANIFESTATION_OF': '#c49c94',
        'COMPLICATES': '#f7b6d2',
        'default': '#c7c7c7'
    },

    # Filtering defaults
    'filters': {
        'confidence_threshold': 0.5,
        'max_nodes_display': 1000,
        'max_edges_display': 5000,
        'top_k_predictions': 100,
    },

    # Export formats
    'export': {
        'graph_format': 'json',  # json, gml, graphml
        'embedding_format': 'npy',  # npy, csv
        'prediction_format': 'csv',  # csv, json
        'image_format': 'png',  # png, svg, pdf
        'image_dpi': 300,
    }
}

# Relation type mappings
RELATION_TYPES = [
    'INTERACTS_WITH',
    'CAUSES',
    'AFFECTS',
    'COEXISTS_WITH',
    'TREATS',
    'INHIBITS',
    'ASSOCIATED_WITH',
    'AUGMENTS',
    'PRODUCES',
    'PREVENTS',
    'STIMULATES',
    'DISRUPTS',
    'PREDISPOSES',
    'MANIFESTATION_OF',
    'COMPLICATES'
]

# Semantic type mappings
SEMANTIC_TYPES = {
    'aapp': 'Amino Acid, Peptide, or Protein',
    'nnon': 'Nucleic Acid, Nucleoside, or Nucleotide',
    'orch': 'Organic Chemical',
    'bacs': 'Biologically Active Substance',
    'phsu': 'Pharmacologic Substance',
    'gngm': 'Gene or Genome',
    'dsyn': 'Disease or Syndrome',
    'sosy': 'Sign or Symptom'
}

def get_semantic_type(entity_id):
    """Extract semantic type from entity ID (e.g., 'C0003250_aapp' -> 'aapp')"""
    if '_' in entity_id:
        return entity_id.split('_')[-1]
    return 'default'

def get_color_for_semantic_type(semantic_type):
    """Get color for a semantic type"""
    return VIZ_CONFIG['semantic_type_colors'].get(semantic_type,
                                                   VIZ_CONFIG['semantic_type_colors']['default'])

def get_color_for_relation(relation):
    """Get color for a relation type"""
    return VIZ_CONFIG['relation_colors'].get(relation,
                                             VIZ_CONFIG['relation_colors']['default'])

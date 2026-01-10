"""
Utility functions for exporting FuseLinker data for visualization

This module provides functions to:
- Export embeddings (node and relation)
- Export graph structure to JSON
- Export predictions with scores
- Save metadata for entities and relations
"""

import json
import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

from .config import VIZ_OUTPUT_DIR, get_semantic_type


def export_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    entity_mapping: Optional[Dict] = None,
    metadata: Optional[Dict] = None
):
    """
    Export node embeddings to NPY file with optional metadata

    Args:
        embeddings: Node embeddings tensor/array (num_nodes x hidden_dim)
        output_path: Path to save the embeddings
        entity_mapping: Optional dict mapping indices to entity IDs
        metadata: Optional metadata to save alongside embeddings
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy if torch tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Save embeddings
    np.save(output_path, embeddings)
    print(f"✓ Saved embeddings to {output_path} | Shape: {embeddings.shape}")

    # Save metadata if provided
    if metadata or entity_mapping:
        meta_path = output_path.with_suffix('.meta.json')
        meta_data = metadata or {}
        if entity_mapping:
            meta_data['entity_mapping'] = entity_mapping
        meta_data['shape'] = embeddings.shape

        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
        print(f"✓ Saved metadata to {meta_path}")


def export_relation_embeddings(
    relation_weights: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    relation_mapping: Optional[Dict] = None
):
    """
    Export relation embeddings to NPY file

    Args:
        relation_weights: Relation weight tensor/array (num_relations x hidden_dim)
        output_path: Path to save the relation embeddings
        relation_mapping: Optional dict mapping indices to relation types
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy if torch tensor
    if isinstance(relation_weights, torch.Tensor):
        relation_weights = relation_weights.detach().cpu().numpy()

    # Save relation embeddings
    np.save(output_path, relation_weights)
    print(f"✓ Saved relation embeddings to {output_path} | Shape: {relation_weights.shape}")

    # Save relation mapping if provided
    if relation_mapping:
        mapping_path = output_path.with_suffix('.mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(relation_mapping, f, indent=2)
        print(f"✓ Saved relation mapping to {mapping_path}")


def export_graph_json(
    triplets: Union[np.ndarray, List],
    entity2index: Dict,
    index2entity: Dict,
    relation2index: Dict,
    index2relation: Dict,
    output_path: Union[str, Path],
    include_metadata: bool = True
):
    """
    Export graph structure to JSON format for visualization

    Args:
        triplets: Array/list of triplets (subject, relation, object)
        entity2index: Entity to index mapping
        index2entity: Index to entity mapping
        relation2index: Relation to index mapping
        index2relation: Index to relation mapping
        output_path: Path to save the JSON file
        include_metadata: Whether to include entity metadata

    Output JSON structure:
    {
        "nodes": [{"id": "C0003250_aapp", "label": "...", "type": "aapp", ...}],
        "edges": [{"source": "C0003250_aapp", "target": "C0011503_nnon", "relation": "INTERACTS_WITH"}],
        "metadata": {...}
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert triplets to numpy if needed
    if isinstance(triplets, torch.Tensor):
        triplets = triplets.cpu().numpy()

    # Extract unique nodes
    unique_node_indices = set()
    for triplet in triplets:
        unique_node_indices.add(int(triplet[0]))
        unique_node_indices.add(int(triplet[2]))

    # Create nodes list
    nodes = []
    for node_idx in tqdm(unique_node_indices, desc="Processing nodes"):
        entity_id = index2entity[node_idx]
        semantic_type = get_semantic_type(entity_id)

        node = {
            "id": entity_id,
            "index": node_idx,
            "type": semantic_type,
            "label": entity_id.split('_')[0] if '_' in entity_id else entity_id
        }

        nodes.append(node)

    # Create edges list
    edges = []
    for triplet in tqdm(triplets, desc="Processing edges"):
        subject_idx, relation_idx, object_idx = int(triplet[0]), int(triplet[1]), int(triplet[2])

        subject_id = index2entity[subject_idx]
        object_id = index2entity[object_idx]

        # Get relation name
        relation_tuple = index2relation[relation_idx]
        if isinstance(relation_tuple, tuple):
            relation_name = relation_tuple[1] if len(relation_tuple) > 1 else str(relation_tuple)
        else:
            relation_name = str(relation_tuple)

        edge = {
            "source": subject_id,
            "target": object_id,
            "relation": relation_name,
            "relation_idx": relation_idx
        }

        edges.append(edge)

    # Create metadata
    metadata = {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_relations": len(relation2index),
        "semantic_types": list(set(node["type"] for node in nodes))
    }

    # Combine into final structure
    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata
    }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=2)

    print(f"✓ Saved graph to {output_path}")
    print(f"  - Nodes: {metadata['num_nodes']}")
    print(f"  - Edges: {metadata['num_edges']}")
    print(f"  - Relations: {metadata['num_relations']}")

    return graph_data


def export_predictions(
    predictions: List[Tuple],
    scores: Union[np.ndarray, List],
    index2entity: Dict,
    index2relation: Dict,
    output_path: Union[str, Path],
    top_k: Optional[int] = None,
    include_scores: bool = True
):
    """
    Export predicted links with confidence scores to CSV

    Args:
        predictions: List of predicted triplets (subject_idx, relation_idx, object_idx)
        scores: Confidence scores for each prediction
        index2entity: Index to entity mapping
        index2relation: Index to relation mapping
        output_path: Path to save the CSV file
        top_k: Optional limit to top K predictions by score
        include_scores: Whether to include confidence scores
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert scores to numpy if needed
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # Create dataframe
    data = []
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        subject_idx, relation_idx, object_idx = pred

        subject = index2entity[int(subject_idx)]
        object_entity = index2entity[int(object_idx)]

        # Get relation name
        relation_tuple = index2relation[int(relation_idx)]
        if isinstance(relation_tuple, tuple):
            relation = relation_tuple[1] if len(relation_tuple) > 1 else str(relation_tuple)
        else:
            relation = str(relation_tuple)

        row = {
            'subject': subject,
            'relation': relation,
            'object': object_entity,
            'subject_type': get_semantic_type(subject),
            'object_type': get_semantic_type(object_entity),
        }

        if include_scores:
            row['confidence_score'] = float(score)

        data.append(row)

    df = pd.DataFrame(data)

    # Sort by score if available
    if include_scores and 'confidence_score' in df.columns:
        df = df.sort_values('confidence_score', ascending=False)

    # Limit to top K if specified
    if top_k:
        df = df.head(top_k)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"✓ Saved {len(df)} predictions to {output_path}")
    if include_scores:
        print(f"  - Score range: [{df['confidence_score'].min():.4f}, {df['confidence_score'].max():.4f}]")

    return df


def save_training_snapshot(
    iteration: int,
    embeddings: torch.Tensor,
    relation_weights: torch.Tensor,
    loss: float,
    metrics: Dict,
    output_dir: Union[str, Path]
):
    """
    Save a snapshot of training state at a specific iteration

    Args:
        iteration: Current training iteration
        embeddings: Current node embeddings
        relation_weights: Current relation weights
        loss: Current loss value
        metrics: Dict of evaluation metrics (MR, MRR, Hits@K)
        output_dir: Directory to save snapshots
    """
    output_dir = Path(output_dir)
    snapshot_dir = output_dir / f"snapshot_iter_{iteration}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    export_embeddings(
        embeddings,
        snapshot_dir / "embeddings.npy",
        metadata={
            "iteration": iteration,
            "loss": loss,
            "metrics": metrics
        }
    )

    # Save relation weights
    export_relation_embeddings(
        relation_weights,
        snapshot_dir / "relation_weights.npy"
    )

    # Save metrics
    metrics_path = snapshot_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            "iteration": iteration,
            "loss": loss,
            **metrics
        }, f, indent=2)

    print(f"✓ Saved training snapshot at iteration {iteration}")


def load_mappings(data_dir: Union[str, Path]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load entity and relation mappings from pickle files

    Args:
        data_dir: Directory containing the mapping files

    Returns:
        entity2index, index2entity, relation2index, index2relation
    """
    data_dir = Path(data_dir)

    with open(data_dir / 'entity2index.pkl', 'rb') as f:
        entity2index = pickle.load(f)

    with open(data_dir / 'index2entity.pkl', 'rb') as f:
        index2entity = pickle.load(f)

    with open(data_dir / 'relation2index.pkl', 'rb') as f:
        relation2index = pickle.load(f)

    with open(data_dir / 'index2relation.pkl', 'rb') as f:
        index2relation = pickle.load(f)

    print(f"✓ Loaded mappings from {data_dir}")
    print(f"  - Entities: {len(entity2index)}")
    print(f"  - Relations: {len(relation2index)}")

    return entity2index, index2entity, relation2index, index2relation


def export_full_visualization_data(
    model,
    graph,
    node_ids,
    rel_ids,
    norm,
    train_data,
    test_data,
    entity2index,
    index2entity,
    relation2index,
    index2relation,
    output_dir: Union[str, Path],
    device: torch.device = torch.device('cpu')
):
    """
    Export all data needed for visualization in one go

    Args:
        model: Trained LinkPredict model
        graph: DGL graph
        node_ids: Node IDs tensor
        rel_ids: Relation IDs tensor
        norm: Normalization tensor
        train_data: Training triplets
        test_data: Test triplets
        entity2index: Entity to index mapping
        index2entity: Index to entity mapping
        relation2index: Relation to index mapping
        index2relation: Index to relation mapping
        output_dir: Directory to save all outputs
        device: Device (CPU/GPU)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Exporting visualization data...")
    print("=" * 60)

    # 1. Export embeddings
    print("\n[1/5] Exporting node embeddings...")
    with torch.no_grad():
        embeddings = model(graph.to(device), node_ids.to(device),
                          rel_ids.to(device), norm.to(device))

    export_embeddings(
        embeddings,
        output_dir / "node_embeddings.npy",
        entity_mapping={str(k): v for k, v in index2entity.items()},
        metadata={"num_nodes": len(index2entity)}
    )

    # 2. Export relation embeddings
    print("\n[2/5] Exporting relation embeddings...")
    export_relation_embeddings(
        model.relation_weights,
        output_dir / "relation_embeddings.npy",
        relation_mapping={str(k): str(v) for k, v in index2relation.items()}
    )

    # 3. Export training graph
    print("\n[3/5] Exporting training graph...")
    export_graph_json(
        train_data,
        entity2index,
        index2entity,
        relation2index,
        index2relation,
        output_dir / "train_graph.json"
    )

    # 4. Export test graph
    print("\n[4/5] Exporting test graph...")
    export_graph_json(
        test_data,
        entity2index,
        index2entity,
        relation2index,
        index2relation,
        output_dir / "test_graph.json"
    )

    # 5. Save mappings for easy access
    print("\n[5/5] Saving mappings...")
    mappings = {
        "entity2index": {k: int(v) for k, v in entity2index.items()},
        "index2entity": {int(k): v for k, v in index2entity.items()},
        "relation2index": {str(k): int(v) for k, v in relation2index.items()},
        "index2relation": {int(k): str(v) for k, v in index2relation.items()}
    }

    with open(output_dir / "mappings.json", 'w') as f:
        json.dump(mappings, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ All visualization data exported to {output_dir}")
    print("=" * 60)

"""
Export utilities for visualization data.
Extracts embeddings, predictions, and graph structure from trained models.
"""

import os
import json
import numpy as np
import torch
import pickle
from pathlib import Path


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
    output_dir,
    device
):
    """
    Export comprehensive visualization data from trained model.

    Args:
        model: Trained LinkPredict model
        graph: DGL graph
        node_ids: Node ID tensor
        rel_ids: Relation ID tensor
        norm: Normalization tensor
        train_data: Training triples (numpy array)
        test_data: Test triples (numpy array)
        entity2index: Entity name to index mapping
        index2entity: Index to entity name mapping
        relation2index: Relation to index mapping
        index2relation: Index to relation mapping
        output_dir: Directory to save visualization data
        device: torch device
    """

    print(f"\nExporting visualization data to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export entity embeddings
    print("  1/7 Extracting entity embeddings...")
    with torch.no_grad():
        entity_embeddings = model(graph, node_ids, rel_ids, norm)
        entity_embeddings_np = entity_embeddings.cpu().numpy()

    np.save(f'{output_dir}/entity_embeddings.npy', entity_embeddings_np)
    print(f"      ✓ Saved entity embeddings: shape {entity_embeddings_np.shape}")

    # 2. Export relation embeddings
    print("  2/7 Extracting relation embeddings...")
    relation_embeddings_np = model.relation_weights.detach().cpu().numpy()
    np.save(f'{output_dir}/relation_embeddings.npy', relation_embeddings_np)
    print(f"      ✓ Saved relation embeddings: shape {relation_embeddings_np.shape}")

    # 3. Export fused embeddings components (if available)
    print("  3/7 Extracting fused embedding components...")
    if hasattr(model, 'entity_embeddings'):
        # Get the individual embedding components
        learned_emb = model.entity_embeddings.weight.detach().cpu().numpy()
        np.save(f'{output_dir}/learned_embeddings.npy', learned_emb)
        print(f"      ✓ Saved learned embeddings: shape {learned_emb.shape}")

        # Try to get pretrained components
        if hasattr(model, 'text_embeddings') and model.text_embeddings is not None:
            text_emb = model.text_embeddings.cpu().numpy()
            np.save(f'{output_dir}/text_embeddings.npy', text_emb)
            print(f"      ✓ Saved text embeddings: shape {text_emb.shape}")

        if hasattr(model, 'domain_embeddings') and model.domain_embeddings is not None:
            domain_emb = model.domain_embeddings.cpu().numpy()
            np.save(f'{output_dir}/domain_embeddings.npy', domain_emb)
            print(f"      ✓ Saved domain embeddings: shape {domain_emb.shape}")
    else:
        print("      (No fused embedding components found)")

    # 4. Export graph structure
    print("  4/7 Exporting graph structure...")
    graph_data = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'train_triples': train_data.tolist(),
        'test_triples': test_data.tolist()
    }

    with open(f'{output_dir}/graph_structure.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f"      ✓ Saved graph structure: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")

    # 5. Export entity and relation mappings
    print("  5/7 Exporting entity/relation mappings...")
    with open(f'{output_dir}/entity2index.pkl', 'wb') as f:
        pickle.dump(entity2index, f)
    with open(f'{output_dir}/index2entity.pkl', 'wb') as f:
        pickle.dump(index2entity, f)
    with open(f'{output_dir}/relation2index.pkl', 'wb') as f:
        pickle.dump(relation2index, f)
    with open(f'{output_dir}/index2relation.pkl', 'wb') as f:
        pickle.dump(index2relation, f)
    print(f"      ✓ Saved mappings: {len(entity2index)} entities, {len(relation2index)} relations")

    # 6. Calculate and export statistics
    print("  6/7 Calculating statistics...")

    # Relation distribution in training data
    relation_counts = {}
    for triple in train_data:
        rel_idx = int(triple[1])
        rel_name = str(index2relation.get(rel_idx, f"rel_{rel_idx}"))
        relation_counts[rel_name] = relation_counts.get(rel_name, 0) + 1

    # Entity degree distribution
    entity_degrees = np.zeros(len(entity2index))
    for triple in train_data:
        entity_degrees[int(triple[0])] += 1  # head entity
        entity_degrees[int(triple[2])] += 1  # tail entity

    stats = {
        'relation_distribution': relation_counts,
        'entity_degree_stats': {
            'mean': float(np.mean(entity_degrees)),
            'std': float(np.std(entity_degrees)),
            'min': int(np.min(entity_degrees)),
            'max': int(np.max(entity_degrees)),
            'median': float(np.median(entity_degrees))
        },
        'embedding_stats': {
            'entity_emb_mean': float(np.mean(entity_embeddings_np)),
            'entity_emb_std': float(np.std(entity_embeddings_np)),
            'relation_emb_mean': float(np.mean(relation_embeddings_np)),
            'relation_emb_std': float(np.std(relation_embeddings_np))
        }
    }

    with open(f'{output_dir}/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"      ✓ Saved statistics")

    # 7. Export model configuration
    print("  7/7 Exporting model configuration...")
    config = {
        'model_type': model.__class__.__name__,
        'num_entities': len(entity2index),
        'num_relations': len(relation2index),
        'embedding_dim': entity_embeddings_np.shape[1],
        'has_text_embeddings': hasattr(model, 'text_embeddings') and model.text_embeddings is not None,
        'has_domain_embeddings': hasattr(model, 'domain_embeddings') and model.domain_embeddings is not None,
    }

    with open(f'{output_dir}/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"      ✓ Saved model configuration")

    print(f"\n✓ Export complete! All data saved to {output_dir}/")
    print(f"\nFiles created:")
    print(f"  - entity_embeddings.npy ({entity_embeddings_np.shape})")
    print(f"  - relation_embeddings.npy ({relation_embeddings_np.shape})")
    if hasattr(model, 'entity_embeddings'):
        print(f"  - learned_embeddings.npy")
    print(f"  - graph_structure.json")
    print(f"  - entity2index.pkl, index2entity.pkl")
    print(f"  - relation2index.pkl, index2relation.pkl")
    print(f"  - statistics.json")
    print(f"  - model_config.json")

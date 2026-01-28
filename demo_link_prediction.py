#!/usr/bin/env python3
"""
Demo Link Prediction - Interactive Prediction Demo

This script demonstrates link prediction by masking head or tail entities
and showing top-K predictions with the rank of ground truth.

Usage:
    # Demo with test triples
    python demo_link_prediction.py --model_dir fuselinker-complex --data primekg

    # Demo with specific triple
    python demo_link_prediction.py --model_dir fuselinker-complex --data primekg \
        --head "Gene_APOE" --relation "ppi" --tail "Gene_APP"

    # Show more predictions
    python demo_link_prediction.py --model_dir fuselinker-complex --data primekg --top_k 20
"""

import argparse
import torch
import numpy as np
import pickle
import json
from pathlib import Path
import random


def load_model_and_data(model_dir, data_dir):
    """Load trained model and visualization data."""
    model_dir = Path(model_dir)
    data_dir = Path(data_dir)
    viz_dir = data_dir / 'visualization_outputs'

    print("Loading model and data...")
    print(f"Model directory: {model_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Visualization directory: {viz_dir}")

    # Load mappings
    with open(data_dir / 'entity2index.pkl', 'rb') as f:
        entity2index = pickle.load(f)
    with open(data_dir / 'index2entity.pkl', 'rb') as f:
        index2entity = pickle.load(f)
    with open(data_dir / 'relation2index.pkl', 'rb') as f:
        relation2index = pickle.load(f)
    with open(data_dir / 'index2relation.pkl', 'rb') as f:
        index2relation = pickle.load(f)

    # Load embeddings
    entity_embeddings = np.load(viz_dir / 'node_embeddings.npy')
    relation_embeddings = np.load(viz_dir / 'relation_embeddings.npy')

    # Load test triples
    with open(viz_dir / 'test_graph.json', 'r') as f:
        test_graph = json.load(f)
        test_triples = np.array(test_graph['triples'])

    # Load train triples for filtering
    with open(viz_dir / 'train_graph.json', 'r') as f:
        train_graph = json.load(f)
        train_triples = np.array(train_graph['triples'])

    print(f"✓ Loaded {len(entity2index)} entities")
    print(f"✓ Loaded {len(relation2index)} relations")
    print(f"✓ Loaded embeddings: {entity_embeddings.shape}")
    print(f"✓ Loaded {len(test_triples)} test triples")
    print(f"✓ Loaded {len(train_triples)} train triples")

    return {
        'entity2index': entity2index,
        'index2entity': index2entity,
        'relation2index': relation2index,
        'index2relation': index2relation,
        'entity_embeddings': entity_embeddings,
        'relation_embeddings': relation_embeddings,
        'test_triples': test_triples,
        'train_triples': train_triples
    }


def score_triples(head_emb, rel_emb, tail_emb):
    """
    Score triples using DistMult scoring function.
    Score = <h, r, t> (element-wise product and sum)
    """
    return np.sum(head_emb * rel_emb * tail_emb, axis=-1)


def predict_tail(head_idx, rel_idx, data, top_k=10, filter_mode='filtered'):
    """
    Predict tail entity given head and relation.

    Args:
        head_idx: Head entity index
        rel_idx: Relation index
        data: Dictionary containing embeddings and data
        top_k: Number of top predictions to return
        filter_mode: 'filtered' or 'raw'

    Returns:
        List of (entity_idx, entity_name, score, rank) tuples
    """
    entity_embeddings = data['entity_embeddings']
    relation_embeddings = data['relation_embeddings']
    index2entity = data['index2entity']

    # Get embeddings
    head_emb = entity_embeddings[head_idx]
    rel_emb = relation_embeddings[rel_idx]

    # Score all possible tails
    scores = np.zeros(len(entity_embeddings))
    for tail_idx in range(len(entity_embeddings)):
        tail_emb = entity_embeddings[tail_idx]
        scores[tail_idx] = score_triples(head_emb, rel_emb, tail_emb)

    # Filter out existing triples if filtered mode
    if filter_mode == 'filtered':
        # Mask out all known (head, rel, ?) triples except the test triple
        for triple in data['train_triples']:
            if triple[0] == head_idx and triple[1] == rel_idx:
                scores[triple[2]] = -np.inf

    # Get top-K predictions
    ranked_indices = np.argsort(-scores)  # Descending order
    top_predictions = []

    for rank, entity_idx in enumerate(ranked_indices[:top_k], 1):
        entity_name = index2entity.get(entity_idx, f'entity_{entity_idx}')
        score = scores[entity_idx]
        top_predictions.append({
            'entity_idx': int(entity_idx),
            'entity_name': entity_name,
            'score': float(score),
            'rank': rank
        })

    return top_predictions, ranked_indices


def predict_head(tail_idx, rel_idx, data, top_k=10, filter_mode='filtered'):
    """
    Predict head entity given tail and relation.
    """
    entity_embeddings = data['entity_embeddings']
    relation_embeddings = data['relation_embeddings']
    index2entity = data['index2entity']

    # Get embeddings
    tail_emb = entity_embeddings[tail_idx]
    rel_emb = relation_embeddings[rel_idx]

    # Score all possible heads
    scores = np.zeros(len(entity_embeddings))
    for head_idx in range(len(entity_embeddings)):
        head_emb = entity_embeddings[head_idx]
        scores[head_idx] = score_triples(head_emb, rel_emb, tail_emb)

    # Filter out existing triples if filtered mode
    if filter_mode == 'filtered':
        # Mask out all known (?, rel, tail) triples except the test triple
        for triple in data['train_triples']:
            if triple[1] == rel_idx and triple[2] == tail_idx:
                scores[triple[0]] = -np.inf

    # Get top-K predictions
    ranked_indices = np.argsort(-scores)  # Descending order
    top_predictions = []

    for rank, entity_idx in enumerate(ranked_indices[:top_k], 1):
        entity_name = index2entity.get(entity_idx, f'entity_{entity_idx}')
        score = scores[entity_idx]
        top_predictions.append({
            'entity_idx': int(entity_idx),
            'entity_name': entity_name,
            'score': float(score),
            'rank': rank
        })

    return top_predictions, ranked_indices


def demo_single_triple(head_name, rel_name, tail_name, data, top_k=10):
    """Demo prediction for a single triple."""
    entity2index = data['entity2index']
    relation2index = data['relation2index']
    index2entity = data['index2entity']
    index2relation = data['index2relation']

    print("\n" + "=" * 80)
    print("LINK PREDICTION DEMO")
    print("=" * 80)

    # Convert names to indices
    head_idx = entity2index.get(head_name)
    tail_idx = entity2index.get(tail_name)

    # Find relation index (handle FuseLinker format)
    rel_idx = None
    for idx, rel_tuple in index2relation.items():
        if rel_tuple[1] == rel_name:  # ('head', relation_name, 'tail')
            rel_idx = idx
            break

    if head_idx is None or tail_idx is None or rel_idx is None:
        print(f"✗ Could not find entities/relation:")
        print(f"  Head '{head_name}': {head_idx}")
        print(f"  Relation '{rel_name}': {rel_idx}")
        print(f"  Tail '{tail_name}': {tail_idx}")
        return

    print(f"\nTest Triple:")
    print(f"  ({head_name}) --[{rel_name}]--> ({tail_name})")
    print(f"  Indices: ({head_idx}, {rel_idx}, {tail_idx})")

    # Task 1: Predict tail (mask tail, predict it)
    print(f"\n{'─' * 80}")
    print(f"Task 1: Predict TAIL")
    print(f"Given: ({head_name}) --[{rel_name}]--> (?)")
    print(f"{'─' * 80}")

    tail_predictions, tail_ranks = predict_tail(head_idx, rel_idx, data, top_k=top_k)

    print(f"\nTop-{top_k} Tail Predictions:")
    for i, pred in enumerate(tail_predictions, 1):
        marker = "✓✓✓" if pred['entity_idx'] == tail_idx else ""
        print(f"  {i:2d}. {pred['entity_name']:40s} (score: {pred['score']:.4f}) {marker}")

    # Find rank of ground truth
    gt_rank = np.where(tail_ranks == tail_idx)[0][0] + 1
    print(f"\n✓ Ground Truth: {tail_name}")
    print(f"✓ Rank of Ground Truth: {gt_rank} / {len(data['entity_embeddings'])}")

    if gt_rank == 1:
        print("  🎉 Perfect! Ground truth is rank 1!")
    elif gt_rank <= 3:
        print(f"  ✓ Good! Ground truth in top 3")
    elif gt_rank <= 10:
        print(f"  ○ Fair. Ground truth in top 10")
    else:
        print(f"  ✗ Ground truth rank is {gt_rank}")

    # Task 2: Predict head (mask head, predict it)
    print(f"\n{'─' * 80}")
    print(f"Task 2: Predict HEAD")
    print(f"Given: (?) --[{rel_name}]--> ({tail_name})")
    print(f"{'─' * 80}")

    head_predictions, head_ranks = predict_head(tail_idx, rel_idx, data, top_k=top_k)

    print(f"\nTop-{top_k} Head Predictions:")
    for i, pred in enumerate(head_predictions, 1):
        marker = "✓✓✓" if pred['entity_idx'] == head_idx else ""
        print(f"  {i:2d}. {pred['entity_name']:40s} (score: {pred['score']:.4f}) {marker}")

    # Find rank of ground truth
    gt_rank = np.where(head_ranks == head_idx)[0][0] + 1
    print(f"\n✓ Ground Truth: {head_name}")
    print(f"✓ Rank of Ground Truth: {gt_rank} / {len(data['entity_embeddings'])}")

    if gt_rank == 1:
        print("  🎉 Perfect! Ground truth is rank 1!")
    elif gt_rank <= 3:
        print(f"  ✓ Good! Ground truth in top 3")
    elif gt_rank <= 10:
        print(f"  ○ Fair. Ground truth in top 10")
    else:
        print(f"  ✗ Ground truth rank is {gt_rank}")

    print("\n" + "=" * 80)


def demo_random_triples(data, num_samples=5, top_k=10):
    """Demo prediction for random test triples."""
    test_triples = data['test_triples']
    index2entity = data['index2entity']
    index2relation = data['index2relation']

    # Sample random triples
    sample_indices = random.sample(range(len(test_triples)), min(num_samples, len(test_triples)))

    print("\n" + "=" * 80)
    print(f"LINK PREDICTION DEMO - {num_samples} Random Test Triples")
    print("=" * 80)

    for i, idx in enumerate(sample_indices, 1):
        triple = test_triples[idx]
        head_idx, rel_idx, tail_idx = int(triple[0]), int(triple[1]), int(triple[2])

        head_name = index2entity.get(head_idx, f'entity_{head_idx}')
        tail_name = index2entity.get(tail_idx, f'entity_{tail_idx}')
        rel_tuple = index2relation.get(rel_idx, ('head', f'rel_{rel_idx}', 'tail'))
        rel_name = rel_tuple[1]

        print(f"\n{'═' * 80}")
        print(f"Sample {i}/{num_samples}: ({head_name}) --[{rel_name}]--> ({tail_name})")
        print(f"{'═' * 80}")

        # Predict tail
        print(f"\n🎯 Task: Predict TAIL given ({head_name}, {rel_name}, ?)")
        tail_predictions, tail_ranks = predict_tail(head_idx, rel_idx, data, top_k=top_k)

        print(f"\nTop-{top_k} Predictions:")
        for j, pred in enumerate(tail_predictions[:5], 1):  # Show top 5
            marker = "✓✓✓ [CORRECT]" if pred['entity_idx'] == tail_idx else ""
            print(f"  {j}. {pred['entity_name']:40s} (score: {pred['score']:.4f}) {marker}")

        if top_k > 5:
            print(f"  ... ({top_k - 5} more predictions)")

        # Find rank of ground truth
        gt_rank = np.where(tail_ranks == tail_idx)[0][0] + 1
        print(f"\n✓ Ground Truth: {tail_name}")
        print(f"✓ Rank: {gt_rank} / {len(data['entity_embeddings'])}")

        if gt_rank == 1:
            print("  🎉 Rank 1 - Perfect prediction!")
        elif gt_rank <= 3:
            print(f"  ✓ Rank {gt_rank} - In top 3 (Hits@3)")
        elif gt_rank <= 10:
            print(f"  ○ Rank {gt_rank} - In top 10 (Hits@10)")
        else:
            print(f"  ✗ Rank {gt_rank} - Outside top 10")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Demo link prediction with trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model_dir', type=str, required=True,
                       help='Model directory (e.g., fuselinker-complex)')
    parser.add_argument('--data', type=str, required=True,
                       help='Data directory (e.g., primekg, suppkg)')
    parser.add_argument('--head', type=str,
                       help='Head entity name for specific triple demo')
    parser.add_argument('--relation', type=str,
                       help='Relation name for specific triple demo')
    parser.add_argument('--tail', type=str,
                       help='Tail entity name for specific triple demo')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of random samples to demo')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top predictions to show')

    args = parser.parse_args()

    # Load model and data
    data = load_model_and_data(args.model_dir, args.data)

    # Demo mode
    if args.head and args.relation and args.tail:
        # Specific triple demo
        demo_single_triple(args.head, args.relation, args.tail, data, top_k=args.top_k)
    else:
        # Random triples demo
        demo_random_triples(data, num_samples=args.num_samples, top_k=args.top_k)


if __name__ == '__main__':
    main()

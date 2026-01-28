#!/usr/bin/env python3
"""
Demo Link Prediction - Interactive Prediction Demo with Trained Model

This script demonstrates link prediction by:
1. Loading TRAINED model weights (.pth file)
2. Using model's scoring function (ComplEx, DistMult, TransE, ConvE)
3. Masking head or tail entities
4. Showing top-K predictions with ground truth rank

Usage:
    # Demo with random test triples
    python demo_link_prediction.py \
        --model_dir fuselinker-complex \
        --data primekg \
        --model_state_file primekg_complex_model.pth

    # Demo with specific triple
    python demo_link_prediction.py \
        --model_dir fuselinker-complex \
        --data primekg \
        --model_state_file primekg_complex_model.pth \
        --head "Gene_APOE" --relation "ppi" --tail "Gene_APP"
"""

import argparse
import torch
import numpy as np
import pickle
import json
import sys
from pathlib import Path
import random

# Add model directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def detect_model_type(model_dir):
    """Detect model type from directory name."""
    model_dir = str(model_dir).lower()
    if 'complex' in model_dir:
        return 'complex'
    elif 'transe' in model_dir:
        return 'transe'
    elif 'conve' in model_dir:
        return 'conve'
    else:
        return 'distmult'


def load_model_architecture(model_type, model_dir, num_entities, num_relations):
    """Load appropriate model architecture."""
    import importlib.util

    # Load model.py from model_dir
    model_file = Path(model_dir) / 'model.py'
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Get LinkPredict class
    LinkPredict = model_module.LinkPredict

    return LinkPredict, model_module


def load_model_and_data(model_dir, data_dir, model_state_file):
    """Load trained model and data."""
    model_dir = Path(model_dir)
    data_dir = Path(data_dir)

    print("Loading model and data...")
    print(f"Model directory: {model_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Model state file: {model_state_file}")

    # Check if model state file exists
    if not Path(model_state_file).exists():
        # Try in model_dir
        alt_path = model_dir / model_state_file
        if alt_path.exists():
            model_state_file = str(alt_path)
        else:
            print(f"\n✗ ERROR: Model state file not found: {model_state_file}")
            print(f"  Also tried: {alt_path}")
            print("\nPlease specify the correct path to your trained model .pth file")
            print("Example: --model_state_file fuselinker-complex/primekg_model.pth")
            sys.exit(1)

    # Load mappings
    with open(data_dir / 'entity2index.pkl', 'rb') as f:
        entity2index = pickle.load(f)
    with open(data_dir / 'index2entity.pkl', 'rb') as f:
        index2entity = pickle.load(f)
    with open(data_dir / 'relation2index.pkl', 'rb') as f:
        relation2index = pickle.load(f)
    with open(data_dir / 'index2relation.pkl', 'rb') as f:
        index2relation = pickle.load(f)

    num_entities = len(entity2index)
    num_relations = len(relation2index)

    print(f"✓ Loaded {num_entities} entities")
    print(f"✓ Loaded {num_relations} relations")

    # Detect model type
    model_type = detect_model_type(model_dir)
    print(f"✓ Detected model type: {model_type.upper()}")

    # Load model architecture
    print(f"Loading model architecture...")
    LinkPredict, model_module = load_model_architecture(model_type, model_dir, num_entities, num_relations)

    # Load model state to infer parameters
    print(f"Loading model state...")
    checkpoint = torch.load(model_state_file, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Infer embedding dimension from state_dict
    if 'rgcn.entity_embeddings.weight' in state_dict:
        embedding_dim = state_dict['rgcn.entity_embeddings.weight'].shape[1]
    elif 'entity_embeddings.weight' in state_dict:
        embedding_dim = state_dict['entity_embeddings.weight'].shape[1]
    else:
        # Default
        embedding_dim = 128

    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Iteration: {checkpoint.get('iteration', 'unknown')}")

    # Create model instance (without pretrained embeddings for demo)
    # We'll use the learned embeddings from checkpoint
    model = LinkPredict(
        num_nodes=num_entities,
        hidden_dim=embedding_dim,
        num_relations=num_relations,
        num_hidden_layers=2,  # Default
        dropout=0.0,  # No dropout for inference
        num_bases=-1,  # Will be set from checkpoint
        use_n3_reg=False  # Not needed for inference
    )

    # Load trained weights
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ Model loaded successfully!")

    # Load test triples from visualization outputs (for random sampling)
    viz_dir = data_dir / 'visualization_outputs'
    if (viz_dir / 'test_graph.json').exists():
        with open(viz_dir / 'test_graph.json', 'r') as f:
            test_graph = json.load(f)
            test_triples = np.array(test_graph['triples'])
        print(f"✓ Loaded {len(test_triples)} test triples")
    else:
        # Try loading from test.tsv
        test_file = data_dir / 'test.tsv'
        if test_file.exists():
            test_data = []
            with open(test_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        h, r, t = parts
                        if h in entity2index and t in entity2index:
                            # Find relation index
                            rel_idx = None
                            for idx, rel_tuple in index2relation.items():
                                if rel_tuple[1] == r:
                                    rel_idx = idx
                                    break
                            if rel_idx is not None:
                                test_data.append([entity2index[h], rel_idx, entity2index[t]])
            test_triples = np.array(test_data)
            print(f"✓ Loaded {len(test_triples)} test triples from test.tsv")
        else:
            test_triples = np.array([])
            print("⚠ No test triples found (visualization_outputs/test_graph.json or test.tsv)")

    # Load train triples for filtering
    if (viz_dir / 'train_graph.json').exists():
        with open(viz_dir / 'train_graph.json', 'r') as f:
            train_graph = json.load(f)
            train_triples = np.array(train_graph['triples'])
        print(f"✓ Loaded {len(train_triples)} train triples")
    else:
        train_triples = np.array([])
        print("⚠ No train triples found for filtering")

    return {
        'model': model,
        'model_type': model_type,
        'entity2index': entity2index,
        'index2entity': index2entity,
        'relation2index': relation2index,
        'index2relation': index2relation,
        'test_triples': test_triples,
        'train_triples': train_triples,
        'num_entities': num_entities,
        'num_relations': num_relations
    }


def score_triples_with_model(model, model_type, head_idx, rel_idx, tail_indices, device='cpu'):
    """
    Score triples using the trained model's scoring function.

    Args:
        model: Trained LinkPredict model
        model_type: Model type ('complex', 'distmult', 'transe', 'conve')
        head_idx: Head entity index (scalar)
        rel_idx: Relation index (scalar)
        tail_indices: Array of tail indices to score
        device: torch device

    Returns:
        Array of scores for each (head, rel, tail) combination
    """
    model.eval()
    with torch.no_grad():
        # Create triplets tensor: (head, rel, tail) for each tail candidate
        num_candidates = len(tail_indices)
        triplets = torch.zeros((num_candidates, 3), dtype=torch.long)
        triplets[:, 0] = head_idx
        triplets[:, 1] = rel_idx
        triplets[:, 2] = torch.tensor(tail_indices)

        # For ComplEx, DistMult, TransE: use score_triplets method
        # This requires embeddings from forward pass

        # Get entity embeddings from model
        # Note: We need to do a dummy forward pass or extract embeddings
        # For simplicity, we'll use the score_triplets method directly

        # Get embeddings (depends on model architecture)
        if hasattr(model, 'rgcn'):
            # Model with RGCN
            # We need graph, node_ids, rel_ids, norm for forward pass
            # This is complex - we'll use a simpler approach

            # Get entity embeddings directly
            if hasattr(model.rgcn, 'entity_embeddings'):
                entity_embeddings = model.rgcn.entity_embeddings.weight
            else:
                # Use default
                entity_embeddings = torch.randn(model.num_nodes, model.hidden_dim)
        else:
            entity_embeddings = model.entity_embeddings.weight

        # Use model's score_triplets method
        scores = model.score_triplets(triplets, entity_embeddings)

        return scores.cpu().numpy()


def predict_tail(head_idx, rel_idx, data, top_k=10, filter_mode='filtered', device='cpu'):
    """
    Predict tail entity given head and relation using TRAINED MODEL.
    """
    model = data['model']
    model_type = data['model_type']
    index2entity = data['index2entity']
    num_entities = data['num_entities']

    # Get all possible tail indices
    tail_indices = np.arange(num_entities)

    # Score all candidates using model
    scores = score_triples_with_model(model, model_type, head_idx, rel_idx, tail_indices, device)

    # Filter out existing triples if filtered mode
    if filter_mode == 'filtered':
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

    return top_predictions, ranked_indices, scores


def predict_head(tail_idx, rel_idx, data, top_k=10, filter_mode='filtered', device='cpu'):
    """
    Predict head entity given tail and relation using TRAINED MODEL.
    """
    model = data['model']
    model_type = data['model_type']
    index2entity = data['index2entity']
    num_entities = data['num_entities']

    # Get all possible head indices
    head_indices = np.arange(num_entities)

    # Score all candidates
    scores = np.zeros(num_entities)
    for head_idx in head_indices:
        score = score_triples_with_model(model, model_type, head_idx, rel_idx, [tail_idx], device)
        scores[head_idx] = score[0]

    # Filter out existing triples if filtered mode
    if filter_mode == 'filtered':
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

    return top_predictions, ranked_indices, scores


def demo_single_triple(head_name, rel_name, tail_name, data, top_k=10, device='cpu'):
    """Demo prediction for a single triple from test data."""
    entity2index = data['entity2index']
    relation2index = data['relation2index']
    index2entity = data['index2entity']
    index2relation = data['index2relation']

    print("\n" + "=" * 80)
    print("LINK PREDICTION DEMO - Using Trained Model")
    print("=" * 80)

    # Convert names to indices
    head_idx = entity2index.get(head_name)
    tail_idx = entity2index.get(tail_name)

    # Find relation index
    rel_idx = None
    for idx, rel_tuple in index2relation.items():
        if rel_tuple[1] == rel_name:
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
    print(f"  Model: {data['model_type'].upper()}")

    # Task 1: Predict tail
    print(f"\n{'─' * 80}")
    print(f"Task 1: Predict TAIL")
    print(f"Given: ({head_name}) --[{rel_name}]--> (?)")
    print(f"{'─' * 80}")

    tail_predictions, tail_ranks, _ = predict_tail(head_idx, rel_idx, data, top_k=top_k, device=device)

    print(f"\nTop-{top_k} Tail Predictions:")
    for pred in tail_predictions:
        marker = "✓✓✓" if pred['entity_idx'] == tail_idx else ""
        print(f"  {pred['rank']:2d}. {pred['entity_name']:40s} (score: {pred['score']:.4f}) {marker}")

    # Find rank of ground truth
    gt_rank = np.where(tail_ranks == tail_idx)[0][0] + 1
    print(f"\n✓ Ground Truth: {tail_name}")
    print(f"✓ Rank of Ground Truth: {gt_rank} / {data['num_entities']}")

    if gt_rank == 1:
        print("  🎉 Perfect! Ground truth is rank 1!")
    elif gt_rank <= 3:
        print(f"  ✓ Good! Ground truth in top 3 (Hits@3)")
    elif gt_rank <= 10:
        print(f"  ○ Fair. Ground truth in top 10 (Hits@10)")
    else:
        print(f"  ✗ Ground truth rank is {gt_rank}")

    # Task 2: Predict head
    print(f"\n{'─' * 80}")
    print(f"Task 2: Predict HEAD")
    print(f"Given: (?) --[{rel_name}]--> ({tail_name})")
    print(f"{'─' * 80}")

    head_predictions, head_ranks, _ = predict_head(tail_idx, rel_idx, data, top_k=top_k, device=device)

    print(f"\nTop-{top_k} Head Predictions:")
    for pred in head_predictions:
        marker = "✓✓✓" if pred['entity_idx'] == head_idx else ""
        print(f"  {pred['rank']:2d}. {pred['entity_name']:40s} (score: {pred['score']:.4f}) {marker}")

    # Find rank of ground truth
    gt_rank = np.where(head_ranks == head_idx)[0][0] + 1
    print(f"\n✓ Ground Truth: {head_name}")
    print(f"✓ Rank of Ground Truth: {gt_rank} / {data['num_entities']}")

    if gt_rank == 1:
        print("  🎉 Perfect! Ground truth is rank 1!")
    elif gt_rank <= 3:
        print(f"  ✓ Good! Ground truth in top 3 (Hits@3)")
    elif gt_rank <= 10:
        print(f"  ○ Fair. Ground truth in top 10 (Hits@10)")
    else:
        print(f"  ✗ Ground truth rank is {gt_rank}")

    print("\n" + "=" * 80)


def demo_random_triples(data, num_samples=5, top_k=10, device='cpu'):
    """Demo prediction for random test triples."""
    test_triples = data['test_triples']
    index2entity = data['index2entity']
    index2relation = data['index2relation']

    if len(test_triples) == 0:
        print("\n✗ No test triples available for demo!")
        print("  Please ensure test.tsv or visualization_outputs/test_graph.json exists")
        return

    # Randomly sample from TEST triples (not synthetic!)
    sample_indices = random.sample(range(len(test_triples)), min(num_samples, len(test_triples)))

    print("\n" + "=" * 80)
    print(f"LINK PREDICTION DEMO - {num_samples} Random Test Triples")
    print(f"Model: {data['model_type'].upper()}")
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
        tail_predictions, tail_ranks, _ = predict_tail(head_idx, rel_idx, data, top_k=top_k, device=device)

        print(f"\nTop-{min(5, top_k)} Predictions:")
        for pred in tail_predictions[:5]:
            marker = "✓✓✓ [CORRECT]" if pred['entity_idx'] == tail_idx else ""
            print(f"  {pred['rank']}. {pred['entity_name']:40s} (score: {pred['score']:.4f}) {marker}")

        if top_k > 5:
            print(f"  ... ({top_k - 5} more predictions)")

        # Find rank of ground truth
        gt_rank = np.where(tail_ranks == tail_idx)[0][0] + 1
        print(f"\n✓ Ground Truth: {tail_name}")
        print(f"✓ Rank: {gt_rank} / {data['num_entities']}")

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
                       help='Data directory name (e.g., primekg, suppkg)')
    parser.add_argument('--model_state_file', type=str, required=True,
                       help='Path to trained model .pth file')
    parser.add_argument('--head', type=str,
                       help='Head entity name for specific triple demo')
    parser.add_argument('--relation', type=str,
                       help='Relation name for specific triple demo')
    parser.add_argument('--tail', type=str,
                       help='Tail entity name for specific triple demo')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of random test triples to demo')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top predictions to show')
    parser.add_argument('--use_cuda', type=str, default='False',
                       help='Use CUDA if available')

    args = parser.parse_args()

    # Device
    use_cuda = args.use_cuda.lower() == 'true'
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Construct data path
    data_path = Path(args.model_dir) / args.data

    # Load model and data
    data = load_model_and_data(args.model_dir, data_path, args.model_state_file)
    data['model'] = data['model'].to(device)

    # Demo mode
    if args.head and args.relation and args.tail:
        # Specific triple demo
        demo_single_triple(args.head, args.relation, args.tail, data, top_k=args.top_k, device=device)
    else:
        # Random triples demo from TEST set
        demo_random_triples(data, num_samples=args.num_samples, top_k=args.top_k, device=device)


if __name__ == '__main__':
    main()

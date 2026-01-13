"""
Link Prediction Extraction and Analysis.
Extracts top-k predictions from trained model and analyzes prediction quality.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import pickle
import json
from pathlib import Path
from tqdm import tqdm


def load_model_and_data(model_dir, data_dir):
    """Load trained model and data."""
    print(f"Loading model and data...")

    data = {}

    # PKL files are in parent directory (suppkg/), not in visualization_outputs/
    # viz_dir = fuselinker-complex/suppkg/visualization_outputs
    # pkl_dir = fuselinker-complex/suppkg
    from pathlib import Path
    pkl_dir = str(Path(data_dir).parent)

    # Load mappings from suppkg/ directory
    with open(f'{pkl_dir}/entity2index.pkl', 'rb') as f:
        data['entity2index'] = pickle.load(f)
    with open(f'{pkl_dir}/index2entity.pkl', 'rb') as f:
        data['index2entity'] = pickle.load(f)
    with open(f'{pkl_dir}/relation2index.pkl', 'rb') as f:
        data['relation2index'] = pickle.load(f)
    with open(f'{pkl_dir}/index2relation.pkl', 'rb') as f:
        data['index2relation'] = pickle.load(f)

    # Load embeddings from visualization_outputs/ directory
    data['entity_embeddings'] = np.load(f'{data_dir}/node_embeddings.npy')
    data['relation_embeddings'] = np.load(f'{data_dir}/relation_embeddings.npy')

    # Load graph structure from visualization_outputs/ directory
    with open(f'{data_dir}/train_graph.json', 'r') as f:
        train_graph = json.load(f)
        data['train_triples'] = np.array(train_graph['triples'])

    with open(f'{data_dir}/test_graph.json', 'r') as f:
        test_graph = json.load(f)
        data['test_triples'] = np.array(test_graph['triples'])

    print(f"  ✓ Loaded {len(data['entity2index'])} entities, {len(data['relation2index'])} relations")
    print(f"  ✓ Train: {len(data['train_triples'])} triples, Test: {len(data['test_triples'])} triples")

    return data


def calculate_scores_distmult(subject_emb, relation_emb, object_emb):
    """Calculate scores using DistMult formula: sum(h * r * t)"""
    # subject_emb: [batch, dim]
    # relation_emb: [batch, dim]
    # object_emb: [num_entities, dim]

    # Broadcast: [batch, dim] * [batch, dim] -> [batch, dim]
    sr = subject_emb * relation_emb

    # Matrix multiply: [batch, dim] @ [dim, num_entities] -> [batch, num_entities]
    scores = sr @ object_emb.T

    return scores


def calculate_scores_transe(subject_emb, relation_emb, object_emb, norm=2):
    """Calculate scores using TransE formula: -||h + r - t||"""
    import torch.nn.functional as F

    # Convert to torch if numpy
    if isinstance(subject_emb, np.ndarray):
        subject_emb = torch.from_numpy(subject_emb).float()
        relation_emb = torch.from_numpy(relation_emb).float()
        object_emb = torch.from_numpy(object_emb).float()

    # Normalize
    subject_emb = F.normalize(subject_emb, p=2, dim=1)
    object_emb = F.normalize(object_emb, p=2, dim=1)

    # [batch, 1, dim] + [batch, 1, dim] - [1, num_entities, dim]
    # -> [batch, num_entities, dim]
    diff = (subject_emb.unsqueeze(1) + relation_emb.unsqueeze(1) - object_emb.unsqueeze(0))

    # Compute norm
    scores = -torch.norm(diff, p=norm, dim=2)

    return scores.numpy()


def calculate_scores_complex(subject_emb, relation_emb, object_emb,
                             relation_emb_imag=None, entity_emb_imag=None):
    """Calculate scores using ComplEx formula."""
    # For now, use DistMult as approximation (real part only)
    # Full ComplEx requires imaginary embeddings which we may not have exported
    return calculate_scores_distmult(subject_emb, relation_emb, object_emb)


def predict_top_k_for_query(query_idx, data, top_k=10, model_type='distmult'):
    """
    Predict top-k objects for a given (subject, relation, ?) query.

    Args:
        query_idx: Index of the query triple in test set
        data: Dictionary with embeddings and mappings
        top_k: Number of top predictions to return
        model_type: 'distmult', 'transe', 'complex', or 'conve'

    Returns:
        predictions: List of (entity_idx, entity_name, score) tuples
    """
    query_triple = data['test_triples'][query_idx]
    subject_idx = int(query_triple[0])
    relation_idx = int(query_triple[1])
    true_object_idx = int(query_triple[2])

    # Get embeddings
    subject_emb = data['entity_embeddings'][subject_idx:subject_idx+1]  # [1, dim]
    relation_emb = data['relation_embeddings'][relation_idx:relation_idx+1]  # [1, dim]
    all_object_emb = data['entity_embeddings']  # [num_entities, dim]

    # Calculate scores based on model type
    if model_type.lower() in ['distmult', 'rgcn']:
        scores = calculate_scores_distmult(subject_emb, relation_emb, all_object_emb)
    elif model_type.lower() == 'transe':
        scores = calculate_scores_transe(subject_emb, relation_emb, all_object_emb)
    elif model_type.lower() == 'complex':
        scores = calculate_scores_complex(subject_emb, relation_emb, all_object_emb)
    else:
        # Default to DistMult
        scores = calculate_scores_distmult(subject_emb, relation_emb, all_object_emb)

    scores = scores.flatten()  # [num_entities]

    # Get top-k indices
    top_k_indices = np.argsort(scores)[::-1][:top_k]

    # Build predictions
    predictions = []
    for rank, entity_idx in enumerate(top_k_indices):
        entity_name = data['index2entity'].get(entity_idx, f'entity_{entity_idx}')
        score = float(scores[entity_idx])
        is_correct = (entity_idx == true_object_idx)

        predictions.append({
            'rank': rank + 1,
            'entity_idx': int(entity_idx),
            'entity_name': str(entity_name),
            'score': score,
            'is_correct': is_correct
        })

    # Query info
    subject_name = data['index2entity'].get(subject_idx, f'entity_{subject_idx}')
    relation_name = data['index2relation'].get(relation_idx, f'relation_{relation_idx}')
    true_object_name = data['index2entity'].get(true_object_idx, f'entity_{true_object_idx}')

    query_info = {
        'query_idx': query_idx,
        'subject': {'idx': subject_idx, 'name': str(subject_name)},
        'relation': {'idx': relation_idx, 'name': str(relation_name)},
        'true_object': {'idx': true_object_idx, 'name': str(true_object_name)},
        'true_object_rank': None,
        'true_object_score': float(scores[true_object_idx])
    }

    # Find rank of true object
    true_rank = np.where(np.argsort(scores)[::-1] == true_object_idx)[0]
    if len(true_rank) > 0:
        query_info['true_object_rank'] = int(true_rank[0]) + 1

    return query_info, predictions


def extract_all_predictions(data, num_queries=100, top_k=10, model_type='distmult', random_seed=42):
    """
    Extract predictions for multiple test queries.

    Args:
        data: Data dictionary
        num_queries: Number of test queries to analyze
        top_k: Number of top predictions per query
        model_type: Model type for scoring
        random_seed: Random seed for sampling

    Returns:
        all_results: List of (query_info, predictions) tuples
    """
    print(f"\nExtracting predictions for {num_queries} test queries...")
    print(f"Model type: {model_type}, Top-K: {top_k}")

    np.random.seed(random_seed)

    # Sample queries
    num_test = len(data['test_triples'])
    if num_queries > num_test:
        query_indices = range(num_test)
    else:
        query_indices = np.random.choice(num_test, num_queries, replace=False)

    all_results = []

    for query_idx in tqdm(query_indices, desc="Predicting"):
        query_info, predictions = predict_top_k_for_query(
            query_idx, data, top_k=top_k, model_type=model_type
        )

        all_results.append({
            'query': query_info,
            'predictions': predictions
        })

    print(f"  ✓ Extracted {len(all_results)} query results")

    return all_results


def analyze_predictions(all_results):
    """Analyze prediction quality."""
    print("\nAnalyzing predictions...")

    stats = {
        'total_queries': len(all_results),
        'hits_at_1': 0,
        'hits_at_3': 0,
        'hits_at_10': 0,
        'average_true_rank': 0,
        'mrr': 0,
    }

    ranks = []

    for result in all_results:
        true_rank = result['query']['true_object_rank']

        if true_rank is not None:
            ranks.append(true_rank)

            if true_rank == 1:
                stats['hits_at_1'] += 1
            if true_rank <= 3:
                stats['hits_at_3'] += 1
            if true_rank <= 10:
                stats['hits_at_10'] += 1

    if ranks:
        stats['average_true_rank'] = np.mean(ranks)
        stats['mrr'] = np.mean([1.0 / r for r in ranks])

    # Convert to percentages
    stats['hits_at_1'] /= stats['total_queries']
    stats['hits_at_3'] /= stats['total_queries']
    stats['hits_at_10'] /= stats['total_queries']

    return stats


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_predictions(all_results, stats, output_file):
    """Save predictions to JSON file."""
    # Convert numpy types to native Python types
    output_data = {
        'statistics': convert_to_serializable(stats),
        'predictions': convert_to_serializable(all_results)
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved predictions to: {output_file}")


def print_sample_predictions(all_results, num_samples=5):
    """Print sample predictions."""
    print(f"\n{'='*80}")
    print(f"SAMPLE PREDICTIONS (showing {num_samples} examples)")
    print(f"{'='*80}\n")

    for i, result in enumerate(all_results[:num_samples]):
        query = result['query']
        preds = result['predictions']

        print(f"Query {i+1}:")
        print(f"  ({query['subject']['name']}, {query['relation']['name']}, ?)")
        print(f"\n  True Answer: {query['true_object']['name']}")
        print(f"  True Rank: {query['true_object_rank']}")
        print(f"  True Score: {query['true_object_score']:.4f}")
        print(f"\n  Top-10 Predictions:")

        for pred in preds:
            marker = "✓ CORRECT" if pred['is_correct'] else ""
            print(f"    {pred['rank']:2d}. {pred['entity_name'][:50]:50s}  Score: {pred['score']:8.4f}  {marker}")

        print()


def main():
    parser = argparse.ArgumentParser(description='Extract link predictions from trained model')
    parser.add_argument(
        '--viz_dir',
        default='fuselinker-complex/suppkg/visualization_outputs',
        help='Directory containing exported visualization data (e.g., fuselinker-complex/suppkg/visualization_outputs)'
    )
    parser.add_argument(
        '--output_file',
        default='fuselinker-complex/suppkg/link_predictions.json',
        help='Output file for predictions'
    )
    parser.add_argument(
        '--num_queries',
        type=int,
        default=100,
        help='Number of test queries to analyze'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top predictions to extract'
    )
    parser.add_argument(
        '--model_type',
        default='distmult',
        choices=['distmult', 'transe', 'complex', 'conve'],
        help='Model type for scoring function'
    )
    parser.add_argument(
        '--show_samples',
        type=int,
        default=5,
        help='Number of sample predictions to display'
    )

    args = parser.parse_args()

    print("="*80)
    print("LINK PREDICTION EXTRACTION")
    print("="*80)

    # Load data
    data = load_model_and_data(args.viz_dir, args.viz_dir)

    # Extract predictions
    all_results = extract_all_predictions(
        data,
        num_queries=args.num_queries,
        top_k=args.top_k,
        model_type=args.model_type
    )

    # Analyze
    stats = analyze_predictions(all_results)

    # Print statistics
    print(f"\n{'='*80}")
    print("PREDICTION STATISTICS")
    print(f"{'='*80}")
    print(f"Total queries analyzed: {stats['total_queries']}")
    print(f"Hits@1:  {stats['hits_at_1']:.4f}")
    print(f"Hits@3:  {stats['hits_at_3']:.4f}")
    print(f"Hits@10: {stats['hits_at_10']:.4f}")
    print(f"MRR:     {stats['mrr']:.4f}")
    print(f"Average true rank: {stats['average_true_rank']:.2f}")

    # Show samples
    if args.show_samples > 0:
        print_sample_predictions(all_results, args.show_samples)

    # Save
    save_predictions(all_results, stats, args.output_file)

    print(f"\n{'='*80}")
    print("✓ EXTRACTION COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

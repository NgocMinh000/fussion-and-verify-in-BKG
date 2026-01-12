"""
Explainability Module for Link Predictions.
Explains WHY the model makes certain predictions by analyzing:
- Embedding similarities
- Nearest neighbors
- Score decomposition
- Training data patterns
"""

import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def load_prediction_data(prediction_file, viz_dir):
    """Load predictions and visualization data."""
    print(f"Loading prediction data...")

    # Load predictions
    with open(prediction_file, 'r') as f:
        pred_data = json.load(f)

    # Load visualization data
    data = {}
    data['entity_embeddings'] = np.load(f'{viz_dir}/entity_embeddings.npy')
    data['relation_embeddings'] = np.load(f'{viz_dir}/relation_embeddings.npy')

    with open(f'{viz_dir}/index2entity.pkl', 'rb') as f:
        data['index2entity'] = pickle.load(f)
    with open(f'{viz_dir}/index2relation.pkl', 'rb') as f:
        data['index2relation'] = pickle.load(f)

    # Load graph for pattern analysis
    with open(f'{viz_dir}/graph_structure.json', 'r') as f:
        graph_data = json.load(f)
        data['train_triples'] = np.array(graph_data['train_triples'])

    print(f"  ✓ Loaded predictions and embeddings")

    return pred_data, data


def find_nearest_neighbors(entity_idx, entity_embeddings, index2entity, top_k=5):
    """Find nearest neighbors in embedding space."""
    target_emb = entity_embeddings[entity_idx:entity_idx+1]

    # Compute cosine similarities
    similarities = cosine_similarity(target_emb, entity_embeddings)[0]

    # Get top-k (excluding self)
    similarities[entity_idx] = -np.inf  # Exclude self
    top_indices = np.argsort(similarities)[::-1][:top_k]

    neighbors = []
    for idx in top_indices:
        neighbors.append({
            'entity_idx': int(idx),
            'entity_name': str(index2entity.get(idx, f'entity_{idx}')),
            'similarity': float(similarities[idx])
        })

    return neighbors


def analyze_embedding_similarity(query, predictions, data):
    """
    Analyze why predictions have high scores based on embedding similarity.

    For DistMult: score = sum(h * r * t)
    High score means:
    - h and t are similar in directions emphasized by r
    - OR h, r, t are all aligned
    """
    subject_idx = query['subject']['idx']
    relation_idx = query['relation']['idx']

    subject_emb = data['entity_embeddings'][subject_idx]
    relation_emb = data['relation_embeddings'][relation_idx]

    explanations = []

    for pred in predictions[:5]:  # Top 5
        object_idx = pred['entity_idx']
        object_emb = data['entity_embeddings'][object_idx]

        # Cosine similarities
        h_t_sim = float(cosine_similarity([subject_emb], [object_emb])[0, 0])
        h_r_sim = float(cosine_similarity([subject_emb], [relation_emb])[0, 0])
        r_t_sim = float(cosine_similarity([relation_emb], [object_emb])[0, 0])

        # Element-wise product (DistMult scoring)
        h_r = subject_emb * relation_emb
        hr_t_sim = float(cosine_similarity([h_r], [object_emb])[0, 0])

        explanation = {
            'rank': pred['rank'],
            'entity_name': pred['entity_name'],
            'score': pred['score'],
            'similarities': {
                'subject_object': h_t_sim,
                'subject_relation': h_r_sim,
                'relation_object': r_t_sim,
                'combined_alignment': hr_t_sim
            },
            'reasoning': []
        }

        # Generate reasoning
        if hr_t_sim > 0.7:
            explanation['reasoning'].append(
                f"Strong alignment between (subject ⊙ relation) and object (similarity: {hr_t_sim:.3f})"
            )
        elif hr_t_sim > 0.5:
            explanation['reasoning'].append(
                f"Moderate alignment between (subject ⊙ relation) and object (similarity: {hr_t_sim:.3f})"
            )

        if h_t_sim > 0.7:
            explanation['reasoning'].append(
                f"Subject and object are very similar entities (similarity: {h_t_sim:.3f})"
            )

        if r_t_sim > 0.7:
            explanation['reasoning'].append(
                f"Object strongly associated with this relation (similarity: {r_t_sim:.3f})"
            )

        if not explanation['reasoning']:
            explanation['reasoning'].append(
                f"Moderate embedding alignment (combined: {hr_t_sim:.3f})"
            )

        explanations.append(explanation)

    return explanations


def analyze_training_patterns(query, predictions, data):
    """
    Analyze training data patterns to explain predictions.

    Check:
    - How many times subject appears with this relation
    - How many times predicted objects appear with this relation
    - Common patterns
    """
    subject_idx = query['subject']['idx']
    relation_idx = query['relation']['idx']

    train_triples = data['train_triples']

    # Find triples with same (subject, relation)
    same_sr = train_triples[
        (train_triples[:, 0] == subject_idx) & (train_triples[:, 1] == relation_idx)
    ]

    # Find triples with same relation (any subject)
    same_r = train_triples[train_triples[:, 1] == relation_idx]

    # Count object frequencies for this relation
    object_counts = defaultdict(int)
    for triple in same_r:
        object_counts[int(triple[2])] += 1

    # Analyze predictions
    pattern_explanations = []

    for pred in predictions[:5]:
        object_idx = pred['entity_idx']

        explanation = {
            'rank': pred['rank'],
            'entity_name': pred['entity_name'],
            'patterns': []
        }

        # Check if (subject, relation, object) exists in training
        exists_in_training = np.any(
            (train_triples[:, 0] == subject_idx) &
            (train_triples[:, 1] == relation_idx) &
            (train_triples[:, 2] == object_idx)
        )

        if exists_in_training:
            explanation['patterns'].append("✓ This exact triple was in training data")
        else:
            # Check frequency of this object with this relation
            freq = object_counts.get(object_idx, 0)
            total_r = len(same_r)

            if freq > 0:
                percentage = (freq / total_r) * 100 if total_r > 0 else 0
                explanation['patterns'].append(
                    f"This object appears with this relation {freq} times in training ({percentage:.1f}% of all triples with this relation)"
                )
            else:
                explanation['patterns'].append(
                    "This object never appeared with this relation in training (pure generalization)"
                )

        # Check if subject has this relation in training
        subject_r_count = len(same_sr)
        if subject_r_count > 0:
            explanation['patterns'].append(
                f"Subject has {subject_r_count} training examples with this relation"
            )
        else:
            explanation['patterns'].append(
                "Subject never seen with this relation in training (pure generalization)"
            )

        pattern_explanations.append(explanation)

    # Overall statistics
    stats = {
        'subject_relation_count': int(len(same_sr)),
        'relation_total_count': int(len(same_r)),
        'most_common_objects': []
    }

    # Top objects for this relation
    if object_counts:
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for obj_idx, count in top_objects:
            obj_name = data['index2entity'].get(obj_idx, f'entity_{obj_idx}')
            stats['most_common_objects'].append({
                'entity_name': str(obj_name),
                'count': count
            })

    return pattern_explanations, stats


def find_similar_queries(query, data, top_k=5):
    """Find similar queries (subject-relation pairs) in training data."""
    subject_idx = query['subject']['idx']
    relation_idx = query['relation']['idx']

    subject_emb = data['entity_embeddings'][subject_idx]
    relation_emb = data['relation_embeddings'][relation_idx]

    # Combined subject-relation embedding
    sr_emb = subject_emb * relation_emb

    # Find similar subject-relation pairs in training
    train_triples = data['train_triples']

    similarities = []

    for triple in train_triples[:1000]:  # Sample for speed
        train_subject_idx = int(triple[0])
        train_relation_idx = int(triple[1])
        train_object_idx = int(triple[2])

        train_subject_emb = data['entity_embeddings'][train_subject_idx]
        train_relation_emb = data['relation_embeddings'][train_relation_idx]
        train_sr_emb = train_subject_emb * train_relation_emb

        sim = float(cosine_similarity([sr_emb], [train_sr_emb])[0, 0])

        similarities.append({
            'similarity': sim,
            'subject': str(data['index2entity'].get(train_subject_idx, f'entity_{train_subject_idx}')),
            'relation': str(data['index2relation'].get(train_relation_idx, f'rel_{train_relation_idx}')),
            'object': str(data['index2entity'].get(train_object_idx, f'entity_{train_object_idx}'))
        })

    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)

    return similarities[:top_k]


def explain_single_query(query_result, data):
    """Generate comprehensive explanation for a single query."""
    query = query_result['query']
    predictions = query_result['predictions']

    print(f"\n{'='*100}")
    print(f"EXPLANATION FOR QUERY")
    print(f"{'='*100}")
    print(f"\nQuery: ({query['subject']['name']}, {query['relation']['name']}, ?)")
    print(f"True Answer: {query['true_object']['name']} (Rank: {query['true_object_rank']})\n")

    # 1. Embedding similarity analysis
    print(f"{'─'*100}")
    print("1. EMBEDDING SIMILARITY ANALYSIS")
    print(f"{'─'*100}")
    print("Why does the model predict these entities?\n")

    emb_explanations = analyze_embedding_similarity(query, predictions, data)

    for expl in emb_explanations:
        print(f"Rank {expl['rank']}: {expl['entity_name']} (score: {expl['score']:.4f})")
        print(f"  Similarities:")
        print(f"    - Subject ↔ Object: {expl['similarities']['subject_object']:.3f}")
        print(f"    - Subject ↔ Relation: {expl['similarities']['subject_relation']:.3f}")
        print(f"    - Relation ↔ Object: {expl['similarities']['relation_object']:.3f}")
        print(f"    - Combined alignment: {expl['similarities']['combined_alignment']:.3f}")
        print(f"  Reasoning:")
        for reason in expl['reasoning']:
            print(f"    • {reason}")
        print()

    # 2. Training pattern analysis
    print(f"{'─'*100}")
    print("2. TRAINING DATA PATTERNS")
    print(f"{'─'*100}")
    print("What did the model learn from training data?\n")

    pattern_explanations, stats = analyze_training_patterns(query, predictions, data)

    print(f"Training Statistics:")
    print(f"  - Subject with this relation: {stats['subject_relation_count']} times")
    print(f"  - This relation (all subjects): {stats['relation_total_count']} times")
    print(f"\nMost common objects for this relation in training:")
    for obj in stats['most_common_objects']:
        print(f"  • {obj['entity_name']}: {obj['count']} times")

    print(f"\nPrediction patterns:")
    for expl in pattern_explanations:
        print(f"\nRank {expl['rank']}: {expl['entity_name']}")
        for pattern in expl['patterns']:
            print(f"  • {pattern}")

    # 3. Nearest neighbors
    print(f"\n{'─'*100}")
    print("3. NEAREST NEIGHBORS IN EMBEDDING SPACE")
    print(f"{'─'*100}")

    subject_idx = query['subject']['idx']
    print(f"\nNearest neighbors to subject '{query['subject']['name']}':")
    neighbors = find_nearest_neighbors(subject_idx, data['entity_embeddings'], data['index2entity'], top_k=5)
    for i, neighbor in enumerate(neighbors, 1):
        print(f"  {i}. {neighbor['entity_name']} (similarity: {neighbor['similarity']:.3f})")

    # 4. Similar training queries
    print(f"\n{'─'*100}")
    print("4. SIMILAR QUERIES IN TRAINING DATA")
    print(f"{'─'*100}")
    print("Model may generalize from these similar patterns:\n")

    similar = find_similar_queries(query, data, top_k=5)
    for i, sim_query in enumerate(similar, 1):
        print(f"  {i}. ({sim_query['subject']}, {sim_query['relation']}, {sim_query['object']})")
        print(f"     Similarity: {sim_query['similarity']:.3f}")

    print(f"\n{'='*100}\n")

    return {
        'query': query,
        'embedding_analysis': emb_explanations,
        'training_patterns': pattern_explanations,
        'training_stats': stats,
        'nearest_neighbors': neighbors,
        'similar_queries': similar
    }


def explain_multiple_queries(pred_data, data, num_queries=5):
    """Generate explanations for multiple queries."""
    print("="*100)
    print(f"GENERATING EXPLANATIONS FOR {num_queries} QUERIES")
    print("="*100)

    all_explanations = []

    predictions = pred_data['predictions'][:num_queries]

    for i, query_result in enumerate(predictions):
        print(f"\n\n{'#'*100}")
        print(f"QUERY {i+1}/{num_queries}")
        print(f"{'#'*100}")

        explanation = explain_single_query(query_result, data)
        all_explanations.append(explanation)

    return all_explanations


def save_explanations(explanations, output_file):
    """Save explanations to JSON."""
    with open(output_file, 'w') as f:
        json.dump(explanations, f, indent=2)

    print(f"\n✓ Saved explanations to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Explain link predictions')
    parser.add_argument(
        '--prediction_file',
        default='fuselinker-complex/suppkg/link_predictions.json',
        help='Prediction file from link_predictor.py (e.g., fuselinker-complex/suppkg/link_predictions.json)'
    )
    parser.add_argument(
        '--viz_dir',
        default='fuselinker-complex/suppkg/visualization_outputs',
        help='Visualization data directory'
    )
    parser.add_argument(
        '--output_file',
        default='fuselinker-complex/suppkg/prediction_explanations.json',
        help='Output file for explanations'
    )
    parser.add_argument(
        '--num_queries',
        type=int,
        default=5,
        help='Number of queries to explain'
    )

    args = parser.parse_args()

    # Load data
    pred_data, data = load_prediction_data(args.prediction_file, args.viz_dir)

    # Generate explanations
    explanations = explain_multiple_queries(pred_data, data, args.num_queries)

    # Save
    save_explanations(explanations, args.output_file)

    print("\n" + "="*100)
    print("✓ EXPLANATION GENERATION COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()

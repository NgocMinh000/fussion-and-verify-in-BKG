#!/usr/bin/env python
"""
Script để predict và visualize các liên kết mới (fused links) từ trained FuseLinker model

Usage:
    python predict_new_links.py --model suppkg_model_state.pth --data suppkg

Hoặc với custom parameters:
    python predict_new_links.py --model suppkg_model_state.pth --data suppkg \\
        --top_k 200 --min_score 0.8 --output predicted_links.csv
"""

import argparse
import sys
from pathlib import Path

# Add visualization to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.link_predictor import LinkPredictor, quick_predict


def main():
    parser = argparse.ArgumentParser(
        description="Predict new links from trained FuseLinker model"
    )

    # Required arguments
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model state file (.pth)'
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Data directory (containing train.tsv, entity2index.pkl, etc.)'
    )

    # Optional arguments
    parser.add_argument(
        '--text_embedding',
        default=None,
        help='Path to text embeddings (default: <data>/pubmedbert_pretrained_embeddings_768.npy)'
    )
    parser.add_argument(
        '--knowledge_embedding',
        default=None,
        help='Path to domain knowledge embeddings (default: <data>/poincare_embeddings.npy)'
    )
    parser.add_argument(
        '--output',
        default='predicted_links.csv',
        help='Output CSV file for predictions (default: predicted_links.csv)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=100,
        help='Top K predictions per relation (default: 100)'
    )
    parser.add_argument(
        '--min_score',
        type=float,
        default=0.7,
        help='Minimum confidence score (default: 0.7)'
    )

    # Model parameters (should match training)
    parser.add_argument('--n_hidden', type=int, default=200)
    parser.add_argument('--num_bases', type=int, default=20)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--reg_param', type=float, default=0.01)
    parser.add_argument('--w', type=float, default=0.75)

    # Filtering options
    parser.add_argument(
        '--filter_semantic_types',
        nargs='+',
        help='Filter predictions to specific semantic types (e.g., dsyn phsu)'
    )
    parser.add_argument(
        '--filter_relations',
        nargs='+',
        help='Filter predictions to specific relations (e.g., TREATS PREVENTS)'
    )

    args = parser.parse_args()

    # Set default embedding paths
    data_dir = Path(args.data)
    if args.text_embedding is None:
        args.text_embedding = str(data_dir / 'pubmedbert_pretrained_embeddings_768.npy')
    if args.knowledge_embedding is None:
        args.knowledge_embedding = str(data_dir / 'poincare_embeddings.npy')

    # Use quick_predict function
    print("\n" + "="*80)
    print("FUSELINKER LINK PREDICTION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Top K per relation: {args.top_k}")
    print(f"Min score: {args.min_score}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    predictions = quick_predict(
        model_state_path=args.model,
        data_dir=args.data,
        text_embedding_path=args.text_embedding,
        knowledge_embedding_path=args.knowledge_embedding,
        output_path=args.output,
        top_k_per_relation=args.top_k,
        min_score=args.min_score,
        n_hidden=args.n_hidden,
        num_bases=args.num_bases,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        reg_param=args.reg_param,
        w=args.w
    )

    # Apply additional filters if specified
    if args.filter_semantic_types or args.filter_relations:
        print("\nApplying filters...")
        predictor = LinkPredictor()
        predictor.predictions = predictions

        filtered = predictor.filter_predictions(
            semantic_types=args.filter_semantic_types,
            relations=args.filter_relations
        )

        print(f"Filtered from {len(predictions)} to {len(filtered)} predictions")

        # Save filtered results
        filtered_output = args.output.replace('.csv', '_filtered.csv')
        filtered.to_csv(filtered_output, index=False)
        print(f"✓ Saved filtered predictions to {filtered_output}")

    print("\n" + "="*80)
    print("DONE! Check the output file for all predictions.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

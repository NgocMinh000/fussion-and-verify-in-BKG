#!/usr/bin/env python3
"""
Convert UMLS triples to FuseLinker format

Converts a CSV file with header (head,relation,tail) to FuseLinker format:
- train.tsv (80%)
- valid.tsv (10%)
- test.tsv (10%)

All TSV files are tab-separated with NO header.

Usage:
    python convert_to_fuselinker_format.py --input umls_triples_multi_v2.txt --output mybkg
    python convert_to_fuselinker_format.py --input data.csv --output output_dir --split 0.7 0.15 0.15
"""

import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def load_triples(input_file):
    """
    Load triples from CSV/TXT file.

    Supports:
    - CSV with header: head,relation,tail
    - TSV with header
    - Auto-detects delimiter
    """
    print(f"Loading triples from: {input_file}")

    # Try to detect delimiter
    with open(input_file, 'r') as f:
        first_line = f.readline()
        if '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','

    delimiter_name = 'TAB' if delimiter == '\t' else 'COMMA'
    print(f"Detected delimiter: {delimiter_name}")

    # Load data
    df = pd.read_csv(input_file, sep=delimiter)

    # Check columns
    expected_cols = ['head', 'relation', 'tail']
    if list(df.columns) != expected_cols:
        print(f"Warning: Expected columns {expected_cols}, got {list(df.columns)}")
        # Try to rename if it has 3 columns
        if len(df.columns) == 3:
            df.columns = expected_cols
            print(f"Renamed columns to: {expected_cols}")

    print(f"✓ Loaded {len(df)} triples")
    print(f"✓ Unique entities: {df['head'].nunique() + df['tail'].nunique()}")
    print(f"✓ Unique relations: {df['relation'].nunique()}")

    return df


def print_statistics(df, name="Dataset"):
    """Print dataset statistics."""
    num_triples = len(df)
    num_heads = df['head'].nunique()
    num_tails = df['tail'].nunique()
    num_entities = len(set(df['head'].unique()).union(set(df['tail'].unique())))
    num_relations = df['relation'].nunique()

    print(f"\n{name} Statistics:")
    print(f"  Triples: {num_triples}")
    print(f"  Entities: {num_entities} (heads: {num_heads}, tails: {num_tails})")
    print(f"  Relations: {num_relations}")

    # Show relation distribution
    print(f"\n  Relation distribution:")
    for rel, count in df['relation'].value_counts().items():
        percentage = (count / num_triples) * 100
        print(f"    {rel}: {count} ({percentage:.1f}%)")


def split_data(df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split data into train/valid/test sets.

    Args:
        df: DataFrame with columns [head, relation, tail]
        train_ratio: Ratio for training set (default: 0.8)
        valid_ratio: Ratio for validation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        random_seed: Random seed for reproducibility

    Returns:
        train_df, valid_df, test_df
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    print(f"\nSplitting data: {train_ratio:.0%} train, {valid_ratio:.0%} valid, {test_ratio:.0%} test")

    # Shuffle data
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Calculate split indices
    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    # Split
    train_df = df_shuffled[:train_end]
    valid_df = df_shuffled[train_end:valid_end]
    test_df = df_shuffled[valid_end:]

    print(f"✓ Train: {len(train_df)} triples")
    print(f"✓ Valid: {len(valid_df)} triples")
    print(f"✓ Test: {len(test_df)} triples")

    return train_df, valid_df, test_df


def create_entity_mappings(df):
    """
    Create entity2index and index2entity mappings from DataFrame.

    Args:
        df: DataFrame with columns [head, relation, tail]

    Returns:
        entity2index: dict mapping entity -> index
        index2entity: dict mapping index -> entity
    """
    # Get all unique entities (from both head and tail)
    all_entities = set()
    all_entities.update(df['head'].unique())
    all_entities.update(df['tail'].unique())

    # Sort for consistent ordering
    sorted_entities = sorted(list(all_entities))

    # Create mappings
    entity2index = {entity: idx for idx, entity in enumerate(sorted_entities)}
    index2entity = {idx: entity for idx, entity in enumerate(sorted_entities)}

    return entity2index, index2entity


def create_relation_mappings(df):
    """
    Create relation2index and index2relation mappings from DataFrame.

    Args:
        df: DataFrame with columns [head, relation, tail]

    Returns:
        relation2index: dict mapping ('head', relation, 'tail') -> index
        index2relation: dict mapping index -> ('head', relation, 'tail')
    """
    # Get all unique relations
    unique_relations = sorted(df['relation'].unique())

    # Create mappings in FuseLinker format: ('head', relation, 'tail')
    relation2index = {}
    index2relation = {}

    for idx, relation in enumerate(unique_relations):
        key = ('head', relation, 'tail')
        relation2index[key] = idx
        index2relation[idx] = key

    return relation2index, index2relation


def save_tsv(df, output_path, name=""):
    """Save DataFrame as TSV file (no header, tab-separated)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save without header, tab-separated
    df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"✓ Saved {name}: {output_path} ({len(df)} triples)")


def save_pickle(obj, output_path, name=""):
    """Save object as pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✓ Saved {name}: {output_path} ({len(obj)} items)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert triples to FuseLinker format (train.tsv, valid.tsv, test.tsv)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input file (CSV/TXT with header: head,relation,tail)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for train.tsv, valid.tsv, test.tsv')
    parser.add_argument('--train', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--valid', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--stats', action='store_true',
                       help='Print detailed statistics for each split')

    args = parser.parse_args()

    print("="*70)
    print("FuseLinker Format Converter")
    print("="*70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Split:  train={args.train}, valid={args.valid}, test={args.test}")
    print(f"Seed:   {args.seed}")
    print("="*70)

    # Load data
    df = load_triples(args.input)

    # Print initial statistics
    print_statistics(df, "Original Data")

    # Split data
    train_df, valid_df, test_df = split_data(
        df,
        train_ratio=args.train,
        valid_ratio=args.valid,
        test_ratio=args.test,
        random_seed=args.seed
    )

    # Print split statistics if requested
    if args.stats:
        print_statistics(train_df, "Train Set")
        print_statistics(valid_df, "Valid Set")
        print_statistics(test_df, "Test Set")

    # Create entity and relation mappings from ENTIRE dataset (not just train)
    print("\nCreating entity and relation mappings...")
    entity2index, index2entity = create_entity_mappings(df)
    relation2index, index2relation = create_relation_mappings(df)

    print(f"✓ Entity mappings: {len(entity2index)} unique entities")
    print(f"✓ Relation mappings: {len(relation2index)} unique relations")

    # Save to output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving TSV files...")
    save_tsv(train_df, output_dir / 'train.tsv', 'train.tsv')
    save_tsv(valid_df, output_dir / 'valid.tsv', 'valid.tsv')
    save_tsv(test_df, output_dir / 'test.tsv', 'test.tsv')

    print("\nSaving pickle files...")
    save_pickle(entity2index, output_dir / 'entity2index.pkl', 'entity2index.pkl')
    save_pickle(index2entity, output_dir / 'index2entity.pkl', 'index2entity.pkl')
    save_pickle(relation2index, output_dir / 'relation2index.pkl', 'relation2index.pkl')
    save_pickle(index2relation, output_dir / 'index2relation.pkl', 'index2relation.pkl')

    # Summary
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"✓ Created 7 files in: {output_dir}")
    print(f"\n  TSV files (triples):")
    print(f"    - train.tsv: {len(train_df)} triples")
    print(f"    - valid.tsv: {len(valid_df)} triples")
    print(f"    - test.tsv: {len(test_df)} triples")
    print(f"\n  Pickle files (mappings):")
    print(f"    - entity2index.pkl: {len(entity2index)} entities")
    print(f"    - index2entity.pkl: {len(index2entity)} entities")
    print(f"    - relation2index.pkl: {len(relation2index)} relations")
    print(f"    - index2relation.pkl: {len(index2relation)} relations")
    print(f"\nNext steps:")
    print(f"1. Use with FuseLinker:")
    print(f"   cd fuselinker-complex")
    print(f"   python main.py --data {output_dir} --text_embedding_file sapbert_embeddings ...")
    print("="*70)


if __name__ == '__main__':
    main()

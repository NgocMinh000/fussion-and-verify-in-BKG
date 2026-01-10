#!/usr/bin/env python3
"""
Generate SapBERT embeddings for knowledge graph entities

Usage:
    python generate_sapbert_embeddings.py --data suppkg --output sapbert_embeddings_768.npy
    python generate_sapbert_embeddings.py --data mybkg --batch_size 64
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os


def load_entities(data_dir):
    """
    Load entity names from knowledge graph TSV files

    Args:
        data_dir: Directory containing train.tsv, valid.tsv, test.tsv

    Returns:
        list of unique entity names (sorted)
    """
    print(f"Loading entities from {data_dir}...")

    # Load train, valid, test files
    train_path = os.path.join(data_dir, 'train.tsv')
    valid_path = os.path.join(data_dir, 'valid.tsv')
    test_path = os.path.join(data_dir, 'test.tsv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.tsv not found in {data_dir}")

    train = pd.read_csv(train_path, sep='\t', header=None, names=['head', 'relation', 'tail'])

    # Load valid and test if they exist
    valid = pd.read_csv(valid_path, sep='\t', header=None, names=['head', 'relation', 'tail']) if os.path.exists(valid_path) else pd.DataFrame()
    test = pd.read_csv(test_path, sep='\t', header=None, names=['head', 'relation', 'tail']) if os.path.exists(test_path) else pd.DataFrame()

    # Get unique entities
    all_entities = set()
    for df in [train, valid, test]:
        if len(df) > 0:
            all_entities.update(df['head'].unique())
            all_entities.update(df['tail'].unique())

    # Sort for consistent ordering
    entities = sorted(list(all_entities))
    print(f"✓ Found {len(entities)} unique entities")

    return entities


def generate_sapbert_embeddings(entity_names, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                                batch_size=32, max_length=25):
    """
    Generate SapBERT embeddings for entity names

    Args:
        entity_names: List of entity name strings
        model_name: HuggingFace model name
        batch_size: Batch size for encoding
        max_length: Maximum sequence length

    Returns:
        numpy array of embeddings [num_entities, 768]
    """
    # Load model
    print(f"\nLoading SapBERT model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying to download model for the first time...")
        print("This may take a few minutes...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"✓ Using device: {device}")
    if device.type == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    # Generate embeddings in batches
    all_embeddings = []
    num_batches = (len(entity_names) + batch_size - 1) // batch_size

    print(f"\nGenerating embeddings for {len(entity_names)} entities...")
    print(f"Batch size: {batch_size}, Max length: {max_length}")
    print(f"Total batches: {num_batches}")

    with torch.no_grad():
        for i in tqdm(range(0, len(entity_names), batch_size), desc="Progress", unit="batch"):
            batch_texts = entity_names[i:i+batch_size]

            # Tokenize
            encoded = tokenizer.batch_encode_plus(
                batch_texts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )

            # Move to device
            encoded = {key: val.to(device) for key, val in encoded.items()}

            # Get [CLS] token embeddings
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

            all_embeddings.append(cls_embeddings.cpu().numpy())

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)

    print(f"\n✓ Generated embeddings shape: {embeddings.shape}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")

    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Generate SapBERT embeddings for knowledge graph entities')
    parser.add_argument('--data', type=str, required=True,
                       help='Data directory containing train.tsv, valid.tsv, test.tsv (e.g., suppkg)')
    parser.add_argument('--output', type=str, default='sapbert_embeddings_768.npy',
                       help='Output file name (default: sapbert_embeddings_768.npy)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for encoding (default: 32, use 64+ for GPU)')
    parser.add_argument('--max_length', type=int, default=25,
                       help='Maximum sequence length (default: 25)')
    parser.add_argument('--model', type=str, default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
                       help='SapBERT model name from HuggingFace')

    args = parser.parse_args()

    # Validate data directory
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data directory not found: {args.data}")

    print("=" * 70)
    print("SapBERT Embedding Generator")
    print("=" * 70)
    print(f"Data directory: {args.data}")
    print(f"Output file: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {args.model}")
    print("=" * 70)

    # Load entities
    entities = load_entities(args.data)

    # Generate embeddings
    embeddings = generate_sapbert_embeddings(
        entities,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Save to .npy file
    output_path = os.path.join(args.data, args.output)
    print(f"\nSaving embeddings to {output_path}...")
    np.save(output_path, embeddings)
    print(f"✓ Embeddings saved successfully")

    # Also save entity list for reference
    entity_list_path = os.path.join(args.data, 'entity_list_sapbert.txt')
    print(f"Saving entity list to {entity_list_path}...")
    with open(entity_list_path, 'w', encoding='utf-8') as f:
        for entity in entities:
            f.write(f"{entity}\n")
    print(f"✓ Entity list saved successfully")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Processed {len(entities)} entities")
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Output file: {output_path}")
    print(f"✓ Entity list: {entity_list_path}")
    print("\nNext steps:")
    print(f"1. Use these embeddings in training:")
    print(f"   python main.py --data {args.data} \\")
    print(f"       --text_embedding_file {args.output} \\")
    print(f"       --knowledge_embedding_file poincare_embeddings.npy \\")
    print(f"       --w 0.75")
    print("\n2. Compare with PubMedBERT to see improvement!")
    print("=" * 70)


if __name__ == '__main__':
    main()

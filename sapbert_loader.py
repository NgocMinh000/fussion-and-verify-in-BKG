"""
SapBERT Embedding Loader - On-the-fly generation during training

This module provides functionality to generate SapBERT embeddings dynamically
when training starts, without requiring pre-saved .npy files.

Usage:
    from sapbert_loader import load_sapbert_embeddings

    embeddings = load_sapbert_embeddings(
        index2entity=knowledge_graph.index2entity,
        batch_size=128,
        device='cuda'
    )
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def load_sapbert_embeddings(
    index2entity,
    model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    batch_size=128,
    max_length=25,
    device=None
):
    """
    Generate SapBERT embeddings for entities on-the-fly.

    Args:
        index2entity: Dictionary mapping entity indices to entity names
        model_name: HuggingFace model name for SapBERT
        batch_size: Batch size for encoding (default: 128)
        max_length: Maximum sequence length (default: 25)
        device: torch device ('cuda' or 'cpu'). If None, auto-detect.

    Returns:
        numpy array of embeddings [num_entities, 768]
    """
    print("\n" + "="*70)
    print("Loading SapBERT Embeddings (On-the-fly Generation)")
    print("="*70)

    # Auto-detect device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"Device: {device}")
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Extract entity names in order (sorted by index)
    num_entities = len(index2entity)
    entity_names = []
    for i in range(num_entities):
        entity_name = index2entity.get(i, f"entity_{i}")
        # Convert to string to handle any data type
        entity_names.append(str(entity_name))

    print(f"Number of entities: {num_entities}")

    # Load SapBERT model
    print(f"\nLoading SapBERT model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading SapBERT model: {e}")
        print("Please ensure transformers library is installed and you have internet connection")
        raise

    # Move model to device
    model = model.to(device)
    model.eval()

    # Generate embeddings in batches
    print(f"\nGenerating embeddings...")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_length}")

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, num_entities, batch_size), desc="Encoding entities"):
            batch_texts = entity_names[i:i+batch_size]

            # Tokenize batch
            encoded = tokenizer.batch_encode_plus(
                batch_texts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )

            # Move to device
            encoded_cuda = {key: val.to(device) for key, val in encoded.items()}

            # Get [CLS] token representation as embedding
            outputs = model(**encoded_cuda)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

            # Move to CPU and store
            all_embeddings.append(cls_embeddings.cpu().numpy())

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)

    print(f"\n✓ Successfully generated SapBERT embeddings")
    print(f"✓ Shape: {embeddings.shape}")
    print(f"✓ Dimension: {embeddings.shape[1]}")
    print("="*70 + "\n")

    return embeddings


def is_sapbert_keyword(embedding_file_path):
    """
    Check if the embedding file path is the SapBERT keyword.

    Args:
        embedding_file_path: Path or keyword string

    Returns:
        bool: True if it's the SapBERT keyword
    """
    # Check for various forms of the keyword
    keywords = ["sapbert_embeddings", "sapbert", "sapbert-embeddings"]

    # Get base name without path
    import os
    base_name = os.path.basename(embedding_file_path).lower()

    return base_name in keywords


def load_text_embeddings(embedding_path, index2entity=None, use_cuda=True):
    """
    Smart loader that handles both .npy files and SapBERT keyword.

    This function automatically detects:
    - If embedding_path is "sapbert_embeddings" -> Generate using SapBERT
    - Otherwise -> Load from .npy file using np.load()

    Args:
        embedding_path: Path to .npy file OR "sapbert_embeddings" keyword
        index2entity: Required if using SapBERT (mapping of indices to entity names)
        use_cuda: Whether to use GPU for SapBERT generation

    Returns:
        numpy array of embeddings or None if loading fails
    """
    import os

    # Check if it's SapBERT keyword
    if is_sapbert_keyword(embedding_path):
        print(f"Detected SapBERT keyword: '{embedding_path}'")

        if index2entity is None:
            raise ValueError(
                "index2entity mapping is required for SapBERT embeddings. "
                "Please pass index2entity parameter."
            )

        # Generate SapBERT embeddings on-the-fly
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        embeddings = load_sapbert_embeddings(
            index2entity=index2entity,
            batch_size=128,
            device=device
        )
        return embeddings

    else:
        # Standard .npy file loading
        print(f"Loading embeddings from file: {embedding_path}")

        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

        try:
            embeddings = np.load(embedding_path)
            print(f"✓ Loaded embeddings successfully! Shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"✗ Failed to load embeddings: {e}")
            raise


if __name__ == "__main__":
    # Test the module
    print("SapBERT Loader Module")
    print("="*70)
    print("\nTest 1: Check keyword detection")

    test_cases = [
        "sapbert_embeddings",
        "sapbert",
        "sapbert-embeddings",
        "/path/to/sapbert_embeddings",
        "pubmedbert_embeddings.npy",
        "poincare_embeddings.npy"
    ]

    for test in test_cases:
        result = is_sapbert_keyword(test)
        print(f"  '{test}' -> {result}")

    print("\nTest 2: Create dummy index2entity mapping")
    dummy_index2entity = {
        0: "Disease_Alzheimer",
        1: "Gene_APOE",
        2: "Protein_Amyloid_Beta",
        3: "Pathway_Apoptosis",
        4: "Drug_Memantine"
    }

    print(f"Created mapping with {len(dummy_index2entity)} entities")

    print("\nTo test full embedding generation, run:")
    print("  from sapbert_loader import load_sapbert_embeddings")
    print("  embeddings = load_sapbert_embeddings(index2entity)")

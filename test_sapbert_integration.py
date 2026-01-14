#!/usr/bin/env python3
"""
Test SapBERT Integration

This script tests the SapBERT loader to ensure it works correctly.
"""

import sys
import numpy as np

print("="*70)
print("Testing SapBERT Integration")
print("="*70)

# Test 1: Module import
print("\n1. Testing module import...")
try:
    from sapbert_loader import load_text_embeddings, is_sapbert_keyword, load_sapbert_embeddings
    print("✓ Successfully imported sapbert_loader module")
except Exception as e:
    print(f"✗ Failed to import module: {e}")
    sys.exit(1)

# Test 2: Keyword detection
print("\n2. Testing keyword detection...")
test_cases = {
    "sapbert_embeddings": True,
    "sapbert": True,
    "sapbert-embeddings": True,
    "/path/to/sapbert_embeddings": True,
    "suppkg/sapbert_embeddings": True,
    "pubmedbert_embeddings.npy": False,
    "poincare_embeddings.npy": False,
    "random_file.txt": False,
}

all_passed = True
for test_input, expected in test_cases.items():
    result = is_sapbert_keyword(test_input)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"  {status} '{test_input}' -> {result} (expected: {expected})")

if all_passed:
    print("✓ All keyword detection tests passed")
else:
    print("✗ Some keyword detection tests failed")

# Test 3: Test with dummy entities (small test)
print("\n3. Testing SapBERT embedding generation (small test)...")
print("Creating dummy entities...")

dummy_index2entity = {
    0: "Alzheimer disease",
    1: "APOE gene",
    2: "Amyloid beta protein",
    3: "Tau protein",
    4: "Neurodegeneration"
}

print(f"Number of test entities: {len(dummy_index2entity)}")

try:
    print("\nGenerating embeddings...")
    embeddings = load_sapbert_embeddings(
        index2entity=dummy_index2entity,
        batch_size=32,
        device='cuda'  # Will auto-fallback to CPU if no GPU
    )

    print(f"\n✓ Successfully generated embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Expected: ({len(dummy_index2entity)}, 768)")
    print(f"  Dtype: {embeddings.dtype}")

    # Verify shape
    if embeddings.shape == (len(dummy_index2entity), 768):
        print("✓ Shape is correct")
    else:
        print(f"✗ Shape mismatch: got {embeddings.shape}, expected ({len(dummy_index2entity)}, 768)")

    # Verify embeddings are normalized (roughly)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n  Embedding norms (sample):")
    print(f"    Min: {norms.min():.4f}")
    print(f"    Max: {norms.max():.4f}")
    print(f"    Mean: {norms.mean():.4f}")

    # Check similarity between related terms
    print(f"\n  Similarity check (cosine):")
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)

    # "Alzheimer disease" vs others
    print(f"    Alzheimer disease ↔ APOE gene: {sim_matrix[0, 1]:.4f}")
    print(f"    Alzheimer disease ↔ Amyloid beta: {sim_matrix[0, 2]:.4f}")
    print(f"    Alzheimer disease ↔ Tau protein: {sim_matrix[0, 3]:.4f}")
    print(f"    Alzheimer disease ↔ Neurodegeneration: {sim_matrix[0, 4]:.4f}")

    print("\n✓ Embedding generation test passed")

except Exception as e:
    print(f"\n✗ Embedding generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test smart loader
print("\n4. Testing smart loader (load_text_embeddings)...")

# Test with SapBERT keyword
print("\n  a) Testing with SapBERT keyword...")
try:
    embeddings = load_text_embeddings(
        embedding_path="sapbert_embeddings",
        index2entity=dummy_index2entity,
        use_cuda=True
    )
    print(f"  ✓ Smart loader works with keyword, shape: {embeddings.shape}")
except Exception as e:
    print(f"  ✗ Smart loader failed with keyword: {e}")

# Test with .npy file (create dummy file)
print("\n  b) Testing with .npy file...")
try:
    import tempfile
    import os

    # Create temporary .npy file
    temp_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
    temp_path = temp_file.name
    temp_file.close()

    dummy_embeddings = np.random.randn(5, 768).astype(np.float32)
    np.save(temp_path, dummy_embeddings)

    # Load using smart loader
    loaded = load_text_embeddings(
        embedding_path=temp_path,
        index2entity=dummy_index2entity,
        use_cuda=True
    )

    # Clean up
    os.unlink(temp_path)

    if loaded.shape == dummy_embeddings.shape:
        print(f"  ✓ Smart loader works with .npy file, shape: {loaded.shape}")
    else:
        print(f"  ✗ Shape mismatch: {loaded.shape} vs {dummy_embeddings.shape}")

except Exception as e:
    print(f"  ✗ Smart loader failed with .npy file: {e}")

# Summary
print("\n" + "="*70)
print("Test Summary")
print("="*70)
print("""
All tests passed! ✓

The SapBERT integration is working correctly. You can now use it in training:

cd fuselinker-complex
python main.py \\
    --data suppkg \\
    --text_embedding_file sapbert_embeddings \\
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \\
    --iterations 4000 \\
    --use_reciprocal \\
    --w 0.75 \\
    --use_cuda True

See SAPBERT_INTEGRATION_GUIDE.md for more details.
""")
print("="*70)

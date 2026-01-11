#!/usr/bin/env python3
"""Debug script to check embedding file shapes and compatibility"""

import numpy as np
import os

# Paths
medllama_path = "/root/fussion-and-verify-in-BKG/engine/medllama_pretrained_embeddings_4096.npy"
poincare_path = "/root/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy"

print("="*70)
print("EMBEDDING FILES DEBUG")
print("="*70)

# Check MedLLaMA
print("\n1. MedLLaMA Text Embeddings:")
print(f"   Path: {medllama_path}")
print(f"   Exists: {os.path.exists(medllama_path)}")
if os.path.exists(medllama_path):
    file_size = os.path.getsize(medllama_path) / (1024**2)  # MB
    print(f"   File size: {file_size:.2f} MB")

    try:
        # Load without allow_pickle to be safe
        data = np.load(medllama_path, allow_pickle=False)
        print(f"   ✓ Shape: {data.shape}")
        print(f"   ✓ Dtype: {data.dtype}")
        print(f"   ✓ Total elements: {data.size:,}")
        print(f"   ✓ Memory: {data.nbytes / (1024**2):.2f} MB")

        # Possible interpretations
        print(f"\n   Possible shapes:")
        total = data.size
        if total % 4096 == 0:
            n_entities = total // 4096
            print(f"   - ({n_entities:,}, 4096) - {n_entities} entities with 4096D embeddings")
        if total % 43474 == 0:
            dims = total // 43474
            print(f"   - (43474, {dims}) - 43474 entities with {dims}D embeddings")

    except Exception as e:
        print(f"   ✗ Error loading: {e}")

# Check Poincaré
print("\n2. Poincaré Domain Embeddings:")
print(f"   Path: {poincare_path}")
print(f"   Exists: {os.path.exists(poincare_path)}")
if os.path.exists(poincare_path):
    file_size = os.path.getsize(poincare_path) / (1024**2)  # MB
    print(f"   File size: {file_size:.2f} MB")

    try:
        data = np.load(poincare_path, allow_pickle=False)
        print(f"   ✓ Shape: {data.shape}")
        print(f"   ✓ Dtype: {data.dtype}")
        print(f"   ✓ Total elements: {data.size:,}")
        print(f"   ✓ Memory: {data.nbytes / (1024**2):.2f} MB")
    except Exception as e:
        print(f"   ✗ Error loading: {e}")

# Check dataset
print("\n3. Dataset Entities:")
print(f"   Expected entities: 43,474")
print(f"   Expected MedLLaMA shape: (43474, 4096)")
print(f"   Expected Poincaré shape: (43474, 50) or (43474, 200)")

# Check other embedding files
print("\n4. Other embedding files in engine/:")
engine_dir = "/root/fussion-and-verify-in-BKG/engine/"
if os.path.exists(engine_dir):
    for fname in sorted(os.listdir(engine_dir)):
        if fname.endswith('.npy'):
            fpath = os.path.join(engine_dir, fname)
            try:
                data = np.load(fpath, allow_pickle=False)
                file_size = os.path.getsize(fpath) / (1024**2)
                print(f"   {fname}")
                print(f"      Shape: {data.shape}, Size: {file_size:.1f} MB")
            except Exception as e:
                print(f"   {fname}: Error - {e}")

print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)

# Load and check actual shape
try:
    med_data = np.load(medllama_path, allow_pickle=False)
    actual_shape = med_data.shape

    if actual_shape[0] != 43474:
        print(f"\n⚠️  WARNING: MedLLaMA has {actual_shape[0]} entities, but dataset needs 43,474!")
        print(f"   This embedding file is for a different dataset.")
        print(f"\n   Solutions:")
        print(f"   1. Use the correct embedding file with 43,474 entities")
        print(f"   2. Or regenerate embeddings for this dataset")

    elif actual_shape[1] != 4096:
        print(f"\n⚠️  WARNING: MedLLaMA has {actual_shape[1]}D embeddings, not 4096D!")
        print(f"   File name suggests 4096D but actual is {actual_shape[1]}D")
        print(f"\n   Solutions:")
        print(f"   1. Rename file to match actual dimensions")
        print(f"   2. Or use correct 4096D embedding file")
    else:
        print(f"\n✓ MedLLaMA embeddings match dataset requirements!")

except:
    pass

print("="*70)

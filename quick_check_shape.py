import numpy as np

# Quick check
medllama = np.load("/root/fussion-and-verify-in-BKG/engine/medllama_pretrained_embeddings_4096.npy")
poincare = np.load("/root/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy")

print("MedLLaMA shape:", medllama.shape)
print("Poincaré shape:", poincare.shape)
print()
print("Dataset needs: (43474, 4096)")
print("MedLLaMA has:", medllama.shape)
print()

if medllama.shape[0] == 43474:
    print("✓ Number of entities matches!")
else:
    print(f"✗ Wrong number of entities: {medllama.shape[0]} vs 43474")

if medllama.shape[1] == 4096:
    print("✓ Embedding dimensions match!")
else:
    print(f"✗ Wrong dimensions: {medllama.shape[1]} vs 4096")
    print(f"  File name says 4096 but actual is {medllama.shape[1]}")

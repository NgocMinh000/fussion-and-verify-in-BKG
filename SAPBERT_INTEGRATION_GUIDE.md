# SapBERT Integration Guide

## Overview

SapBERT embeddings are now fully integrated into FuseLinker! You can use SapBERT as text embeddings without pre-generating and saving `.npy` files. The system will automatically generate embeddings on-the-fly during training.

## Key Features

✅ **On-the-fly generation**: No need to pre-generate embeddings
✅ **Keyword-based activation**: Use `sapbert_embeddings` as the filename
✅ **Backward compatible**: Existing `.npy` files still work
✅ **All model variants supported**: DistMult, TransE, ComplEx, ConvE

## How to Use SapBERT

### Basic Usage

Simply use `sapbert_embeddings` (or `sapbert`) as the text embedding filename:

```bash
cd fuselinker-complex

python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --use_n3_reg \
    --model_state_file suppkg_sapbert_model.pth
```

### What Happens Behind the Scenes

When you use `sapbert_embeddings` as the text embedding file:

1. **Data Processing**: FuseLinker first processes your data to create entity mappings
2. **Entity Extraction**: All entity names are extracted in the correct order
3. **SapBERT Loading**: The SapBERT model is loaded from HuggingFace
4. **Batch Encoding**: Entity names are encoded in batches (default: 128)
5. **Embedding Generation**: CLS token embeddings are extracted (768-dimensional)
6. **Training Starts**: Embeddings are passed to the model for training

### Supported Keywords

The system recognizes these keywords:
- `sapbert_embeddings` (recommended)
- `sapbert`
- `sapbert-embeddings`

### Example Output

When using SapBERT, you'll see:

```
Loading Pretrained Embeddings files...
Text embedding path: suppkg/sapbert_embeddings

======================================================================
Loading SapBERT Embeddings (On-the-fly Generation)
======================================================================
Device: cuda
GPU: NVIDIA GeForce RTX 3090
Number of entities: 2356

Loading SapBERT model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext...
✓ Model loaded successfully

Generating embeddings...
Batch size: 128
Max sequence length: 25
Encoding entities: 100%|████████████| 19/19 [00:02<00:00,  8.45batch/s]

✓ Successfully generated SapBERT embeddings
✓ Shape: (2356, 768)
✓ Dimension: 768
======================================================================

✓ Loaded Text Embeddings successfully! Shape: (2356, 768)
```

## Comparison with PubMedBERT

### Pre-generated PubMedBERT (Traditional Approach)

```bash
# Step 1: Pre-generate embeddings (separate script)
python generate_pubmedbert_embeddings.py --data suppkg

# Step 2: Use pre-generated file
python main.py \
    --text_embedding_file pubmedbert_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    ...
```

### On-the-fly SapBERT (New Approach)

```bash
# One command - embeddings generated automatically
python main.py \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file poincare_embeddings.npy \
    ...
```

## Full Training Examples

### ComplEx with SapBERT

```bash
cd fuselinker-complex

python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --use_n3_reg \
    --model_state_file sapbert_complex_model.pth
```

### DistMult with SapBERT

```bash
cd fuselinker

python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --model_state_file sapbert_distmult_model.pth
```

### TransE with SapBERT

```bash
cd fuselinker-transe

python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --model_state_file sapbert_transe_model.pth
```

### ConvE with SapBERT

```bash
cd fuselinker-conve

python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --model_state_file sapbert_conve_model.pth
```

## Technical Details

### SapBERT Model

- **Model**: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- **Base**: PubMedBERT with contrastive learning
- **Dimension**: 768
- **Tokenizer**: BERT tokenizer
- **Max Length**: 25 tokens (default)
- **Embedding**: [CLS] token representation

### Implementation

The SapBERT integration is implemented in `sapbert_loader.py`:

```python
from sapbert_loader import load_text_embeddings

# Smart loader - detects keyword or loads .npy file
embeddings = load_text_embeddings(
    embedding_path=text_embedding_path,
    index2entity=knowledge_graph.index2entity,
    use_cuda=args.use_cuda
)
```

### Performance

**Embedding Generation Speed** (example with 2356 entities):
- GPU (RTX 3090): ~2-3 seconds
- CPU: ~10-15 seconds

**Memory Usage**:
- Model: ~400MB
- Embeddings: ~7MB per 1000 entities (768-dim float32)

## Backward Compatibility

**All existing commands still work!** The integration is fully backward compatible:

### Using .npy files (still supported)

```bash
python main.py \
    --text_embedding_file pubmedbert_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    ...
```

### Using absolute paths (still supported)

```bash
python main.py \
    --text_embedding_file ~/embeddings/my_custom_embeddings.npy \
    --knowledge_embedding_file ~/embeddings/poincare.npy \
    ...
```

## Troubleshooting

### Error: "transformers not installed"

```bash
pip install transformers
```

### Error: "No module named 'sapbert_loader'"

Make sure you're running from the correct directory. The loader is in the root:
```bash
# Should be run from fuselinker-complex/, fuselinker/, etc.
cd fuselinker-complex
python main.py ...
```

### Slow embedding generation

- **Use GPU**: Make sure `--use_cuda True` is set
- **Increase batch size**: Modify `batch_size=128` in `sapbert_loader.py`
- **Check GPU memory**: If out of memory, reduce batch size

### SapBERT model download issues

First-time usage requires downloading the model (~400MB). Ensure:
- Internet connection is available
- HuggingFace is not blocked
- Sufficient disk space (~1GB)

## Advanced Usage

### Customizing SapBERT Parameters

Edit `sapbert_loader.py` to customize:

```python
embeddings = load_sapbert_embeddings(
    index2entity=index2entity,
    model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    batch_size=256,      # Increase for faster GPU
    max_length=32,       # Increase for longer entity names
    device='cuda'
)
```

### Using Different SapBERT Variants

You can modify the model name in `sapbert_loader.py`:

```python
# Original SapBERT
model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Other variants
model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
```

## Recommendations

### When to Use SapBERT

✅ **Use SapBERT when**:
- You have biomedical entities
- Entity names are standardized (e.g., UMLS concepts)
- You want synonym/variant mapping
- You need semantic similarity

✅ **Use PubMedBERT when**:
- Pre-computed embeddings are available
- Faster startup time is critical
- Embeddings are reused across experiments

### Optimal Configuration

For best results with SapBERT:

```bash
python main.py \
    --data suppkg \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file poincare_embeddings.npy \
    --w 0.75 \           # Balance text & knowledge embeddings
    --use_reciprocal \   # Add inverse relations
    --iterations 4000    # Sufficient training
```

## Citation

If you use SapBERT in your work, please cite:

```bibtex
@inproceedings{liu2021self,
  title={Self-Alignment Pretraining for Biomedical Entity Representations},
  author={Liu, Fangyu and Shareghi, Ehsan and Meng, Zaiqiao and Basaldella, Marco and Collier, Nigel},
  booktitle={NAACL},
  year={2021}
}
```

## Summary

The SapBERT integration provides:
- 🚀 **Convenience**: No pre-generation needed
- 🔄 **Compatibility**: Works with all model variants
- ⚡ **Speed**: Fast GPU-accelerated generation
- 🎯 **Quality**: State-of-the-art biomedical embeddings

Simply use `--text_embedding_file sapbert_embeddings` and you're ready to go!

#!/bin/bash
# Training commands for FuseLinker with embeddings from engine/ directory
#
# Based on your previous command:
# python main.py --data suppkg --text_embedding_file medllama_pretrained_embeddings_4096.npy \
#   --knowledge_embedding_file poincare_embeddings.npy --num_hidden_layers 2 --iterations 4000 \
#   --evaluate_every 100 --neg_sample_size_eval 100 --w 0.75 --model_state_file suppkg_model_state.pth

# ============================================================================
# IMPORTANT: Set the path to your engine directory
# ============================================================================

# Option 1: If engine/ is in parent directory
ENGINE_DIR="../engine"

# Option 2: If engine/ is somewhere else, use absolute path
# ENGINE_DIR="/path/to/your/engine"

# Option 3: Auto-detect if engine is in specific location
if [ -d "../engine" ]; then
    ENGINE_DIR="../engine"
elif [ -d "~/FuseLinker/engine" ]; then
    ENGINE_DIR="~/FuseLinker/engine"
else
    echo "Error: engine/ directory not found. Please set ENGINE_DIR variable."
    exit 1
fi

echo "Using ENGINE_DIR: $ENGINE_DIR"

# ============================================================================
# Available Text Embeddings (from your engine/ directory)
# ============================================================================

# BERT (768D)
BERT_EMB="$ENGINE_DIR/bert_pretrained_embeddings_768.npy"

# FlanT5 (768D)
FLANT5_EMB="$ENGINE_DIR/flant5_pretrained_embeddings_768.npy"

# Llama2 (4096D)
LLAMA2_EMB="$ENGINE_DIR/llama2_pretrained_embeddings_4096.npy"

# MedLLaMA (4096D) - Your previous choice
MEDLLAMA_EMB="$ENGINE_DIR/medllama_pretrained_embeddings_4096.npy"

# PubMedBERT (768D) - Recommended for biomedical
PUBMEDBERT_EMB="$ENGINE_DIR/pubmedbert_pretrained_embeddings_768.npy"

# Domain knowledge embeddings
POINCARE_EMB="$ENGINE_DIR/poincare_embeddings.npy"

# ============================================================================
# Recommendations by Embedding Dimension
# ============================================================================

# For 768D embeddings (BERT, FlanT5, PubMedBERT):
#   --n_hidden 200   (default, works well)
#   --n_hidden 300   (better performance, more memory)

# For 4096D embeddings (Llama2, MedLLaMA):
#   --n_hidden 200   (recommended, reduce from 4096→200)
#   --n_hidden 300   (if you have enough GPU memory)
#   Note: Autoencoder will compress 4096 → hidden_dim

# ============================================================================
# EXPERIMENT 1: DistMult (Baseline) with MedLLaMA
# ============================================================================

echo ""
echo "================================================================"
echo "Experiment 1: DistMult + MedLLaMA (Your Previous Setup)"
echo "================================================================"

cd fuselinker

# Short test run (4000 iterations like you did)
python main.py \
    --data suppkg \
    --text_embedding_file "$MEDLLAMA_EMB" \
    --knowledge_embedding_file "$POINCARE_EMB" \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_distmult_medllama.pth

# Full run (40000 iterations for better results)
# python main.py \
#     --data suppkg \
#     --text_embedding_file "$MEDLLAMA_EMB" \
#     --knowledge_embedding_file "$POINCARE_EMB" \
#     --num_hidden_layers 2 \
#     --iterations 40000 \
#     --evaluate_every 1000 \
#     --validate_every 2000 \
#     --neg_sample_size_eval 100 \
#     --w 0.75 \
#     --use_cuda \
#     --model_state_file suppkg_distmult_medllama_full.pth

cd ..

# ============================================================================
# EXPERIMENT 2: DistMult with PubMedBERT (Recommended for Biomedical)
# ============================================================================

echo ""
echo "================================================================"
echo "Experiment 2: DistMult + PubMedBERT (Recommended)"
echo "================================================================"

cd fuselinker

python main.py \
    --data suppkg \
    --text_embedding_file "$PUBMEDBERT_EMB" \
    --knowledge_embedding_file "$POINCARE_EMB" \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_distmult_pubmedbert.pth

cd ..

# ============================================================================
# EXPERIMENT 3: TransE with MedLLaMA
# ============================================================================

echo ""
echo "================================================================"
echo "Experiment 3: TransE + MedLLaMA"
echo "================================================================"

cd fuselinker-transe

python main.py \
    --data suppkg \
    --text_embedding_file "$MEDLLAMA_EMB" \
    --knowledge_embedding_file "$POINCARE_EMB" \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda \
    --model_state_file suppkg_transe_medllama.pth

cd ..

# ============================================================================
# EXPERIMENT 4: ComplEx with MedLLaMA
# ============================================================================

echo ""
echo "================================================================"
echo "Experiment 4: ComplEx + MedLLaMA (Expected Best)"
echo "================================================================"

cd fuselinker-complex

python main.py \
    --data suppkg \
    --text_embedding_file "$MEDLLAMA_EMB" \
    --knowledge_embedding_file "$POINCARE_EMB" \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.005 \
    --use_cuda \
    --model_state_file suppkg_complex_medllama.pth

cd ..

# ============================================================================
# EXPERIMENT 5: ConvE with MedLLaMA (Best Performance, Needs GPU)
# ============================================================================

echo ""
echo "================================================================"
echo "Experiment 5: ConvE + MedLLaMA (Best Performance)"
echo "================================================================"

cd fuselinker-conve

# Note: ConvE requires n_hidden to be height*width (default 10*20=200)
python main.py \
    --data suppkg \
    --text_embedding_file "$MEDLLAMA_EMB" \
    --knowledge_embedding_file "$POINCARE_EMB" \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 6000 \
    --evaluate_every 100 \
    --validate_every 500 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.003 \
    --dropout 0.3 \
    --use_cuda \
    --model_state_file suppkg_conve_medllama.pth

cd ..

# ============================================================================
# Quick Comparison: All Methods with PubMedBERT
# ============================================================================

echo ""
echo "================================================================"
echo "Quick Comparison: All Methods with PubMedBERT (768D)"
echo "================================================================"

# DistMult
cd fuselinker
python main.py --data suppkg --text_embedding_file "$PUBMEDBERT_EMB" \
    --knowledge_embedding_file "$POINCARE_EMB" --iterations 4000 \
    --w 0.75 --use_cuda --model_state_file suppkg_distmult_pubmed.pth
cd ..

# TransE
cd fuselinker-transe
python main.py --data suppkg --text_embedding_file "$PUBMEDBERT_EMB" \
    --knowledge_embedding_file "$POINCARE_EMB" --iterations 4000 \
    --w 0.75 --use_cuda --model_state_file suppkg_transe_pubmed.pth
cd ..

# ComplEx
cd fuselinker-complex
python main.py --data suppkg --text_embedding_file "$PUBMEDBERT_EMB" \
    --knowledge_embedding_file "$POINCARE_EMB" --iterations 4000 \
    --lr 0.005 --w 0.75 --use_cuda --model_state_file suppkg_complex_pubmed.pth
cd ..

# ConvE
cd fuselinker-conve
python main.py --data suppkg --text_embedding_file "$PUBMEDBERT_EMB" \
    --knowledge_embedding_file "$POINCARE_EMB" --iterations 6000 \
    --lr 0.003 --dropout 0.3 --w 0.75 --use_cuda --model_state_file suppkg_conve_pubmed.pth
cd ..

echo ""
echo "================================================================"
echo "All experiments completed!"
echo "================================================================"

#!/bin/bash
# Correct commands for FuseLinker with RTX 3090
# Using ABSOLUTE paths for embeddings in ~/fussion-and-verify-in-BKG/engine/

# ============================================================================
# IMPORTANT NOTES:
# 1. Use --use_cuda True (not just --use_cuda)
# 2. Embedding paths are ABSOLUTE (~/fussion-and-verify-in-BKG/engine/...)
# 3. Data path (--data suppkg) remains RELATIVE
# ============================================================================

echo "======================================================================"
echo "FuseLinker Training Commands - Ready to Run"
echo "======================================================================"
echo ""

# ============================================================================
# 1. QUICK TEST (100 iterations, ~2 minutes)
# ============================================================================

echo "1. Quick Test (100 iterations):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 100 \
    --evaluate_every 50 \
    --w 0.75 \
    --use_cuda True
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 2. DISTMULT + MEDLLAMA (4K iterations, ~30 minutes)
# ============================================================================

echo ""
echo "2. DistMult + MedLLaMA (4K iterations):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda True \
    --model_state_file suppkg_distmult_medllama_4k.pth
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 3. DISTMULT + PUBMEDBERT (4K iterations)
# ============================================================================

echo ""
echo "3. DistMult + PubMedBERT (4K iterations):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda True \
    --model_state_file suppkg_distmult_pubmedbert_4k.pth
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 4. TRANSE + MEDLLAMA (4K iterations)
# ============================================================================

echo ""
echo "4. TransE + MedLLaMA (4K iterations):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker-transe

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda True \
    --model_state_file suppkg_transe_medllama_4k.pth
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 5. COMPLEX + MEDLLAMA (4K iterations)
# ============================================================================

echo ""
echo "5. ComplEx + MedLLaMA (4K iterations):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda True \
    --model_state_file suppkg_complex_medllama_4k.pth
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 6. CONVE + MEDLLAMA (4K iterations)
# ============================================================================

echo ""
echo "6. ConvE + MedLLaMA (4K iterations):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker-conve

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda True \
    --model_state_file suppkg_conve_medllama_4k.pth
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 7. COMPLEX + PUBMEDBERT (4K iterations) - Best for comparison
# ============================================================================

echo ""
echo "7. ComplEx + PubMedBERT (4K iterations):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda True \
    --model_state_file suppkg_complex_pubmedbert_4k.pth
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 8. CONVE + PUBMEDBERT (4K iterations) - State-of-the-art
# ============================================================================

echo ""
echo "8. ConvE + PubMedBERT (4K iterations):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker-conve

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_cuda True \
    --model_state_file suppkg_conve_pubmedbert_4k.pth
EOF

echo ""
echo "======================================================================"
echo "Done! Choose a command and run it."
echo "======================================================================"

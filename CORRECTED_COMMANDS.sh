#!/bin/bash
# Correct commands for FuseLinker with RTX 3090
# Engine directory: ~/fussion-and-verify-in-BKG/engine/

# ============================================================================
# CORRECT SYNTAX: --use_cuda True (not --use_cuda)
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
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
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
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
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
    --text_embedding_file ../engine/pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
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
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
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
# 5. COMPLEX + MEDLLAMA (4K iterations) - Expected Best
# ============================================================================

echo ""
echo "5. ComplEx + MedLLaMA (4K iterations, Expected Best):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.005 \
    --use_cuda True \
    --model_state_file suppkg_complex_medllama_4k.pth
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 6. CONVE + MEDLLAMA (6K iterations) - Ultimate
# ============================================================================

echo ""
echo "6. ConvE + MedLLaMA (6K iterations, Ultimate Performance):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker-conve

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 6000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.003 \
    --dropout 0.3 \
    --use_cuda True \
    --model_state_file suppkg_conve_medllama_6k.pth
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 7. FULL TRAINING - COMPLEX + MEDLLAMA (40K iterations)
# ============================================================================

echo ""
echo "7. ComplEx + MedLLaMA - Full Training (40K iterations, ~4 hours):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 40000 \
    --evaluate_every 1000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.005 \
    --use_cuda True \
    --model_state_file suppkg_complex_medllama_40k.pth
EOF

echo ""
echo "======================================================================"

# ============================================================================
# 8. OPTIMIZED FOR RTX 3090 - Higher Hidden Dimension
# ============================================================================

echo ""
echo "8. ComplEx + MedLLaMA - Optimized for RTX 3090 (n_hidden=300):"
echo ""
cat << 'EOF'
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

python main.py --data suppkg \
    --text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ../engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 300 \
    --iterations 4000 \
    --evaluate_every 100 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --lr 0.005 \
    --use_cuda True \
    --model_state_file suppkg_complex_medllama_4k_hd300.pth
EOF

echo ""
echo "======================================================================"
echo "All commands ready!"
echo "Start with command #1 (quick test) to verify setup"
echo "======================================================================"

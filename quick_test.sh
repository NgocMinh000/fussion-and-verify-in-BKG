#!/bin/bash

#################################################################################
# Quick Test Script - Fast validation of all fixes (10 iterations each)
# Use this to quickly verify all fixes are working before running full tests
#################################################################################

DATA_DIR="suppkg"
TEXT_EMB="$HOME/fussion-and-verify-in-BKG/engine/llama2_pretrained_embeddings_4096.npy"
KNOW_EMB="$HOME/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy"
ITERATIONS=10  # Quick test - just 10 iterations
EVAL_EVERY=5

echo "================================================================================"
echo "QUICK TEST - Validating All Fixes (10 iterations each)"
echo "================================================================================"
echo ""

# Test 1: DistMult - verify no regression
echo "TEST 1/4: DistMult (should work same as before)..."
cd "$HOME/fussion-and-verify-in-BKG/fuselinker"
python main.py --data "$DATA_DIR" --text_embedding_file "$TEXT_EMB" \
    --knowledge_embedding_file "$KNOW_EMB" --iterations $ITERATIONS \
    --evaluate_every $EVAL_EVERY --w 0.75 --use_cuda True
echo ""

# Test 2: TransE - CRITICAL: verify Hits@1 > 0 (not 0!)
echo "TEST 2/4: TransE (CRITICAL: Hits@1 should be > 0, not 0!)..."
cd "$HOME/fussion-and-verify-in-BKG/fuselinker-transe"
python main.py --data "$DATA_DIR" --text_embedding_file "$TEXT_EMB" \
    --knowledge_embedding_file "$KNOW_EMB" --iterations $ITERATIONS \
    --evaluate_every $EVAL_EVERY --w 0.75 --use_cuda True
echo ""

# Test 3: ComplEx - verify architecture change works
echo "TEST 3/4: ComplEx (verify independent imaginary embeddings)..."
cd "$HOME/fussion-and-verify-in-BKG/fuselinker-complex"
python main.py --data "$DATA_DIR" --text_embedding_file "$TEXT_EMB" \
    --knowledge_embedding_file "$KNOW_EMB" --iterations $ITERATIONS \
    --evaluate_every $EVAL_EVERY --w 0.75 --use_cuda True --use_n3_reg
echo ""

# Test 4: ConvE - verify batch norm control works
echo "TEST 4/4: ConvE (verify batch norm control)..."
cd "$HOME/fussion-and-verify-in-BKG/fuselinker-conve"
python main.py --data "$DATA_DIR" --text_embedding_file "$TEXT_EMB" \
    --knowledge_embedding_file "$KNOW_EMB" --iterations $ITERATIONS \
    --evaluate_every $EVAL_EVERY --w 0.75 --use_cuda True
echo ""

echo "================================================================================"
echo "QUICK TEST COMPLETE!"
echo "================================================================================"
echo ""
echo "KEY THINGS TO CHECK:"
echo "  ✓ All methods completed without errors"
echo "  ✓ TransE shows Hits@1 > 0 (NOT 0 anymore!)"
echo "  ✓ ComplEx shows 'independent imaginary embeddings (N3 reg: True)' in output"
echo "  ✓ All methods show positive MRR values"
echo ""
echo "If all checks pass, run the full test:"
echo "  ./test_all_methods.sh"
echo ""
echo "================================================================================"

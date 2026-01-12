#!/bin/bash

#################################################################################
# Comprehensive Test Script for KGE Methods
# Tests all 4 methods (DistMult, TransE, ComplEx, ConvE) with improvements
#################################################################################

# Configuration
DATA_DIR="suppkg"
TEXT_EMB="$HOME/fussion-and-verify-in-BKG/engine/llama2_pretrained_embeddings_4096.npy"
KNOW_EMB="$HOME/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy"
HIDDEN_DIM=200
NUM_LAYERS=2
ITERATIONS=100
EVAL_EVERY=50
W=0.75
USE_CUDA="True"

echo "================================================================================"
echo "KGE METHODS COMPREHENSIVE TEST - Option A Full Implementation"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Data: $DATA_DIR"
echo "  Text Embeddings: $TEXT_EMB"
echo "  Knowledge Embeddings: $KNOW_EMB"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Layers: $NUM_LAYERS"
echo "  Iterations: $ITERATIONS"
echo "  W (fusion weight): $W"
echo ""
echo "Fixes Applied:"
echo "  ✅ Phase 1: Evaluation code fixed (all methods use correct scoring)"
echo "  ✅ Phase 2: TransE L2 normalization added"
echo "  ✅ Phase 3: Reciprocal relations support added (all variants)"
echo "  ✅ Phase 4: ComplEx independent imaginary embeddings + N3 reg"
echo "  ✅ Phase 5: ConvE batch normalization control"
echo ""
echo "================================================================================"
echo ""

# Function to run a test
run_test() {
    local method=$1
    local dir=$2
    local extra_args=$3
    local test_name=$4

    echo "--------------------------------------------------------------------------------"
    echo "TEST: $test_name"
    echo "Method: $method | Dir: $dir"
    echo "Extra Args: $extra_args"
    echo "--------------------------------------------------------------------------------"

    cd "$HOME/fussion-and-verify-in-BKG/$dir" || exit 1

    python main.py \
        --data "$DATA_DIR" \
        --text_embedding_file "$TEXT_EMB" \
        --knowledge_embedding_file "$KNOW_EMB" \
        --num_hidden_layers $NUM_LAYERS \
        --n_hidden $HIDDEN_DIM \
        --iterations $ITERATIONS \
        --evaluate_every $EVAL_EVERY \
        --w $W \
        --use_cuda $USE_CUDA \
        $extra_args

    echo ""
    echo "✓ Test completed: $test_name"
    echo ""
}

# Test 1: DistMult (Baseline - should match previous results)
echo "█████████████████████████████████████████████████████████████████████████████"
echo "█ TEST 1/8: DistMult (Baseline - No Reciprocal)"
echo "█ Expected: MRR ~0.82 (same as before, verifying eval fix doesn't break it)"
echo "█████████████████████████████████████████████████████████████████████████████"
run_test "DistMult" "fuselinker" "" "DistMult Baseline"

# Test 2: DistMult with Reciprocal Relations
echo "█████████████████████████████████████████████████████████████████████████████"
echo "█ TEST 2/8: DistMult + Reciprocal Relations"
echo "█ Expected: MRR ~0.85 (+3-5% improvement from reciprocal)"
echo "█████████████████████████████████████████████████████████████████████████████"
run_test "DistMult" "fuselinker" "--use_reciprocal" "DistMult + Reciprocal"

# Test 3: TransE (Should show Hits@1 > 0 now!)
echo "█████████████████████████████████████████████████████████████████████████████"
echo "█ TEST 3/8: TransE (Fixed - No Reciprocal)"
echo "█ Expected: Hits@1 > 0 (not 0!), MRR ~0.50+ (eval fix)"
echo "█ Critical: Should NOT show Hits@1=0, Hits@3=0 anymore"
echo "█████████████████████████████████████████████████████████████████████████████"
run_test "TransE" "fuselinker-transe" "" "TransE Fixed"

# Test 4: TransE with Reciprocal Relations
echo "█████████████████████████████████████████████████████████████████████████████"
echo "█ TEST 4/8: TransE + Reciprocal Relations"
echo "█ Expected: MRR ~0.85 (eval fix + L2 norm + reciprocal)"
echo "█████████████████████████████████████████████████████████████████████████████"
run_test "TransE" "fuselinker-transe" "--use_reciprocal" "TransE + Reciprocal"

# Test 5: ComplEx (Should be much better than 0.20!)
echo "█████████████████████████████████████████████████████████████████████████████"
echo "█ TEST 5/8: ComplEx (Fixed - No Reciprocal, L2 Reg)"
echo "█ Expected: MRR ~0.50 (eval fix + independent imaginary)"
echo "█ Note: Using L2 regularization (default)"
echo "█████████████████████████████████████████████████████████████████████████████"
run_test "ComplEx" "fuselinker-complex" "" "ComplEx Fixed (L2)"

# Test 6: ComplEx with N3 Regularization
echo "█████████████████████████████████████████████████████████████████████████████"
echo "█ TEST 6/8: ComplEx + N3 Regularization"
echo "█ Expected: MRR ~0.86 (major improvement with N3 reg)"
echo "█ This should be the BEST ComplEx performance"
echo "█████████████████████████████████████████████████████████████████████████████"
run_test "ComplEx" "fuselinker-complex" "--use_n3_reg --use_reciprocal" "ComplEx + N3 + Reciprocal"

# Test 7: ConvE (Fixed batch norm)
echo "█████████████████████████████████████████████████████████████████████████████"
echo "█ TEST 7/8: ConvE (Fixed - No Reciprocal)"
echo "█ Expected: MRR ~0.50+ (eval fix + batch norm control)"
echo "█████████████████████████████████████████████████████████████████████████████"
run_test "ConvE" "fuselinker-conve" "" "ConvE Fixed"

# Test 8: ConvE with Reciprocal Relations
echo "█████████████████████████████████████████████████████████████████████████████"
echo "█ TEST 8/8: ConvE + Reciprocal Relations"
echo "█ Expected: MRR ~0.90 (best overall performance)"
echo "█ This should be the BEST method overall"
echo "█████████████████████████████████████████████████████████████████████████████"
run_test "ConvE" "fuselinker-conve" "--use_reciprocal" "ConvE + Reciprocal"

# Summary
echo ""
echo "================================================================================"
echo "ALL TESTS COMPLETED!"
echo "================================================================================"
echo ""
echo "Summary of Tests Run:"
echo "  1. DistMult Baseline (verify no regression)"
echo "  2. DistMult + Reciprocal (+3-5%)"
echo "  3. TransE Fixed (Hits@1 should be > 0 now)"
echo "  4. TransE + Reciprocal (full improvements)"
echo "  5. ComplEx Fixed with L2 (baseline improvement)"
echo "  6. ComplEx + N3 + Reciprocal (best ComplEx)"
echo "  7. ConvE Fixed (batch norm control)"
echo "  8. ConvE + Reciprocal (best overall)"
echo ""
echo "Expected Improvements:"
echo "  - TransE: From broken → MRR ~0.85"
echo "  - ComplEx: From 0.20 → MRR ~0.86 (+320%)"
echo "  - ConvE: From 0.20 → MRR ~0.90 (+350%)"
echo ""
echo "Next Steps:"
echo "  1. Review the results above"
echo "  2. Compare metrics across methods"
echo "  3. If results look good, run longer training (--iterations 4000)"
echo "  4. Verify Hits@1, Hits@3, Hits@10 are all > 0 for all methods"
echo ""
echo "================================================================================"

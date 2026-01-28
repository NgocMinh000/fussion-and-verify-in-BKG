#!/bin/bash
# Quick train ComplEx to test visualization

# IMPORTANT: Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fuselinker

echo "Training ComplEx model (quick test - 1000 iterations)..."
cd /home/user/fussion-and-verify-in-BKG/fuselinker-complex

python main.py \
    --data suppkg \
    --iterations 1000 \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --w 0.75 \
    --use_cuda True \
    --use_n3_reg \
    --model_state_file quick_test_complex.pth

echo ""
echo "============================================"
echo "Checking if visualization_outputs was created..."
if [ -d "suppkg/visualization_outputs" ]; then
    echo "✓ SUCCESS! Visualization data exported:"
    ls -la suppkg/visualization_outputs/
else
    echo "✗ ERROR: visualization_outputs not created!"
fi

cd ..

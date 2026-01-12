#!/bin/bash
# Test visualization với data hiện có

echo "============================================"
echo "Testing Visualization System"
echo "============================================"
echo ""

# Check if visualization_outputs exists
if [ ! -d "fuselinker-complex/suppkg/visualization_outputs" ]; then
    echo "✗ ERROR: fuselinker-complex/suppkg/visualization_outputs/ not found"
    echo ""
    echo "Please train model first:"
    echo "  cd fuselinker-complex"
    echo "  python main.py --data suppkg --iterations 1000 --w 0.75 --use_cuda True"
    exit 1
fi

echo "✓ visualization_outputs/ found"
echo ""

# Check files
echo "Checking required files..."
files=(
    "fuselinker-complex/suppkg/entity2index.pkl"
    "fuselinker-complex/suppkg/index2entity.pkl"
    "fuselinker-complex/suppkg/visualization_outputs/node_embeddings.npy"
    "fuselinker-complex/suppkg/visualization_outputs/relation_embeddings.npy"
    "fuselinker-complex/suppkg/visualization_outputs/train_graph.json"
    "fuselinker-complex/suppkg/visualization_outputs/test_graph.json"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo ""
    echo "✗ ERROR: Some required files are missing"
    exit 1
fi

echo ""
echo "✓ All required files found!"
echo ""
echo "============================================"
echo "Running Link Predictor (10 queries)..."
echo "============================================"
echo ""

python -m visualization.link_predictor \
    --viz_dir fuselinker-complex/suppkg/visualization_outputs \
    --output_file fuselinker-complex/suppkg/link_predictions.json \
    --num_queries 10 \
    --top_k 5 \
    --model_type complex \
    --show_samples 2

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ SUCCESS!"
    echo "============================================"
    echo ""
    echo "Predictions saved to: fuselinker-complex/suppkg/link_predictions.json"
    echo ""
    echo "Next steps:"
    echo "  1. Generate explanations:"
    echo "     python -m visualization.explainer --num_queries 5"
    echo ""
    echo "  2. Create visualizations:"
    echo "     python -m visualization.prediction_visualizer"
    echo ""
    echo "  3. Launch dashboard:"
    echo "     streamlit run visualization/prediction_app.py"
else
    echo ""
    echo "✗ ERROR: Link predictor failed"
    exit 1
fi

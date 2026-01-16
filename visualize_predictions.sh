#!/bin/bash
#
# Visualize Link Predictions - Quick Launch Script
#
# This script helps you quickly visualize link predictions from trained models.
#
# Usage:
#     # Basic usage
#     ./visualize_predictions.sh primekg complex
#
#     # With custom top_k
#     ./visualize_predictions.sh primekg complex 20
#
#     # Launch dashboard only (if predictions already exist)
#     ./visualize_predictions.sh primekg complex --dashboard-only
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <data_dir> <model_type> [top_k] [--dashboard-only]"
    echo ""
    echo "Examples:"
    echo "  $0 primekg complex              # Extract predictions + launch dashboard"
    echo "  $0 suppkg distmult 20           # Extract top 20 predictions"
    echo "  $0 primekg complex --dashboard-only  # Launch dashboard only"
    echo ""
    echo "Model types: distmult, transe, complex, conve"
    exit 1
fi

DATA_DIR="$1"
MODEL_TYPE="$2"
TOP_K="${3:-10}"
DASHBOARD_ONLY=false

# Check for --dashboard-only flag
if [ "$3" = "--dashboard-only" ] || [ "$4" = "--dashboard-only" ]; then
    DASHBOARD_ONLY=true
fi

# Determine model directory
case "$MODEL_TYPE" in
    distmult)
        MODEL_DIR="fuselinker"
        ;;
    transe)
        MODEL_DIR="fuselinker-transe"
        ;;
    complex)
        MODEL_DIR="fuselinker-complex"
        ;;
    conve)
        MODEL_DIR="fuselinker-conve"
        ;;
    *)
        print_error "Unknown model type: $MODEL_TYPE"
        echo "Valid types: distmult, transe, complex, conve"
        exit 1
        ;;
esac

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "================================================"
print_info "FuseLinker Link Prediction Visualizer"
print_info "================================================"
print_info "Data directory: $DATA_DIR"
print_info "Model type: $MODEL_TYPE"
print_info "Model directory: $MODEL_DIR"
print_info "Top K: $TOP_K"
print_info "================================================"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    print_error "Model directory not found: $MODEL_DIR"
    exit 1
fi

cd "$MODEL_DIR"

# Check if visualization_outputs exists
VIZ_DIR="$DATA_DIR/visualization_outputs"
if [ ! -d "$VIZ_DIR" ]; then
    print_error "Visualization directory not found: $VIZ_DIR"
    print_info "Make sure you have trained the model and exported visualization data."
    exit 1
fi

print_info "Found visualization directory: $VIZ_DIR"

# Check required files
REQUIRED_FILES=(
    "$VIZ_DIR/node_embeddings.npy"
    "$VIZ_DIR/relation_embeddings.npy"
    "$VIZ_DIR/train_graph.json"
    "$VIZ_DIR/test_graph.json"
    "$DATA_DIR/entity2index.pkl"
    "$DATA_DIR/index2entity.pkl"
    "$DATA_DIR/relation2index.pkl"
    "$DATA_DIR/index2relation.pkl"
)

print_info "\nChecking required files..."
ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    print_error "\nSome required files are missing!"
    print_info "Make sure the model training completed successfully and exported visualization data."
    exit 1
fi

print_info "✓ All required files found"

# Extract predictions (unless --dashboard-only)
PREDICTIONS_FILE="predictions_${MODEL_TYPE}_${DATA_DIR//\//_}.json"

if [ "$DASHBOARD_ONLY" = true ]; then
    print_warning "\nSkipping prediction extraction (--dashboard-only mode)"

    if [ ! -f "$PREDICTIONS_FILE" ]; then
        print_warning "Predictions file not found: $PREDICTIONS_FILE"
        print_info "Run without --dashboard-only to extract predictions first"
    else
        print_info "Using existing predictions: $PREDICTIONS_FILE"
    fi
else
    print_info "\nExtracting link predictions..."
    print_info "This may take a few minutes..."

    python -m visualization.link_predictor \
        --model_dir . \
        --data_dir "$VIZ_DIR" \
        --output "$PREDICTIONS_FILE" \
        --top_k "$TOP_K"

    if [ $? -eq 0 ]; then
        print_info "✓ Predictions extracted successfully: $PREDICTIONS_FILE"
    else
        print_error "Failed to extract predictions"
        exit 1
    fi
fi

# Launch dashboard
print_info "\n================================================"
print_info "Launching Visualization Dashboard"
print_info "================================================"
print_info "Dashboard will be available at: http://localhost:5000"
print_info "Press Ctrl+C to stop the server"
print_info "================================================\n"

python -m visualization.app \
    --data_dir "$VIZ_DIR" \
    --port 5000

# Note: If dashboard fails, print helpful message
if [ $? -ne 0 ]; then
    print_error "\nDashboard failed to start"
    print_info "\nTry running manually:"
    echo "  cd $MODEL_DIR"
    echo "  python -m visualization.app --data_dir $VIZ_DIR --port 5000"
fi

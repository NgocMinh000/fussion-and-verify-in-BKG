#!/bin/bash
#
# Full Pipeline: Entity Names → UMLS CUI → FuseLinker Format
#
# This script automates the complete conversion workflow.
#
# Usage:
#   ./convert_kg_full_pipeline.sh
#   ./convert_kg_full_pipeline.sh --keep-unmapped
#

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print colored message
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default paths (modify as needed)
KG_FILE="fuselinker/mybkg/kg_clean.txt"
MAPPING_FILE="fuselinker/mybkg/umls_mapping_triples.txt"
CUI_FILE="fuselinker/mybkg/kg_cui.txt"
OUTPUT_DIR="fuselinker/mybkg_cui"
ALLOW_UNMAPPED=""

# Parse arguments
if [ "$1" = "--keep-unmapped" ]; then
    ALLOW_UNMAPPED="--allow-unmapped"
    print_info "Mode: Keep unmapped entities"
else
    print_info "Mode: Skip triples with unmapped entities"
fi

print_info "================================================"
print_info "Full Knowledge Graph Conversion Pipeline"
print_info "================================================"
print_info "Input KG:      $KG_FILE"
print_info "Mapping file:  $MAPPING_FILE"
print_info "CUI output:    $CUI_FILE"
print_info "Final output:  $OUTPUT_DIR"
print_info "================================================"

# Check if input files exist
if [ ! -f "$KG_FILE" ]; then
    print_error "KG file not found: $KG_FILE"
    print_info "Please ensure kg_clean.txt exists in fuselinker/mybkg/"
    exit 1
fi

if [ ! -f "$MAPPING_FILE" ]; then
    print_error "Mapping file not found: $MAPPING_FILE"
    print_info "Please ensure umls_mapping_triples.txt exists in fuselinker/mybkg/"
    exit 1
fi

print_info "✓ Input files found"

# Step 1: Convert entity names to CUI codes
print_info "\n================================================"
print_info "Step 1/3: Converting entity names → UMLS CUI codes"
print_info "================================================\n"

python convert_names_to_cui.py \
    --kg "$KG_FILE" \
    --mapping "$MAPPING_FILE" \
    --output "$CUI_FILE" \
    $ALLOW_UNMAPPED

if [ $? -ne 0 ]; then
    print_error "Step 1 failed: Error converting to CUI codes"
    exit 1
fi

print_info "\n✓ Step 1 complete: CUI graph created"

# Check if unmapped entities file exists
if [ -f "unmapped_entities.txt" ]; then
    UNMAPPED_COUNT=$(tail -n +3 unmapped_entities.txt | wc -l)
    if [ $UNMAPPED_COUNT -gt 0 ]; then
        print_warning "Found $UNMAPPED_COUNT unmapped entities"
        print_info "See: unmapped_entities.txt"
    fi
fi

# Step 2: Convert CUI graph to FuseLinker format
print_info "\n================================================"
print_info "Step 2/3: Converting CUI graph → FuseLinker format"
print_info "================================================\n"

python convert_to_fuselinker_format.py \
    --input "$CUI_FILE" \
    --output "$OUTPUT_DIR" \
    --stats

if [ $? -ne 0 ]; then
    print_error "Step 2 failed: Error converting to FuseLinker format"
    exit 1
fi

print_info "\n✓ Step 2 complete: FuseLinker format created"

# Step 3: Validation and summary
print_info "\n================================================"
print_info "Step 3/3: Validation and Summary"
print_info "================================================\n"

# Check output files
print_info "Checking output files..."
REQUIRED_FILES=(
    "$OUTPUT_DIR/train.tsv"
    "$OUTPUT_DIR/valid.tsv"
    "$OUTPUT_DIR/test.tsv"
    "$OUTPUT_DIR/entity2index.pkl"
    "$OUTPUT_DIR/index2entity.pkl"
    "$OUTPUT_DIR/relation2index.pkl"
    "$OUTPUT_DIR/index2relation.pkl"
)

ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✓ $file ($SIZE)"
    else
        echo "  ✗ $file (missing)"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    print_error "\nSome output files are missing!"
    exit 1
fi

print_info "\n✓ All output files created successfully"

# Print summary
print_info "\n================================================"
print_info "Conversion Complete!"
print_info "================================================"

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files created:"
echo "  - train.tsv, valid.tsv, test.tsv (triples)"
echo "  - entity2index.pkl, index2entity.pkl (mappings)"
echo "  - relation2index.pkl, index2relation.pkl (mappings)"
echo ""
echo "Next steps:"
echo ""
echo "1. Train ComplEx model:"
echo "   cd fuselinker-complex"
echo "   python main.py \\"
echo "       --data $OUTPUT_DIR \\"
echo "       --text_embedding_file sapbert_embeddings \\"
echo "       --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \\"
echo "       --num_hidden_layers 2 \\"
echo "       --iterations 4000 \\"
echo "       --use_reciprocal \\"
echo "       --w 0.75 \\"
echo "       --use_cuda True \\"
echo "       --use_n3_reg \\"
echo "       --model_state_file mybkg_cui_model.pth"
echo ""
echo "2. Or train DistMult model:"
echo "   cd fuselinker"
echo "   python main.py --data $OUTPUT_DIR --text_embedding_file sapbert_embeddings ..."
echo ""
echo "3. Visualize predictions after training:"
echo "   cd ~/fussion-and-verify-in-BKG"
echo "   ./visualize_predictions.sh mybkg_cui complex"
echo ""
print_info "================================================"

# Optional: Show sample data
print_info "\nSample data (first 5 triples from train.tsv):"
echo "----------------------------------------"
head -5 "$OUTPUT_DIR/train.tsv"
echo "----------------------------------------"

print_info "\n✓ Pipeline complete! Ready to train."

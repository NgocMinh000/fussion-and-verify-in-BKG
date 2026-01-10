#!/bin/bash
# Setup embeddings path - Auto-detect or create symlink to engine directory

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "======================================================================"
echo -e "${BLUE}FuseLinker Embeddings Path Setup${NC}"
echo "======================================================================"

# Try to auto-detect engine directory
ENGINE_PATHS=(
    "../engine"
    "../../engine"
    "$HOME/engine"
    "$HOME/FuseLinker/engine"
    "/home/user/engine"
    "/home/user/FuseLinker/engine"
)

FOUND=false
ENGINE_DIR=""

echo ""
echo "Searching for engine/ directory..."

for path in "${ENGINE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo -e "  ${GREEN}✓${NC} Found: $path"

        # Check if it has embedding files
        if ls "$path"/*.npy >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Contains .npy embedding files"
            ENGINE_DIR="$path"
            FOUND=true
            break
        else
            echo -e "  ${YELLOW}⚠${NC} No .npy files found"
        fi
    fi
done

if [ "$FOUND" = false ]; then
    echo ""
    echo -e "${YELLOW}⚠ Could not auto-detect engine/ directory${NC}"
    echo ""
    echo "Please enter the full path to your engine/ directory:"
    read -p "> " ENGINE_DIR

    if [ ! -d "$ENGINE_DIR" ]; then
        echo -e "${RED}✗ Directory not found: $ENGINE_DIR${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✓ Using engine directory: $ENGINE_DIR${NC}"

# List embeddings
echo ""
echo "Available embeddings:"
ls -lh "$ENGINE_DIR"/*.npy 2>/dev/null | awk '{print "  " $9, "(" $5 ")"}'

# Check for required embeddings
echo ""
echo "Checking required embeddings..."

REQUIRED_EMBEDDINGS=(
    "poincare_embeddings.npy"
)

TEXT_EMBEDDINGS=(
    "medllama_pretrained_embeddings_4096.npy"
    "pubmedbert_pretrained_embeddings_768.npy"
    "bert_pretrained_embeddings_768.npy"
    "flant5_pretrained_embeddings_768.npy"
    "llama2_pretrained_embeddings_4096.npy"
)

all_good=true

for emb in "${REQUIRED_EMBEDDINGS[@]}"; do
    if [ -f "$ENGINE_DIR/$emb" ]; then
        echo -e "  ${GREEN}✓${NC} $emb"
    else
        echo -e "  ${RED}✗${NC} $emb (required)"
        all_good=false
    fi
done

echo ""
echo "Text embeddings (at least one required):"
text_count=0
for emb in "${TEXT_EMBEDDINGS[@]}"; do
    if [ -f "$ENGINE_DIR/$emb" ]; then
        echo -e "  ${GREEN}✓${NC} $emb"
        ((text_count++))
    else
        echo -e "  ${YELLOW}⚠${NC} $emb (optional)"
    fi
done

if [ $text_count -eq 0 ]; then
    echo -e "${RED}✗ No text embeddings found!${NC}"
    all_good=false
fi

if [ "$all_good" = false ]; then
    echo ""
    echo -e "${RED}✗ Missing required embedding files${NC}"
    echo "Please ensure all required embeddings are in $ENGINE_DIR"
    exit 1
fi

# Create symlink option
echo ""
echo "======================================================================"
echo "Setup Options"
echo "======================================================================"
echo ""
echo "You can either:"
echo "  1. Use relative path in commands (recommended)"
echo "  2. Create symlink 'engine' in current directory"
echo ""
read -p "Create symlink? [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -L "engine" ]; then
        echo "Removing existing symlink..."
        rm engine
    fi

    ln -s "$ENGINE_DIR" engine
    echo -e "${GREEN}✓ Created symlink: ./engine -> $ENGINE_DIR${NC}"

    echo ""
    echo "You can now use:"
    echo -e "  ${BLUE}--text_embedding_file engine/medllama_pretrained_embeddings_4096.npy${NC}"
    echo -e "  ${BLUE}--knowledge_embedding_file engine/poincare_embeddings.npy${NC}"
else
    # Calculate relative path
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REL_PATH=$(realpath --relative-to="$SCRIPT_DIR" "$ENGINE_DIR")

    echo ""
    echo "Use relative path in commands:"
    echo -e "  ${BLUE}--text_embedding_file $REL_PATH/medllama_pretrained_embeddings_4096.npy${NC}"
    echo -e "  ${BLUE}--knowledge_embedding_file $REL_PATH/poincare_embeddings.npy${NC}"
fi

# Save path to config file
echo ""
echo "Saving configuration..."
cat > embeddings_path.conf <<EOF
# Embeddings path configuration
ENGINE_DIR="$ENGINE_DIR"

# Example usage:
# source embeddings_path.conf
# python main.py --text_embedding_file \$ENGINE_DIR/medllama_pretrained_embeddings_4096.npy
EOF

echo -e "${GREEN}✓ Saved to embeddings_path.conf${NC}"

# Generate example commands
echo ""
echo "======================================================================"
echo "Example Commands"
echo "======================================================================"

if [ -L "engine" ]; then
    EMB_PATH="engine"
else
    EMB_PATH="$ENGINE_DIR"
fi

echo ""
echo -e "${BLUE}1. DistMult + MedLLaMA:${NC}"
cat <<EOF
cd fuselinker
python main.py --data suppkg \\
    --text_embedding_file $EMB_PATH/medllama_pretrained_embeddings_4096.npy \\
    --knowledge_embedding_file $EMB_PATH/poincare_embeddings.npy \\
    --iterations 4000 --w 0.75 --use_cuda
EOF

echo ""
echo -e "${BLUE}2. ComplEx + PubMedBERT:${NC}"
cat <<EOF
cd fuselinker-complex
python main.py --data suppkg \\
    --text_embedding_file $EMB_PATH/pubmedbert_pretrained_embeddings_768.npy \\
    --knowledge_embedding_file $EMB_PATH/poincare_embeddings.npy \\
    --iterations 4000 --lr 0.005 --w 0.75 --use_cuda
EOF

echo ""
echo "======================================================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "======================================================================"
echo ""
echo "You can now run training commands."
echo "See RUN_COMMANDS.md for detailed instructions."
echo ""

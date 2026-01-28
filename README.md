# fussion-and-verify-in-BKG

# Cài đặt git
apt update
apt install git -y
<!-- claude/fix-colbert-yescale-integration-Kr9zQ -->
<!-- 
git clone -b claude/analyze-stage3-umls-mapping-Kr9zQ \
https://github.com/NgocMinh000/GFM.git -->

git clone -b claude/review-umls-mapping-stage3-CpIYp \
https://github.com/NgocMinh000/GFM.git

git clone -b claude/visualize-fused-links-Gz5nE \
https://github.com/NgocMinh000/fussion-and-verify-in-BKG.git
# Cài miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Thêm conda vào PATH thủ công
echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda --version
# Accept Terms of Service của Conda
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda init bash
source ~/.bashrc
# Tạo env 
conda create -n gfm-rag python=3.12 -y
conda activate gfm-rag
# Tải các tài nguyên tính toán 
conda install cuda-toolkit -c nvidia/label/cuda-12.4.1 # Replace with your desired CUDA version
pip install gfmrag
# 1. Install dependencies (nếu chưa có)
pip install matplotlib seaborn
# Gỡ faiss cũ
pip uninstall -y faiss faiss-cpu faiss-gpu faiss-gpu-cu11 faiss-gpu-cu12
# Check gỡ cài đặt faiss trên conda
pip show faiss
pip show faiss-cpu
pip show faiss-gpu
pip show faiss-gpu-cu12

# Cài faiss-gpu-cu12
pip install faiss-gpu-cu12
# Check faiss-gpu-cu12
python - << 'EOF'
import faiss, inspect
print("faiss file:", faiss.__file__)
print("faiss version:", faiss.__version__)
print("num_gpus:", faiss.get_num_gpus())
EOF

# Bước 1: Setup Environment
cd fussion-and-verify-in-BKG/
bash install_environment.sh
conda activate fuselinker
python test_installation.py
python check_gpu.py

# Bước 2: Generate SapBERT (Optional)
python generate_sapbert_embeddings.py --data fuselinker/suppkg

# Run test

cd ~/fussion-and-verify-in-BKG
git pull origin claude/visualize-fused-links-Gz5nE


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

# 1. Install dependencies (nếu chưa có)
pip install matplotlib seaborn
# Chạy workflow tạo BKG
cd ~/GFM
python -m gfmrag.workflow.stage1_index_dataset
python -m gfmrag.workflow.stage2_entity_resolution


# Tải unzip
apt update
apt install -y unzip

cd ~/GFM
# Tải umls
wget -c \
"https://uts-ws.nlm.nih.gov/download?url=https://download.nlm.nih.gov/umls/kss/2024AA/umls-2024AA-full.zip&apiKey=defd6fbb-279b-4b1d-85aa-3cc54486c976" \
-O umls-2024AA-full.zip


mkdir -p data/umls
mv umls-2024AA-full.zip data/umls/
cd data/umls

unzip umls-2024AA-full
cd 2024AA-full
unzip mmsys.zip


cd /root/GFM/data/umls/2024AA-full






# File 2024aa-1-meta.nlm và 2024aa-2-meta.nlm chứa META data
# Unzip chúng:

unzip -q 2024aa-1-meta.nlm -d META1
unzip -q 2024aa-2-meta.nlm -d META2








# Lấy RRF cần thiết

echo "🔧 UMLS RRF Decompression & Merge Script"
echo "========================================"

cd /root/GFM/data/umls/2024AA-full

# Create main META directory
mkdir -p /root/GFM/data/umls/META

echo ""
echo "📦 Processing split files..."

# Function to decompress and merge split files
process_split_file() {
    local base_name=$1
    local output_dir="/root/GFM/data/umls/META"
    
    echo "  → Processing $base_name..."
    
    # Find all parts (.aa.gz, .ab.gz, .ac.gz, etc.)
    local parts=($(find META1/2024AA/META META2/2024AA/META -name "${base_name}.*.gz" 2>/dev/null | sort))
    
    if [ ${#parts[@]} -gt 1 ]; then
        echo "    Found ${#parts[@]} parts, merging..."
        
        # Decompress and concatenate all parts
        for part in "${parts[@]}"; do
            echo "      - $(basename $part)"
        done
        
        # Decompress all parts and merge into one file
        zcat "${parts[@]}" > "$output_dir/$base_name"
        
        echo "    ✓ Merged into $base_name"
    elif [ ${#parts[@]} -eq 1 ]; then
        echo "    Single file, decompressing..."
        gunzip -c "${parts[0]}" > "$output_dir/$base_name"
        echo "    ✓ Decompressed $base_name"
    fi
}

# Function to process regular (non-split) files
process_regular_file() {
    local file=$1
    local base_name=$(basename "$file" .gz)
    local output_dir="/root/GFM/data/umls/META"
    
    echo "  → Processing $base_name..."
    gunzip -c "$file" > "$output_dir/$base_name"
    echo "    ✓ Decompressed"
}

# Process MRCONSO (split file)
echo ""
echo "📂 Processing MRCONSO (split files)..."
process_split_file "MRCONSO.RRF"

# Process MRSTY
echo ""
echo "📂 Processing MRSTY..."
mrsty_files=($(find META1/2024AA/META META2/2024AA/META -name "MRSTY.RRF*.gz" 2>/dev/null))
if [ ${#mrsty_files[@]} -gt 0 ]; then
    process_regular_file "${mrsty_files[0]}"
fi

# Process MRDEF
echo ""
echo "📂 Processing MRDEF..."
mrdef_files=($(find META1/2024AA/META META2/2024AA/META -name "MRDEF.RRF*.gz" 2>/dev/null))
if [ ${#mrdef_files[@]} -gt 0 ]; then
    process_regular_file "${mrdef_files[0]}"
fi

# Process all other RRF files
echo ""
echo "📂 Processing other RRF files..."
for file in $(find META1/2024AA/META META2/2024AA/META -name "*.RRF*.gz" 2>/dev/null | sort -u); do
    base_name=$(basename "$file" | sed 's/\.[a-z][a-z]\.gz$//' | sed 's/\.gz$//')
    
    # Skip if already processed
    if [[ "$base_name" == "MRCONSO.RRF" ]] || \
       [[ "$base_name" == "MRSTY.RRF" ]] || \
       [[ "$base_name" == "MRDEF.RRF" ]]; then
        continue
    fi
    
    # Check if it's a split file
    if [[ "$file" =~ \.[a-z][a-z]\.gz$ ]]; then
        # Only process the first part, process_split_file will handle all parts
        if [[ "$file" =~ \.aa\.gz$ ]]; then
            process_split_file "$base_name"
        fi
    else
        process_regular_file "$file"
    fi
done

# Verification
echo ""
echo "✅ Verification:"
echo "==============="

check_file() {
    local filename=$1
    if [ -f "/root/GFM/data/umls/META/$filename" ]; then
        local size=$(du -h "/root/GFM/data/umls/META/$filename" | cut -f1)
        local lines=$(wc -l < "/root/GFM/data/umls/META/$filename")
        printf "✓ %-20s %8s  %'15d lines\n" "$filename" "$size" "$lines"
    else
        echo "✗ $filename: NOT FOUND"
    fi
}

check_file "MRCONSO.RRF"
check_file "MRSTY.RRF"
check_file "MRDEF.RRF"

echo ""
total_files=$(ls /root/GFM/data/umls/META/*.RRF 2>/dev/null | wc -l)
echo "📊 Total RRF files extracted: $total_files"
echo "📁 Location: /root/GFM/data/umls/META/"

# Cleanup option
echo ""
read -p "🗑️  Delete temporary META1/META2 folders? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf META1 META2
    echo "✓ Cleaned up"
fi

echo ""
echo "🎉 All done! Files ready for pipeline."
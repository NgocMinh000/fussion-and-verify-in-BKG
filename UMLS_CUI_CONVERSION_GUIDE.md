# Hướng dẫn chuyển đổi Knowledge Graph sang UMLS CUI và FuseLinker

## Tổng quan

Workflow này giúp bạn chuyển đổi knowledge graph từ **entity names** → **UMLS CUI codes** → **FuseLinker format**.

**3 bước chính:**
1. Convert entity names → UMLS CUI codes (`convert_names_to_cui.py`)
2. Convert CUI triples → FuseLinker format (`convert_to_fuselinker_format.py`)
3. Train model với FuseLinker

## Input Files

### 1. `umls_mapping_triples.txt` - Mapping file
Format: `entity_name|mapped_to_cui|CUI_CODE`

```
astra zeneca|mapped_to_cui|C5440878
astrocyte morphology|mapped_to_cui|C4021991
asymmetric spasticity in upper and lower limbs|mapped_to_cui|C1273957
copper|mapped_to_cui|C0009968
oxygen|mapped_to_cui|C0030054
```

### 2. `kg_clean.txt` - Knowledge graph với entity names
Format: `entity1,relation,entity2`

```
copper,is classified as part of,growth substances
copper,is classified as part of,elements
copper,is classified as part of,food
oxygen,is,essential element for human survival
oxygen,is used in,medical oxygen therapy
medical oxygen therapy,treats,emphysema
```

## Bước 1: Convert Entity Names → UMLS CUI Codes

### Cách sử dụng cơ bản

```bash
cd ~/fussion-and-verify-in-BKG

python convert_names_to_cui.py \
    --kg fuselinker/mybkg/kg_clean.txt \
    --mapping fuselinker/mybkg/umls_mapping_triples.txt \
    --output fuselinker/mybkg/kg_cui.txt
```

### Output

**File: `kg_cui.txt`**
```csv
head,relation,tail
C0009968,is classified as part of,C0018284
C0009968,is classified as part of,C0013879
C0009968,is classified as part of,C0016452
C0030054,is,C5555555
C0030054,is used in,C1234567
C1234567,treats,C0013990
```

**File: `unmapped_entities.txt`** (nếu có entities không map được)
```
Unmapped Entities
============================================================

growth substances
essential element for human survival
```

### Statistics Output

```
======================================================================
Conversion Statistics
======================================================================
Original triples:      150
Converted triples:     142
Skipped triples:       8
Unmapped entities:     5

Success rate:          94.7%

Converted graph:
  Unique entities:     98
  Unique relations:    12
  Total triples:       142
======================================================================
```

## Bước 2: Convert CUI Graph → FuseLinker Format

Sau khi có `kg_cui.txt`, chuyển sang format FuseLinker:

```bash
python convert_to_fuselinker_format.py \
    --input fuselinker/mybkg/kg_cui.txt \
    --output fuselinker/mybkg_cui \
    --stats
```

### Output

```
mybkg_cui/
├── train.tsv              (80% triples)
├── valid.tsv              (10% triples)
├── test.tsv               (10% triples)
├── entity2index.pkl
├── index2entity.pkl
├── relation2index.pkl
└── index2relation.pkl
```

**File: `train.tsv`**
```tsv
C0009968	is classified as part of	C0018284
C0009968	is classified as part of	C0013879
C0030054	is used in	C1234567
C1234567	treats	C0013990
```

## Bước 3: Train Model với FuseLinker

```bash
cd fuselinker-complex

python main.py \
    --data mybkg_cui \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --use_n3_reg \
    --model_state_file mybkg_cui_model.pth
```

## Workflow Đầy Đủ

### Ví dụ hoàn chỉnh

```bash
# Di chuyển vào thư mục project
cd ~/fussion-and-verify-in-BKG

# Bước 1: Convert names → CUI codes
python convert_names_to_cui.py \
    --kg fuselinker/mybkg/kg_clean.txt \
    --mapping fuselinker/mybkg/umls_mapping_triples.txt \
    --output fuselinker/mybkg/kg_cui.txt

# Kiểm tra kết quả
echo "=== Converted CUI graph (first 10 lines) ==="
head -10 fuselinker/mybkg/kg_cui.txt

# Kiểm tra unmapped entities
if [ -f unmapped_entities.txt ]; then
    echo "=== Unmapped entities ==="
    cat unmapped_entities.txt
fi

# Bước 2: Convert CUI graph → FuseLinker format
python convert_to_fuselinker_format.py \
    --input fuselinker/mybkg/kg_cui.txt \
    --output fuselinker/mybkg_cui \
    --stats

# Kiểm tra output files
echo "=== Output files ==="
ls -lh fuselinker/mybkg_cui/

# Bước 3: Train model
cd fuselinker-complex

python main.py \
    --data mybkg_cui \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --use_n3_reg \
    --model_state_file mybkg_cui_model.pth

# Bước 4: Visualize predictions
cd ~/fussion-and-verify-in-BKG
./visualize_predictions.sh mybkg_cui complex
```

## Options Chi Tiết

### convert_names_to_cui.py

```bash
python convert_names_to_cui.py --help
```

**Arguments:**

- `--kg`: Input knowledge graph file (required)
- `--mapping`: UMLS mapping file (required)
- `--output`: Output file for CUI-based graph (default: kg_cui.txt)
- `--allow-unmapped`: Keep unmapped entities as-is instead of skipping (flag)
- `--report-unmapped`: File to save unmapped entities list (default: unmapped_entities.txt)

**Ví dụ:**

```bash
# Basic usage
python convert_names_to_cui.py \
    --kg kg_clean.txt \
    --mapping umls_mapping_triples.txt \
    --output kg_cui.txt

# Keep unmapped entities (không bỏ triples có entity chưa map)
python convert_names_to_cui.py \
    --kg kg_clean.txt \
    --mapping umls_mapping_triples.txt \
    --output kg_cui.txt \
    --allow-unmapped

# Custom unmapped report file
python convert_names_to_cui.py \
    --kg kg_clean.txt \
    --mapping umls_mapping_triples.txt \
    --output kg_cui.txt \
    --report-unmapped my_unmapped.txt
```

### convert_to_fuselinker_format.py

```bash
python convert_to_fuselinker_format.py --help
```

**Arguments:**

- `--input, -i`: Input CUI graph file (required)
- `--output, -o`: Output directory (required)
- `--train`: Train split ratio (default: 0.8)
- `--valid`: Valid split ratio (default: 0.1)
- `--test`: Test split ratio (default: 0.1)
- `--seed`: Random seed (default: 42)
- `--stats`: Show detailed statistics (flag)

## Xử lý Unmapped Entities

### Kiểm tra unmapped entities

Sau khi chạy `convert_names_to_cui.py`, kiểm tra file `unmapped_entities.txt`:

```bash
cat unmapped_entities.txt
```

### Option 1: Bỏ triples có unmapped entities (Default)

```bash
# Triples có entity không map được sẽ bị bỏ qua
python convert_names_to_cui.py \
    --kg kg_clean.txt \
    --mapping umls_mapping_triples.txt \
    --output kg_cui.txt
```

**Ưu điểm:** Chỉ giữ lại triples với CUI codes hợp lệ
**Nhược điểm:** Mất một số triples

### Option 2: Giữ unmapped entities (--allow-unmapped)

```bash
# Giữ entity name gốc cho entities chưa map được
python convert_names_to_cui.py \
    --kg kg_clean.txt \
    --mapping umls_mapping_triples.txt \
    --output kg_cui.txt \
    --allow-unmapped
```

**Output sẽ có:**
```csv
head,relation,tail
C0009968,is classified as part of,C0018284
C0009968,is classified as part of,growth substances    ← kept as-is
oxygen,is,essential element                             ← kept as-is
```

**Ưu điểm:** Không mất triples
**Nhược điểm:** Mix CUI codes và entity names

### Option 3: Thêm mapping cho unmapped entities

Nếu có nhiều unmapped entities, bạn có thể:

1. Xem list unmapped entities:
```bash
cat unmapped_entities.txt
```

2. Tìm CUI codes cho các entities này (thủ công hoặc dùng UMLS API)

3. Thêm vào file `umls_mapping_triples.txt`:
```
growth substances|mapped_to_cui|C0018284
essential element|mapped_to_cui|C0013879
```

4. Chạy lại conversion:
```bash
python convert_names_to_cui.py \
    --kg kg_clean.txt \
    --mapping umls_mapping_triples.txt \
    --output kg_cui.txt
```

## Entity Name Normalization

Script tự động normalize entity names để matching tốt hơn:

**Normalization rules:**
- Convert to lowercase
- Remove extra whitespace
- Trim leading/trailing spaces

**Examples:**
```
"Copper"           → "copper"
"OXYGEN  "         → "oxygen"
"Medical  Therapy" → "medical therapy"
```

**Matching:**
```
kg_clean.txt:              "Copper, is, Element"
umls_mapping_triples.txt:  "copper|mapped_to_cui|C0009968"
                           ↓ matched (case-insensitive)
kg_cui.txt:                "C0009968,is,C0013879"
```

## Troubleshooting

### Lỗi: "No mappings found"

**Nguyên nhân:** File mapping sai format hoặc rỗng

**Giải pháp:**
```bash
# Kiểm tra file format
head -5 umls_mapping_triples.txt

# Phải có format: entity|mapped_to_cui|CUI
# VD: copper|mapped_to_cui|C0009968
```

### Lỗi: "Too many unmapped entities"

**Nguyên nhân:** Entity names trong kg_clean.txt không match với mapping file

**Giải pháp:**

1. Kiểm tra case sensitivity:
```bash
# Xem entities trong KG
cut -d',' -f1,3 fuselinker/mybkg/kg_clean.txt | sort -u | head -20

# Xem entities trong mapping
cut -d'|' -f1 fuselinker/mybkg/umls_mapping_triples.txt | sort -u | head -20

# So sánh
```

2. Check whitespace issues:
```bash
# Find entities with weird whitespace
grep -P '\s\s+' kg_clean.txt
```

3. Dùng `--allow-unmapped` tạm thời:
```bash
python convert_names_to_cui.py \
    --kg kg_clean.txt \
    --mapping umls_mapping_triples.txt \
    --output kg_cui.txt \
    --allow-unmapped
```

### Lỗi: "Invalid format in kg_clean.txt"

**Nguyên nhân:** File có format không đúng

**Giải pháp:**
```bash
# Kiểm tra format
head -10 kg_clean.txt

# Phải có format: entity1,relation,entity2
# VD: copper,is classified as,element
```

### Warning: "Duplicate mapping"

**Nguyên nhân:** Cùng một entity map tới nhiều CUI codes khác nhau

**Example:**
```
copper|mapped_to_cui|C0009968
copper|mapped_to_cui|C1234567  ← duplicate!
```

**Giải pháp:** Chọn CUI code chính xác nhất và xóa duplicates.

## Performance Tips

### Với file lớn (>100K triples)

```bash
# Tăng tốc bằng cách dùng PyPy (nếu có)
pypy3 convert_names_to_cui.py \
    --kg large_kg.txt \
    --mapping umls_mapping.txt \
    --output kg_cui.txt

# Hoặc dùng parallel processing (split file trước)
split -l 50000 kg_clean.txt kg_part_
```

### Memory optimization

```bash
# Nếu file quá lớn, xử lý theo batch
# Sẽ cần modify script để đọc file theo chunks
```

## Examples

### Example 1: Basic conversion

```bash
# Files
# - fuselinker/mybkg/kg_clean.txt (100 triples)
# - fuselinker/mybkg/umls_mapping_triples.txt (50 mappings)

python convert_names_to_cui.py \
    --kg fuselinker/mybkg/kg_clean.txt \
    --mapping fuselinker/mybkg/umls_mapping_triples.txt \
    --output fuselinker/mybkg/kg_cui.txt

# Output:
# ✓ Loaded 50 entity-to-CUI mappings
# ✓ Loaded 100 triples
# ✓ Converted 85 triples
# ✗ Skipped 15 triples due to unmapped entities
# ⚠ Found 8 unmapped entities
```

### Example 2: Keep all triples

```bash
python convert_names_to_cui.py \
    --kg fuselinker/mybkg/kg_clean.txt \
    --mapping fuselinker/mybkg/umls_mapping_triples.txt \
    --output fuselinker/mybkg/kg_cui.txt \
    --allow-unmapped

# Output:
# ✓ Converted 100 triples (no triples skipped)
# ⚠ Found 8 unmapped entities (kept as-is)
```

### Example 3: Full pipeline

```bash
#!/bin/bash
# full_pipeline.sh - Complete conversion pipeline

set -e

# Config
KG_FILE="fuselinker/mybkg/kg_clean.txt"
MAPPING_FILE="fuselinker/mybkg/umls_mapping_triples.txt"
CUI_FILE="fuselinker/mybkg/kg_cui.txt"
OUTPUT_DIR="fuselinker/mybkg_cui"

echo "Step 1: Convert to CUI codes"
python convert_names_to_cui.py \
    --kg "$KG_FILE" \
    --mapping "$MAPPING_FILE" \
    --output "$CUI_FILE"

echo ""
echo "Step 2: Convert to FuseLinker format"
python convert_to_fuselinker_format.py \
    --input "$CUI_FILE" \
    --output "$OUTPUT_DIR" \
    --stats

echo ""
echo "Step 3: Check output"
ls -lh "$OUTPUT_DIR/"

echo ""
echo "✓ Conversion complete!"
echo "Next: Train model with:"
echo "  cd fuselinker-complex"
echo "  python main.py --data $OUTPUT_DIR --text_embedding_file sapbert_embeddings ..."
```

## Validation

### Kiểm tra conversion đúng không

```bash
# 1. Kiểm tra số lượng triples
echo "Original KG:"
wc -l fuselinker/mybkg/kg_clean.txt

echo "CUI KG:"
wc -l fuselinker/mybkg/kg_cui.txt

# 2. Kiểm tra CUI codes hợp lệ
echo "Sample CUI triples:"
head -5 fuselinker/mybkg/kg_cui.txt

# 3. Kiểm tra unique entities
echo "Unique CUI codes:"
tail -n +2 fuselinker/mybkg/kg_cui.txt | cut -d',' -f1,3 | tr ',' '\n' | sort -u | wc -l

# 4. Kiểm tra relations
echo "Unique relations:"
tail -n +2 fuselinker/mybkg/kg_cui.txt | cut -d',' -f2 | sort -u
```

## Best Practices

### ✅ Recommended

1. **Kiểm tra mapping file trước:**
   ```bash
   head -20 umls_mapping_triples.txt
   wc -l umls_mapping_triples.txt
   ```

2. **Review unmapped entities:**
   ```bash
   cat unmapped_entities.txt
   # Quyết định: skip hay thêm mapping
   ```

3. **Validate CUI codes:**
   ```bash
   # Check if CUIs are valid (start with C followed by numbers)
   grep -v '^C[0-9]' kg_cui.txt | head
   ```

4. **Backup original files:**
   ```bash
   cp kg_clean.txt kg_clean.txt.backup
   ```

### ⚠️ Common Mistakes

1. ❌ Không check unmapped entities
2. ❌ Sử dụng mapping file sai format
3. ❌ Quên normalize entity names
4. ❌ Không validate CUI codes

## Summary

✅ **Script 1**: `convert_names_to_cui.py` - Names → CUI codes
✅ **Script 2**: `convert_to_fuselinker_format.py` - CUI graph → FuseLinker format
✅ **Full workflow**: Names → CUI → FuseLinker → Train

**Quickest workflow:**
```bash
# 1. Convert to CUI
python convert_names_to_cui.py \
    --kg fuselinker/mybkg/kg_clean.txt \
    --mapping fuselinker/mybkg/umls_mapping_triples.txt \
    --output fuselinker/mybkg/kg_cui.txt

# 2. Convert to FuseLinker format
python convert_to_fuselinker_format.py \
    --input fuselinker/mybkg/kg_cui.txt \
    --output fuselinker/mybkg_cui

# 3. Train
cd fuselinker-complex
python main.py --data mybkg_cui --text_embedding_file sapbert_embeddings ...
```

Sẵn sàng convert knowledge graph của bạn! 🚀

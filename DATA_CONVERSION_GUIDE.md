# Data Conversion Guide

## Chuyển đổi dữ liệu sang format FuseLinker

Script này chuyển đổi file CSV/TXT với format:
```
head,relation,tail
C3495801,ppi,C0263313
C0524662,ppi,C0020097
C0341503,ppi,C0040963
```

Thành format FuseLinker:
- `train.tsv` (80% data)
- `valid.tsv` (10% data)
- `test.tsv` (10% data)
- `entity2index.pkl` (entity → index mapping)
- `index2entity.pkl` (index → entity mapping)
- `relation2index.pkl` (relation → index mapping)
- `index2relation.pkl` (index → relation mapping)

## Cách sử dụng

### 1. Basic Usage

```bash
python convert_to_fuselinker_format.py \
    --input fuselinker/mybkg/umls_triples_multi_v2.txt \
    --output fuselinker/mybkg_new
```

Kết quả:
```
fuselinker/mybkg_new/
├── train.tsv
├── valid.tsv
├── test.tsv
├── entity2index.pkl
├── index2entity.pkl
├── relation2index.pkl
└── index2relation.pkl
```

### 2. Custom Split Ratios

Thay đổi tỷ lệ train/valid/test:

```bash
python convert_to_fuselinker_format.py \
    --input umls_triples_multi_v2.txt \
    --output output_dir \
    --train 0.7 \
    --valid 0.15 \
    --test 0.15
```

### 3. Show Statistics

Hiển thị thống kê chi tiết cho mỗi split:

```bash
python convert_to_fuselinker_format.py \
    --input umls_triples_multi_v2.txt \
    --output output_dir \
    --stats
```

### 4. Custom Random Seed

Để reproducibility:

```bash
python convert_to_fuselinker_format.py \
    --input umls_triples_multi_v2.txt \
    --output output_dir \
    --seed 123
```

## Input Format Support

Script hỗ trợ nhiều format:

### CSV with comma delimiter
```csv
head,relation,tail
C3495801,ppi,C0263313
C0524662,ppi,C0020097
```

### TSV with tab delimiter
```tsv
head	relation	tail
C3495801	ppi	C0263313
C0524662	ppi	C0020097
```

### Different column names
Nếu file có 3 cột, script sẽ tự động đổi tên thành `head,relation,tail`.

## Output Format

Các file output là **tab-separated**, **không có header**:

**train.tsv:**
```
C3495801	ppi	C0263313
C0524662	ppi	C0020097
C0341503	ppi	C0040963
...
```

**valid.tsv:**
```
C0023343	ppi	C0020097
C0456498	ppi	C0270075
...
```

**test.tsv:**
```
C1234567	ppi	C9876543
...
```

## Example Output

```
======================================================================
FuseLinker Format Converter
======================================================================
Input:  umls_triples_multi_v2.txt
Output: mybkg_new
Split:  train=0.8, valid=0.1, test=0.1
Seed:   42
======================================================================
Loading triples from: umls_triples_multi_v2.txt
Detected delimiter: COMMA
✓ Loaded 5000 triples
✓ Unique entities: 2500
✓ Unique relations: 5

Original Data Statistics:
  Triples: 5000
  Entities: 2500
  Relations: 5

  Relation distribution:
    ppi: 2000 (40.0%)
    pathway: 1500 (30.0%)
    disease: 800 (16.0%)
    drug: 500 (10.0%)
    gene: 200 (4.0%)

Splitting data: 80% train, 10% valid, 10% test
✓ Train: 4000 triples
✓ Valid: 500 triples
✓ Test: 500 triples

Creating entity and relation mappings...
✓ Entity mappings: 2500 unique entities
✓ Relation mappings: 5 unique relations

Saving TSV files...
✓ Saved train.tsv: mybkg_new/train.tsv (4000 triples)
✓ Saved valid.tsv: mybkg_new/valid.tsv (500 triples)
✓ Saved test.tsv: mybkg_new/test.tsv (500 triples)

Saving pickle files...
✓ Saved entity2index.pkl: mybkg_new/entity2index.pkl (2500 items)
✓ Saved index2entity.pkl: mybkg_new/index2entity.pkl (2500 items)
✓ Saved relation2index.pkl: mybkg_new/relation2index.pkl (5 items)
✓ Saved index2relation.pkl: mybkg_new/index2relation.pkl (5 items)

======================================================================
Conversion Complete!
======================================================================
✓ Created 7 files in: mybkg_new

  TSV files (triples):
    - train.tsv: 4000 triples
    - valid.tsv: 500 triples
    - test.tsv: 500 triples

  Pickle files (mappings):
    - entity2index.pkl: 2500 entities
    - index2entity.pkl: 2500 entities
    - relation2index.pkl: 5 relations
    - index2relation.pkl: 5 relations

Next steps:
1. Use with FuseLinker:
   cd fuselinker-complex
   python main.py --data mybkg_new --text_embedding_file sapbert_embeddings ...
======================================================================
```

## Sử dụng với FuseLinker

Sau khi chuyển đổi, dùng với FuseLinker:

### Với DistMult
```bash
cd fuselinker

python main.py \
    --data mybkg_new \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True
```

### Với ComplEx
```bash
cd fuselinker-complex

python main.py \
    --data mybkg_new \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --use_n3_reg
```

### Với TransE
```bash
cd fuselinker-transe

python main.py \
    --data mybkg_new \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True
```

### Với ConvE
```bash
cd fuselinker-conve

python main.py \
    --data mybkg_new \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True
```

## Full Workflow Example

### Step 1: Convert data
```bash
python convert_to_fuselinker_format.py \
    --input fuselinker/mybkg/umls_triples_multi_v2.txt \
    --output fuselinker/mybkg_umls \
    --stats
```

### Step 2: Train with ComplEx + SapBERT
```bash
cd fuselinker-complex

python main.py \
    --data mybkg_umls \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --use_n3_reg \
    --model_state_file mybkg_umls_model.pth
```

### Step 3: Extract predictions
```bash
python -m visualization.link_predictor \
    --model_dir . \
    --data_dir mybkg_umls/visualization_outputs \
    --output predictions_mybkg_umls.json \
    --top_k 10
```

### Step 4: Visualize
```bash
python -m visualization.app
```

## Script Options

### Required Arguments
- `--input, -i`: Input file path (CSV/TXT with header)
- `--output, -o`: Output directory path

### Optional Arguments
- `--train`: Training set ratio (default: 0.8)
- `--valid`: Validation set ratio (default: 0.1)
- `--test`: Test set ratio (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--stats`: Show detailed statistics for each split

### Help
```bash
python convert_to_fuselinker_format.py --help
```

## Notes

### ✅ Supported
- CSV files with comma delimiter
- TSV files with tab delimiter
- Files with header row
- Auto-detection of delimiter
- Custom split ratios
- Reproducible splits (random seed)

### ⚠️ Important
- Input file **must** have header row
- Output files are **tab-separated** without header
- Split ratios must sum to 1.0
- Default split: 80% train, 10% valid, 10% test

### 📊 Statistics
Use `--stats` flag to see:
- Number of triples in each split
- Number of unique entities
- Number of unique relations
- Relation distribution
- Entity distribution (heads vs tails)

## Troubleshooting

### Error: "Ratios must sum to 1.0"
```bash
# Wrong
--train 0.8 --valid 0.2 --test 0.2  # Sum = 1.2

# Correct
--train 0.7 --valid 0.15 --test 0.15  # Sum = 1.0
```

### Error: "File not found"
Make sure input file path is correct:
```bash
# Check file exists
ls -la fuselinker/mybkg/umls_triples_multi_v2.txt

# Use absolute path if needed
python convert_to_fuselinker_format.py \
    --input /full/path/to/umls_triples_multi_v2.txt \
    --output output_dir
```

### Wrong delimiter detected
If auto-detection fails, edit the script and force delimiter:
```python
# In load_triples function, replace auto-detection with:
delimiter = ','  # Force comma
# or
delimiter = '\t'  # Force tab
```

## Summary

✅ **Script**: `convert_to_fuselinker_format.py`
✅ **Input**: CSV/TXT with header (head, relation, tail)
✅ **Output**:
   - 3 TSV files: train.tsv, valid.tsv, test.tsv (tab-separated, no header)
   - 4 PKL files: entity2index.pkl, index2entity.pkl, relation2index.pkl, index2relation.pkl
✅ **Usage**: `python convert_to_fuselinker_format.py -i input.txt -o output_dir`
✅ **Default Split**: 80% train / 10% valid / 10% test

Ready to convert your UMLS triples! 🚀

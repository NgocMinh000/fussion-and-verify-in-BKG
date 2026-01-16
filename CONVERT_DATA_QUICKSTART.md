# Quick Start - Chuyển đổi dữ liệu cho FuseLinker

## Tổng quan

Script `convert_to_fuselinker_format.py` chuyển đổi file CSV/TXT sang format FuseLinker.

**Input:**
```csv
head,relation,tail
C3495801,ppi,C0263313
C0524662,ppi,C0020097
C0341503,ppi,C0040963
```

**Output:** 7 files
- `train.tsv`, `valid.tsv`, `test.tsv` (tab-separated, no header)
- `entity2index.pkl`, `index2entity.pkl` (entity mappings)
- `relation2index.pkl`, `index2relation.pkl` (relation mappings)

## Cách dùng cơ bản

### 1. Chuyển đổi file của bạn

```bash
python convert_to_fuselinker_format.py \
    --input fuselinker/mybkg/umls_triples_multi_v2.txt \
    --output fuselinker/mybkg_converted
```

### 2. Xem kết quả

```bash
ls fuselinker/mybkg_converted/

# Output:
# train.tsv
# valid.tsv
# test.tsv
# entity2index.pkl
# index2entity.pkl
# relation2index.pkl
# index2relation.pkl
```

### 3. Sử dụng với FuseLinker

```bash
cd fuselinker-complex

python main.py \
    --data mybkg_converted \
    --text_embedding_file sapbert_embeddings \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --iterations 4000 \
    --use_reciprocal \
    --w 0.75 \
    --use_cuda True \
    --use_n3_reg
```

## Ví dụ với output

```bash
python convert_to_fuselinker_format.py \
    --input umls_triples_multi_v2.txt \
    --output mybkg_new \
    --stats
```

**Kết quả:**
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
  Entities: 2500 (heads: 2300, tails: 2400)
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
    - test.tsv: 2 triples

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

## Options nâng cao

### Tùy chỉnh tỷ lệ split

```bash
# 70% train, 15% valid, 15% test
python convert_to_fuselinker_format.py \
    --input umls_triples.txt \
    --output mybkg_new \
    --train 0.7 \
    --valid 0.15 \
    --test 0.15
```

### Thay đổi random seed

```bash
# Để tạo split khác nhau
python convert_to_fuselinker_format.py \
    --input umls_triples.txt \
    --output mybkg_new \
    --seed 123
```

### Xem thống kê chi tiết

```bash
python convert_to_fuselinker_format.py \
    --input umls_triples.txt \
    --output mybkg_new \
    --stats  # Hiển thị thống kê cho từng split
```

### Xem help

```bash
python convert_to_fuselinker_format.py --help
```

## Workflow đầy đủ

### Step 1: Chuyển đổi dữ liệu
```bash
python convert_to_fuselinker_format.py \
    --input fuselinker/mybkg/umls_triples_multi_v2.txt \
    --output fuselinker/mybkg_umls \
    --stats
```

### Step 2: Train ComplEx với SapBERT
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
    --model_state_file mybkg_umls_complex.pth
```

### Step 3: Đánh giá model
Model sẽ tự động đánh giá và hiển thị:
```
MR: 45.234567
MRR: 0.635841
Hits @ 1 = 0.488712
Hits @ 3 = 0.723456
Hits @ 10 = 0.876543
```

### Step 4: Extract predictions
```bash
python -m visualization.link_predictor \
    --model_dir . \
    --data_dir mybkg_umls/visualization_outputs \
    --output predictions_mybkg.json \
    --top_k 10
```

### Step 5: Visualize
```bash
python -m visualization.app
# Mở browser tại http://localhost:5000
```

## Format chi tiết

### Input format (umls_triples_multi_v2.txt)
```csv
head,relation,tail
C3495801,ppi,C0263313
C0524662,ppi,C0020097
C0341503,ppi,C0040963
```

**Lưu ý:**
- Phải có header row: `head,relation,tail`
- Delimiter: comma (`,`) hoặc tab (`\t`) - tự động detect
- Entities: UMLS concepts (C-codes) hoặc bất kỳ string nào
- Relations: tên quan hệ (ppi, pathway, disease, etc.)

### Output format

**TSV files (train.tsv, valid.tsv, test.tsv):**
```tsv
C3495801	ppi	C0263313
C0524662	ppi	C0020097
C0341503	ppi	C0040963
```
- Tab-separated (`\t`)
- NO header
- Same format as suppkg files

**Pickle files:**
- `entity2index.pkl`: `{'C3495801': 0, 'C0263313': 1, ...}`
- `index2entity.pkl`: `{0: 'C3495801', 1: 'C0263313', ...}`
- `relation2index.pkl`: `{('head', 'ppi', 'tail'): 0, ...}`
- `index2relation.pkl`: `{0: ('head', 'ppi', 'tail'), ...}`

## So sánh với suppkg

### Cấu trúc tương tự:
```
suppkg/                       mybkg_converted/
├── train.tsv                ├── train.tsv
├── valid.tsv                ├── valid.tsv
├── test.tsv                 ├── test.tsv
├── entity2index.pkl         ├── entity2index.pkl
├── index2entity.pkl         ├── index2entity.pkl
├── relation2index.pkl       ├── relation2index.pkl
└── index2relation.pkl       └── index2relation.pkl
```

### Sử dụng giống nhau:
```bash
# suppkg
python main.py --data suppkg --text_embedding_file sapbert_embeddings ...

# mybkg_converted
python main.py --data mybkg_converted --text_embedding_file sapbert_embeddings ...
```

## Troubleshooting

### Lỗi: "No module named 'pandas'"
```bash
pip install pandas
```

### Lỗi: "Ratios must sum to 1.0"
```bash
# Sai: train + valid + test ≠ 1.0
--train 0.8 --valid 0.2 --test 0.2

# Đúng: train + valid + test = 1.0
--train 0.7 --valid 0.15 --test 0.15
```

### Lỗi: "File not found"
```bash
# Kiểm tra file tồn tại
ls -la fuselinker/mybkg/umls_triples_multi_v2.txt

# Dùng đường dẫn tuyệt đối
python convert_to_fuselinker_format.py \
    --input /full/path/to/umls_triples_multi_v2.txt \
    --output output_dir
```

### Delimiter không đúng
Nếu auto-detection sai, sửa trong script:
```python
# Line ~40 trong convert_to_fuselinker_format.py
delimiter = ','   # Force comma
# hoặc
delimiter = '\t'  # Force tab
```

## Tips

### ✅ Best Practices
- Luôn dùng `--stats` để xem thống kê chi tiết
- Giữ tỷ lệ mặc định (80/10/10) cho dataset nhỏ-trung bình
- Dùng random seed cố định để reproduce kết quả
- Backup file gốc trước khi chuyển đổi

### ⚠️ Lưu ý
- File input PHẢI có header row
- Output files là tab-separated, không có header
- Entity và relation mappings được tạo từ TOÀN BỘ dataset (không chỉ train)
- Đảm bảo đủ disk space (output ~1-2x size of input)

### 🚀 Performance
- Nhanh: ~1000 triples/second
- Memory: ~100MB cho 10K triples
- Disk: ~same size as input file

## Summary

✅ **Script**: `convert_to_fuselinker_format.py`
✅ **Chức năng**: Chuyển CSV/TXT → FuseLinker format (7 files)
✅ **Usage**: `python convert_to_fuselinker_format.py -i input.txt -o output_dir`
✅ **Default**: 80% train / 10% valid / 10% test
✅ **Output**: 3 TSV + 4 PKL files

**Đơn giản nhất:**
```bash
python convert_to_fuselinker_format.py \
    --input umls_triples_multi_v2.txt \
    --output mybkg_new
```

Sau đó dùng ngay với FuseLinker! 🎉

Xem chi tiết: `DATA_CONVERSION_GUIDE.md`

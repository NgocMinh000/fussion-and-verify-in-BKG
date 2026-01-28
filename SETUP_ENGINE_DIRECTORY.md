# Hướng dẫn: Setup thư mục engine với embeddings

## Vấn đề đã được fix

✅ Đã sửa code trong tất cả các thư mục fuselinker để xử lý đúng đường dẫn relative và absolute.

**Các file đã sửa:**
- `fuselinker/main.py`
- `fuselinker-transe/main.py`
- `fuselinker-complex/main.py`
- `fuselinker-conve/main.py`

**Thay đổi:** Code giờ đây sẽ kiểm tra nếu đường dẫn bắt đầu bằng `../` hoặc là absolute path, thì sẽ sử dụng trực tiếp mà không thêm prefix `{args.data}/`

## Bạn cần làm gì tiếp theo

### Bước 1: Tạo thư mục engine

```bash
cd ~/fussion-and-verify-in-BKG
mkdir -p engine
```

### Bước 2: Copy các file embeddings vào thư mục engine

Bạn cần copy các file embeddings sau vào `~/fussion-and-verify-in-BKG/engine/`:

**Bắt buộc (để chạy baseline):**
- `medllama_pretrained_embeddings_4096.npy` (4096 dimensions)
- `poincare_embeddings.npy` (50 dimensions)

**Khuyến nghị (để so sánh kết quả):**
- `pubmedbert_pretrained_embeddings_768.npy` (768 dimensions)
- `bert_pretrained_embeddings_768.npy`
- `flant5_pretrained_embeddings.npy`
- `llama2_pretrained_embeddings.npy`

```bash
# Ví dụ: copy từ vị trí cũ của bạn
cp /path/to/old/location/medllama_pretrained_embeddings_4096.npy ~/fussion-and-verify-in-BKG/engine/
cp /path/to/old/location/poincare_embeddings.npy ~/fussion-and-verify-in-BKG/engine/
cp /path/to/old/location/pubmedbert_pretrained_embeddings_768.npy ~/fussion-and-verify-in-BKG/engine/
```

### Bước 3: Kiểm tra các file đã tồn tại

```bash
ls -lh ~/fussion-and-verify-in-BKG/engine/*.npy
```

Kết quả mong muốn:
```
-rw-r--r-- 1 user user 712M Jan 10 12:00 medllama_pretrained_embeddings_4096.npy
-rw-r--r-- 1 user user 8.5M Jan 10 12:00 poincare_embeddings.npy
-rw-r--r-- 1 user user 134M Jan 10 12:00 pubmedbert_pretrained_embeddings_768.npy
```

### Bước 4: Test lại lệnh

```bash
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
```

**Kết quả mong muốn:**
```
Loading Pretrained Embeddings files...
Loaded Text Embeddings file successfully!        ← ✅ Thành công!
Loaded Domain Knowledge Embeddings file successfully!  ← ✅ Thành công!
w: 0.75
Data Processing...
# entities: 43474
# relations: 15
# edges: 305986
cuda                                             ← ✅ Đang dùng GPU!
```

## Nếu bạn không có sẵn các file embeddings

### Tạo embeddings từ đầu

#### 1. Poincaré Embeddings

Sử dụng công cụ như `gensim` hoặc tham khảo paper gốc để tạo Poincaré embeddings từ knowledge graph.

#### 2. Text Embeddings

**Option A: Dùng PubMedBERT (dễ nhất)**
```bash
cd ~/fussion-and-verify-in-BKG
python << 'EOF'
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Load entity names
train = pd.read_csv('fuselinker/suppkg/train.tsv', sep='\t', header=None)
valid = pd.read_csv('fuselinker/suppkg/valid.tsv', sep='\t', header=None)
test = pd.read_csv('fuselinker/suppkg/test.tsv', sep='\t', header=None)

entities = pd.concat([train[0], train[2], valid[0], valid[2], test[0], test[2]]).unique()
print(f"Total entities: {len(entities)}")

# Load PubMedBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Generate embeddings
embeddings = []
batch_size = 32

with torch.no_grad():
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        encoded = tokenizer.batch_encode_plus(
            batch.tolist(),
            padding=True,
            truncation=True,
            max_length=25,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

        if i % 1000 == 0:
            print(f"Processed {i}/{len(entities)} entities")

embeddings = np.vstack(embeddings)
np.save('engine/pubmedbert_pretrained_embeddings_768.npy', embeddings)
print(f"Saved embeddings with shape: {embeddings.shape}")
EOF
```

**Option B: Dùng SapBERT** (tốt hơn cho biomedical entities)
```bash
cd ~/fussion-and-verify-in-BKG
python generate_sapbert_embeddings.py \
    --data_dir fuselinker/suppkg \
    --output_file engine/sapbert_embeddings_768.npy
```

## Troubleshooting

### Lỗi: File not found

Kiểm tra lại đường dẫn:
```bash
cd ~/fussion-and-verify-in-BKG/fuselinker
ls -la ../engine/
```

### Lỗi: File shape mismatch

Kiểm tra shape của embeddings:
```python
import numpy as np
emb = np.load('../engine/medllama_pretrained_embeddings_4096.npy')
print(f"Shape: {emb.shape}")  # Should be (43474, 4096) for MedLLaMA
```

Số entities phải khớp với số entities trong dataset (43474 cho suppkg).

### Lỗi: Memory error

Nếu file quá lớn, giảm batch size hoặc dùng embeddings nhỏ hơn (768D thay vì 4096D).

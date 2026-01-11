# Giải pháp: Fix lỗi không load được embeddings

## Nguyên nhân

Code trong `main.py` dòng 15-16 tự động thêm `{args.data}/` trước đường dẫn embedding:

```python
text_embedding_path = f'{args.data}/{args.text_embedding_file}'
knowledge_embedding_path = f'{args.data}/{args.knowledge_embedding_file}'
```

Khi dùng relative path `../engine/...`, nó sẽ tìm file ở sai vị trí.

## Giải pháp 1: Dùng đường dẫn tuyệt đối (KHUYẾN NGHỊ)

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker

python main.py --data suppkg \
    --text_embedding_file ~/fussion-and-verify-in-BKG/engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file ~/fussion-and-verify-in-BKG/engine/poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --n_hidden 200 \
    --iterations 100 \
    --evaluate_every 50 \
    --w 0.75 \
    --use_cuda True
```

**Lưu ý:** Đường dẫn sẽ trở thành `suppkg/~/fussion...` nhưng Python sẽ normalize nó đúng.

## Giải pháp 2: Tạo symlink trong thư mục suppkg

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker/suppkg
ln -s ../../engine engine

# Sau đó dùng relative path
cd ~/fussion-and-verify-in-BKG/fuselinker
python main.py --data suppkg \
    --text_embedding_file engine/medllama_pretrained_embeddings_4096.npy \
    --knowledge_embedding_file engine/poincare_embeddings.npy \
    ...
```

## Giải pháp 3: Sửa code main.py (KHUYẾN NGHỊ NHẤT)

Thay đổi dòng 15-16 trong `main.py`:

```python
# CŨ:
text_embedding_path = f'{args.data}/{args.text_embedding_file}'
knowledge_embedding_path = f'{args.data}/{args.knowledge_embedding_file}'

# MỚI:
# Chỉ thêm args.data/ nếu path không phải absolute và không bắt đầu bằng ../
import os
if os.path.isabs(args.text_embedding_file) or args.text_embedding_file.startswith('../'):
    text_embedding_path = args.text_embedding_file
else:
    text_embedding_path = f'{args.data}/{args.text_embedding_file}'

if os.path.isabs(args.knowledge_embedding_file) or args.knowledge_embedding_file.startswith('../'):
    knowledge_embedding_path = args.knowledge_embedding_file
else:
    knowledge_embedding_path = f'{args.data}/{args.knowledge_embedding_file}'
```

Sau khi sửa, bạn có thể dùng relative path như ban đầu:
```bash
--text_embedding_file ../engine/medllama_pretrained_embeddings_4096.npy
```

## Kiểm tra embeddings đã tồn tại chưa

```bash
ls -lh ~/fussion-and-verify-in-BKG/engine/*.npy
```

Nếu thư mục `engine/` chưa tồn tại, tạo nó và copy các file embeddings vào.
